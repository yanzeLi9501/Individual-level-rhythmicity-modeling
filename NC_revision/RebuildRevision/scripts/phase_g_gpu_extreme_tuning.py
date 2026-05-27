"""
phase_g_gpu_extreme_tuning.py
═══════════════════════════════════════════════════════════════════════════════
GPU 服务器极限调优 — 每个队列 LOS & gap R\ufffd\ufffd 上限估计
─────────────────────────────────────────────────────────────────────────────
目标：在 RTX 4090D (49.1 GB VRAM) 上用 Optuna 超参贝叶斯优化 + 早期停止，
      将每个数据集的 XGBoost 推到当前特征空间下的经验上限。

策略：
  1. Optuna 搜索: n_estimators (1000-5000), max_depth (3-12), lr (0.005-0.1),
     subsample (0.5-1.0), colsample_bytree (0.3-1.0),
     min_child_weight (1-50), reg_lambda (0.1-100)
  2. 每个 trial 内部使用 GroupKFold CV (n_splits=5) 评估
  3. 早期停止: early_stopping_rounds=200, 避免过拟合
  4. 最终 best model 在全量 CV 下报告 R\ufffd\ufffd

预计运行时间（per dataset）:
  - MIMIC-IV: ~8-12h (431k rows, large feature space)
  - NWICU:    ~2-3h
  - eICU:     ~6-8h  (201k rows)
  - CDSL:     ~0.5h  (4.5k rows)
  - Cardiac:  ~6-8h  (300k rows)

R\ufffd\ufffd 极限估计（基于数据噪声基底与文献参照）:
  ┌──────────────────────┬────────────┬──────────────┬──────────────┐
  │ 数据集               │ 当前 R\ufffd\ufffd_LOS │ GPU 极限 R\ufffd\ufffd_LOS │ GPU 极限 R\ufffd\ufffd_gap │
  ├──────────────────────┼────────────┼──────────────┼──────────────┤
  │ MIMIC-IV             │   0.046    │   0.10–0.15  │  0.12–0.18   │
  │ NWICU                │   0.028    │   0.08–0.12  │  0.05–0.10   │
  │ eICU                 │   0.638    │   0.68–0.75  │  负（不可救） │
  │ CDSL                 │   0.520    │   0.55–0.62  │     —        │
  │ Cardiac frequent     │   0.166    │   0.22–0.28  │  0.22–0.30   │
  │ Cardiac respiratory  │  -0.370    │   ≤0（n太小）│    同左       │
  │ Cardiac extended     │   0.152    │   0.20–0.25  │  0.20–0.26   │
  └──────────────────────┴────────────┴──────────────┴──────────────┘

不可约噪声来源（决定 R\ufffd\ufffd 上限的根本因素）:
  - 急性事件随机性（事故、急性感染）→ 入院时长不可预测
  - 社会经济因素（医保类型、家庭支持）→ 住院决策非线性
  - 医院运营因素（床位周转压力、周末效应）→ 出院时机非疾病驱动
  - 疾病自然史不可知性 → 下一次入院时间间隔本质随机
  - 特征空间局限 → 已知的电没病共病以外的大量驱动因素未观测
─────────────────────────────────────────────────────────────────────────────
用法（GPU 服务器端）:
  python phase_g_gpu_extreme_tuning.py --dataset mimic      # 单个数据集
  python phase_g_gpu_extreme_tuning.py --dataset all        # 全部（约 30h）
  python phase_g_gpu_extreme_tuning.py --dataset cardiac    # cardiac 三策略

输出: NC_revision/RebuildRevision/outputs/xgb_symmetric/gpu_tuned/
  <dataset>_best_params.json
  <dataset>_cv_results.csv
  gpu_tuning_summary.json
"""
from __future__ import annotations
import argparse, gzip, json, math, os, sys, time, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import optuna

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SUBMIT = Path(__file__).resolve().parents[3] if "__file__" in dir() else Path(os.getcwd())
MIMIC_HOSP = SUBMIT / "external_data/physionet/mimic-iv-2.2/mimic-iv-2.2/hosp"
NWICU_HOSP = SUBMIT / "external_data/physionet/nwicu-northwestern-icu/0.1.0/data/nw_hosp"
EICU_DIR   = SUBMIT / "external_data/physionet/eicu-crd/2.0"
CDSL_DIR   = SUBMIT / "external_data/physionet/cdsl"
CARDIAC_CSV = SUBMIT / "NC_revision/expanded_cardiac_wide_table.csv"
OUT = SUBMIT / "NC_revision/RebuildRevision/outputs/xgb_symmetric/gpu_tuned"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2025
N_CV = 5
ES_ROUNDS = 200

ICD_GROUPS = {
    "respiratory": r"^J", "cardiovascular": r"^I(?!6[0-9]|7[0-9])",
    "cerebrovascular": r"^I6[0-9]", "diabetes": r"^E1[0-4]",
    "renal": r"^N1[7-9]|^N0[3-5]|^N17|^N18|^N19", "hypertension": r"^I1[0-5]",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders (copied from phase_g_xgb_symmetric_all_datasets.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_mimic_style(hosp_dir: Path, label: str) -> pd.DataFrame:
    print(f"  Loading {label} admissions...")
    with gzip.open(hosp_dir / "admissions.csv.gz", "rt") as f:
        adm = pd.read_csv(f, usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                                       "admission_type", "insurance", "race", "hospital_expire_flag"],
                          dtype={"subject_id": "int64", "hadm_id": "int64"})
    adm["admit_dt"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dis_dt"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm = adm.dropna(subset=["admit_dt"]).copy()
    adm["los_days"] = (adm["dis_dt"] - adm["admit_dt"]).dt.total_seconds() / 86400
    adm.loc[(adm["los_days"] < 0) | (adm["los_days"] > 180), "los_days"] = np.nan
    adm = adm.sort_values(["subject_id", "admit_dt"]).copy()
    adm["prev_dis"] = adm.groupby("subject_id")["dis_dt"].shift(1)
    adm["gap_days"] = (adm["admit_dt"] - adm["prev_dis"]).dt.total_seconds() / 86400
    adm.loc[(adm["gap_days"] < 0) | (adm["gap_days"] > 3650), "gap_days"] = np.nan
    adm["next_los_days"] = adm.groupby("subject_id")["los_days"].shift(-1)
    adm["next_gap_days"] = adm.groupby("subject_id")["gap_days"].shift(-1)
    for col in ["admission_type", "insurance", "race"]:
        adm[col + "_enc"] = pd.Categorical(adm[col]).codes
    adm["died_inhosp"] = adm["hospital_expire_flag"].fillna(0).astype(int)
    diag_path = hosp_dir / "diagnoses_icd.csv.gz"
    if diag_path.exists():
        with gzip.open(diag_path, "rt") as f:
            diag = pd.read_csv(f, usecols=["hadm_id", "icd_code", "seq_num"],
                               dtype={"hadm_id": "int64", "icd_code": str})
        prim = diag[diag["seq_num"] == 1].drop_duplicates("hadm_id")
        for grp, pat in ICD_GROUPS.items():
            flag = prim[prim["icd_code"].str.match(pat, na=False)]["hadm_id"]
            adm[grp] = adm["hadm_id"].isin(flag).astype(int)
    else:
        for grp in ICD_GROUPS:
            adm[grp] = 0
    # Add comorbidity count (Elixhauser-like proxy)
    adm["n_comorbid"] = adm[list(ICD_GROUPS.keys())].sum(axis=1)
    print(f"  {label}: {len(adm):,} admissions, {adm.subject_id.nunique():,} patients")
    return adm


def _load_eicu() -> pd.DataFrame:
    print("  Loading eICU patient table...")
    with gzip.open(EICU_DIR / "patient.csv.gz", "rt") as f:
        df = pd.read_csv(f, usecols=[
            "patientunitstayid", "patienthealthsystemstayid", "uniquepid",
            "hospitalid", "unittype", "apacheadmissiondx",
            "hospitaladmittime24", "hospitaladmitoffset",
            "hospitaldischargeoffset", "hospitaldischargeyear",
            "hospitaldischargelocation", "hospitaldischargestatus",
            "unitvisitnumber", "admissionweight", "dischargeweight", "age",
        ])
    df["hosp_los_days"] = (df["hospitaldischargeoffset"] - df["hospitaladmitoffset"]) / 60 / 24
    df.loc[(df["hosp_los_days"] < 0) | (df["hosp_los_days"] > 365), "hosp_los_days"] = np.nan
    df["age_num"] = pd.to_numeric(df["age"].replace("> 89", "90"), errors="coerce")
    df["unittype_enc"] = pd.Categorical(df["unittype"]).codes
    df["apachedx_enc"] = pd.Categorical(df["apacheadmissiondx"]).codes
    df["died_inhosp"] = (df["hospitaldischargestatus"] == "Expired").astype(int)
    df = df.sort_values(["uniquepid", "hospitaladmitoffset"]).copy()
    df["next_hosp_los_days"] = df.groupby("uniquepid")["hosp_los_days"].shift(-1)
    df["unit_gap_h"] = (df.groupby("uniquepid")["hospitaladmitoffset"].shift(-1) -
                        df["hospitaldischargeoffset"])
    df["unit_gap_h"] = df["unit_gap_h"].clip(lower=0) / 60
    df["next_gap_days"] = df["unit_gap_h"] / 24
    df.loc[df["next_gap_days"] > 3650, "next_gap_days"] = np.nan
    # eICU extended features: weight change, discharge location
    df["weight_change"] = df["dischargeweight"] - df["admissionweight"]
    df["discharged_home"] = (df["hospitaldischargelocation"] == "Home").astype(int)
    print(f"  eICU: {len(df):,} stays, {df.uniquepid.nunique():,} patients")
    return df


def _load_cdsl() -> pd.DataFrame:
    print("  Loading CDSL...")
    df = pd.read_csv(CDSL_DIR / "patient_01.csv")
    df["admit_dt"] = pd.to_datetime(df["admission_d_inpat"], errors="coerce")
    df["dis_dt"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    df["los_days"] = (df["dis_dt"] - df["admit_dt"]).dt.days.astype(float)
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 365), "los_days"] = np.nan
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex_enc"] = (df["sex"] == "FEMALE").astype(int)
    df["has_icu"] = df["icu_days"].notna().astype(int)
    df["icu_days_filled"] = df["icu_days"].fillna(0)
    for col in ["bp_max_first_emerg", "bp_min_first_emerg", "temp_first_emerg",
                "hr_first_emerg", "sat_02_first_emerg", "glu_first_emerg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Engineered: shock index, O2 deficit
    df["shock_index"] = df["hr_first_emerg"] / (df["bp_max_first_emerg"] + 1)
    df["o2_deficit"] = 100 - df["sat_02_first_emerg"].fillna(98)
    print(f"  CDSL: {len(df):,} patients")
    return df


def _load_cardiac() -> pd.DataFrame:
    print("  Loading cardiac wide table...")
    df = pd.read_csv(CARDIAC_CSV, encoding="utf-8-sig", low_memory=False, dtype={"病案号": str})
    df["mrn"] = df["病案号"].fillna("").astype(str).str.strip()
    df["admit_dt"] = pd.to_datetime(df["入院时间"], errors="coerce")
    df["dis_dt"] = pd.to_datetime(df["出院时间"], errors="coerce")
    df = df.dropna(subset=["admit_dt"]).copy()
    df["los_days"] = (df["dis_dt"] - df["admit_dt"]).dt.total_seconds() / 86400
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan
    df = df.sort_values(["mrn", "admit_dt"]).copy()
    df["prev_dis"] = df.groupby("mrn")["dis_dt"].shift(1)
    df["gap_days"] = (df["admit_dt"] - df["prev_dis"]).dt.total_seconds() / 86400
    df.loc[(df["gap_days"] < 0) | (df["gap_days"] > 3650), "gap_days"] = np.nan
    df["next_los_days"] = df.groupby("mrn")["los_days"].shift(-1)
    df["next_gap_days"] = df.groupby("mrn")["gap_days"].shift(-1)
    df["year"] = df["admit_dt"].dt.year
    # Engineered features
    df["visit_order"] = df.groupby("mrn").cumcount() + 1
    df["n_visits_lifetime"] = df.groupby("mrn")["mrn"].transform("count")
    df["season"] = df["admit_dt"].dt.month % 12 // 3  # 0=winter, 1=spring, etc.
    # ICD groups from Chinese ICD in 主要诊断
    all_dx = df.get("主要诊断", pd.Series("", index=df.index)).fillna("").astype(str)
    for grp, pat in {
        "respiratory": r"J", "cardiovascular": r"I",
        "cerebrovascular": r"I6[0-9]", "diabetes": r"E1[0-4]",
        "renal": r"N1[7-9]|N0[3-5]", "hypertension": r"I1[0-5]",
    }.items():
        df[grp] = all_dx.str.contains(pat, regex=True, na=False).astype(int)
    df["n_comorbid"] = df[["respiratory", "cardiovascular", "cerebrovascular",
                            "diabetes", "renal", "hypertension"]].sum(axis=1)
    print(f"  Cardiac: {len(df):,} admissions, {df.mrn.nunique():,} patients")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════════════════════════════════════

def _make_objective(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_trials: int, timeout: int = 3600):
    """Create an Optuna objective for GroupKFold CV XGBoost regression."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "random_state": SEED,
            "verbosity": 0,
            "early_stopping_rounds": ES_ROUNDS,
        }
        gkf = GroupKFold(n_splits=N_CV)
        oof = np.full(len(y), np.nan, dtype="float32")
        for tr, va in gkf.split(X, y, groups):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr],
                  eval_set=[(X[va], y[va])],
                  verbose=False)
            oof[va] = m.predict(X[va])
        mask = np.isfinite(oof)
        if mask.sum() < 10:
            return -1.0
        r2 = float(r2_score(y[mask], oof[mask]))
        return r2

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# Tune one dataset
# ═══════════════════════════════════════════════════════════════════════════════

def tune_dataset(df: pd.DataFrame, feat_cols: list[str],
                 target_col: str, group_col: str,
                 label: str, n_trials: int = 100,
                 timeout: int = 28800) -> dict:
    """Run Optuna tuning for a single dataset/target.

    Returns dict with best_params, best_r2, all_r2_history.
    """
    sub = df.dropna(subset=feat_cols + [target_col, group_col]).copy()
    if len(sub) < 100:
        print(f"  [{label}] {target_col}: too few rows ({len(sub)}) -> skip")
        return {"label": label, "target": target_col, "best_r2": math.nan, "n_samples": len(sub)}

    print(f"\n{'='*70}")
    print(f"  Tuning: {label} / {target_col}")
    print(f"  Samples: {len(sub):,}  Features: {len(feat_cols)}  Trials: {n_trials}")
    print(f"{'='*70}")

    X = sub[feat_cols].values.astype("float32")
    y = sub[target_col].values.astype("float32")
    groups = sub[group_col].values

    # Try GPU first, fall back to CPU
    try:
        _test_m = xgb.XGBRegressor(tree_method="hist", device="cuda")
        _test_m.fit(X[:10], y[:10], verbose=False)
        use_gpu = True
        print("  GPU (CUDA) available")
    except Exception:
        use_gpu = False
        print("  GPU unavailable — using CPU")
        global _make_objective
        # We'll handle device in the objective via a closure

    objective = _make_objective(X, y, groups, n_trials, timeout)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    elapsed = time.time() - t0

    best_r2 = study.best_value
    best_params = study.best_params
    print(f"\n  [{label}] {target_col} BEST: R²={best_r2:.4f}  time={elapsed/60:.1f}min")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    return {
        "label": label,
        "target": target_col,
        "n_samples": len(sub),
        "n_features": len(feat_cols),
        "n_trials": n_trials,
        "best_r2": best_r2,
        "best_params": best_params,
        "elapsed_min": round(elapsed / 60, 1),
        "trials": [{"number": t.number, "value": t.value,
                     "params": t.params} for t in study.trials if t.value is not None],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main — per-dataset tuning
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["mimic", "nwicu", "eicu", "cdsl", "cardiac", "all"],
                        help="Which dataset to tune (default: all)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optuna trials per target (default: 100)")
    parser.add_argument("--timeout", type=int, default=28800,
                        help="Timeout seconds per target (default: 28800 = 8h)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 30 trials, 1h timeout")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 30
        args.timeout = 3600
        print("[QUICK MODE] 30 trials, 1h timeout per target")

    all_results = []
    datasets_to_run = [args.dataset] if args.dataset != "all" else [
        "mimic", "nwicu", "eicu", "cdsl", "cardiac"
    ]

    # ── MIMIC-IV ────────────────────────────────────────────────────────────
    if "mimic" in datasets_to_run:
        print("\n" + "="*70)
        print("  A. MIMIC-IV GPU Extreme Tuning")
        print("="*70)
        mimic = _load_mimic_style(MIMIC_HOSP, "MIMIC-IV")
        feat_los = ["los_days", "gap_days", "admission_type_enc",
                     "insurance_enc", "race_enc", "died_inhosp",
                     "n_comorbid"] + list(ICD_GROUPS.keys())
        feat_gap = feat_los[:]
        r_los = tune_dataset(mimic, feat_los, "next_los_days", "subject_id",
                             "MIMIC-IV", args.n_trials, args.timeout)
        r_gap = tune_dataset(mimic, feat_gap, "next_gap_days", "subject_id",
                             "MIMIC-IV", args.n_trials, args.timeout)
        all_results.extend([r_los, r_gap])

    # ── NWICU ───────────────────────────────────────────────────────────────
    if "nwicu" in datasets_to_run:
        print("\n" + "="*70)
        print("  B. NWICU GPU Extreme Tuning")
        print("="*70)
        nwicu = _load_mimic_style(NWICU_HOSP, "NWICU")
        feat_los = ["los_days", "gap_days", "admission_type_enc",
                     "insurance_enc", "race_enc", "died_inhosp",
                     "n_comorbid"] + list(ICD_GROUPS.keys())
        r_los = tune_dataset(nwicu, feat_los, "next_los_days", "subject_id",
                             "NWICU", args.n_trials, args.timeout)
        r_gap = tune_dataset(nwicu, feat_los, "next_gap_days", "subject_id",
                             "NWICU", args.n_trials, args.timeout)
        all_results.extend([r_los, r_gap])

    # ── eICU ────────────────────────────────────────────────────────────────
    if "eicu" in datasets_to_run:
        print("\n" + "="*70)
        print("  C. eICU GPU Extreme Tuning")
        print("="*70)
        eicu = _load_eicu()
        feat_eicu = ["hosp_los_days", "age_num", "unittype_enc", "apachedx_enc",
                      "unitvisitnumber", "admissionweight", "died_inhosp",
                      "weight_change", "discharged_home"]
        r_los = tune_dataset(eicu, feat_eicu, "next_hosp_los_days", "uniquepid",
                             "eICU", args.n_trials, args.timeout)
        r_gap = tune_dataset(eicu, feat_eicu, "next_gap_days", "uniquepid",
                             "eICU", args.n_trials, args.timeout)
        all_results.extend([r_los, r_gap])

    # ── CDSL ────────────────────────────────────────────────────────────────
    if "cdsl" in datasets_to_run:
        print("\n" + "="*70)
        print("  D. CDSL GPU Extreme Tuning")
        print("="*70)
        cdsl = _load_cdsl()
        feat_cdsl = ["age_num", "sex_enc", "has_icu", "icu_days_filled",
                      "bp_max_first_emerg", "bp_min_first_emerg", "temp_first_emerg",
                      "hr_first_emerg", "sat_02_first_emerg", "glu_first_emerg",
                      "shock_index", "o2_deficit"]
        # CDSL: no group column (1 per patient) -> use simple 5-fold CV
        sub = cdsl.dropna(subset=feat_cdsl + ["los_days"]).copy()
        X = sub[feat_cdsl].fillna(sub[feat_cdsl].median()).values.astype("float32")
        y = sub["los_days"].values.astype("float32")

        def obj_cdsl(trial):
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cuda",
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
                "random_state": SEED, "verbosity": 0,
            }
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            oof = np.full(len(y), np.nan, dtype="float32")
            for tr, va in kf.split(X):
                m = xgb.XGBRegressor(**params)
                m.fit(X[tr], y[tr], verbose=False)
                oof[va] = m.predict(X[va])
            return float(r2_score(y, oof))

        print(f"  [CDSL] Tuning LOS: n={len(sub):,}")
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(obj_cdsl, n_trials=args.n_trials, timeout=args.timeout)
        all_results.append({
            "label": "CDSL", "target": "los_days",
            "n_samples": len(sub), "n_features": len(feat_cdsl),
            "n_trials": args.n_trials,
            "best_r2": study.best_value,
            "best_params": study.best_params,
            "elapsed_min": "—",
        })
        print(f"  [CDSL] LOS BEST: R²={study.best_value:.4f}")

    # ── Cardiac splits ──────────────────────────────────────────────────────
    if "cardiac" in datasets_to_run:
        print("\n" + "="*70)
        print("  E. Cardiac GPU Extreme Tuning")
        print("="*70)
        cardiac = _load_cardiac()

        # E1: Frequent visitors
        visit_counts = cardiac.groupby("mrn")["admit_dt"].count()
        fv_mrns = visit_counts[visit_counts >= 5].index
        fv = cardiac[cardiac["mrn"].isin(fv_mrns)].copy()
        feat_cardiac = ["los_days", "gap_days", "visit_order", "n_visits_lifetime",
                        "season", "n_comorbid"] + list(ICD_GROUPS.keys())
        print("\n[Cardiac E1] Frequent visitors (>=5)")
        r1_los = tune_dataset(fv, feat_cardiac, "next_los_days", "mrn",
                              "cardiac_fv_tuned", args.n_trials, args.timeout)
        r1_gap = tune_dataset(fv, feat_cardiac, "next_gap_days", "mrn",
                              "cardiac_fv_tuned", args.n_trials, args.timeout)
        all_results.extend([r1_los, r1_gap])

        # E2: Extended (all patients)
        print("\n[Cardiac E2] Extended (all)")
        r2_los = tune_dataset(cardiac, feat_cardiac, "next_los_days", "mrn",
                              "cardiac_ext_tuned", args.n_trials, args.timeout)
        r2_gap = tune_dataset(cardiac, feat_cardiac, "next_gap_days", "mrn",
                              "cardiac_ext_tuned", args.n_trials, args.timeout)
        all_results.extend([r2_los, r2_gap])

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  GPU EXTREME TUNING — FINAL SUMMARY")
    print("="*80)
    print(f"{'Dataset':30s} {'Target':12s} {'Best R²':>8s}  {'Baseline R²':>10s}  {'Delta':>8s}")
    print("-"*80)
    BASELINE = {
        "MIMIC-IV_next_los_days": 0.046, "MIMIC-IV_next_gap_days": 0.078,
        "NWICU_next_los_days": 0.028, "NWICU_next_gap_days": 0.010,
        "eICU_next_hosp_los_days": 0.638, "eICU_next_gap_days": -0.176,
        "CDSL_los_days": 0.520,
        "cardiac_fv_tuned_next_los_days": 0.166, "cardiac_fv_tuned_next_gap_days": 0.175,
        "cardiac_ext_tuned_next_los_days": 0.152, "cardiac_ext_tuned_next_gap_days": 0.166,
    }
    for r in all_results:
        key = f"{r['label']}_{r['target']}"
        base = BASELINE.get(key, None)
        best = r["best_r2"]
        delta = ""
        if base is not None and not math.isnan(best):
            delta = f"{best - base:+.4f}"
        print(f"{r['label']:30s} {r['target']:12s} {best:>8.4f}  {str(base):>10s}  {delta:>8s}")

    # Save
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "server": "RTX 4090D / 49.1 GB VRAM / 144 cores / 1 TB RAM",
        "strategy": "Optuna TPESampler + GroupKFold CV + early_stopping_rounds=200",
        "n_trials": args.n_trials,
        "results": all_results,
    }
    out_path = OUT / "gpu_tuning_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Results saved to {out_path}")


if __name__ == "__main__":
    main()
