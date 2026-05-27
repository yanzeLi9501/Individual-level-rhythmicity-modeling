#!/usr/bin/env python3
r"""Cross-institution LGDI experiment: local retraining + local evaluation.

Questions:
  Q1. If we retrain XGBoost locally on each institution's own features,
      does LGDI become viable for cross-institution deployment?
  Q2. Can Pearson RDI + multi-group consensus serve as a COVID module?

Strategy:
  - For each external dataset with sufficient respiratory endpoints (MIMIC-IV,
    NWICU), train a local XGBoost model using ONLY that dataset's features.
  - Compute LGDI using the locally-trained model.
  - Evaluate LGDI against the dataset's own respiratory ground truth.
  - Compare with Pearson RDI performance on the same dataset.
  - For MIMIC: use influenza-coded admissions as endpoint (not FluNet calendar).
  - For NWICU: use COVID-coded admissions as endpoint.

Outputs:
  NC_revision/cross_institution_lgdi_results/
    mimic_localtrain_summary.json
    nwicu_localtrain_summary.json
    cross_institution_comparison.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "cross_institution_lgdi_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ───────────────────────────────────────────────────────────────
MIMIC_DIR = Path(r"data\mimic-iv-2.2\mimic-iv-2.2")
NWICU_DIR = BASE.parent / "external_data" / "physionet" / "nwicu-northwestern-icu" / "0.1.0" / "data" / "nw_hosp"

RANDOM_STATE = 20260516

# ── Comorbidity patterns (English ICD for MIMIC/NWICU) ──────────────────
COMORBIDITY_PATTERNS_EN = {
    "Cardiovascular": r"coronary|angina|myocardial|infarction|heart failure|cardiomyopathy|arrhythmia|atrial fibrillation|valvular|atherosclerosis|stent|I25|I21|I22|I50|I48|I35",
    "Hypertension": r"hypertension|I10|I11|I12|I13|I15",
    "Diabetes": r"diabetes|E08|E09|E10|E11|E13",
    "Cerebrovascular": r"stroke|cerebrovascular|cerebral infarction|intracerebral|I63|I61|I64",
    "Renal": r"renal|kidney|nephro|dialysis|CKD|N18|N19|N17",
    "Respiratory": r"COPD|asthma|pneumonia|pulmonary fibrosis|respiratory failure|bronchiectasis|J44|J45|J18|J84|J96",
}
GROUPS = list(COMORBIDITY_PATTERNS_EN.keys())
RESP = "Respiratory"

# Common lab columns to look for
LAB_CANDIDATES = {
    "lab_WBC": ["wbc", "white_blood", "leukocyte"],
    "lab_CRP": ["crp", "c_reactive"],
    "lab_HGB": ["hemoglobin", "hgb", "haemoglobin"],
    "lab_ALB": ["albumin"],
    "lab_CREA": ["creatinine", "creat"],
    "lab_GLU": ["glucose", "glu"],
    "lab_K": ["potassium", "^k$"],
    "lab_Na": ["sodium", "^na$"],
}


# ══════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════

def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower().strip(): c for c in frame.columns}
    for cand in candidates:
        pat = cand.lower()
        for cl, orig in cols_lower.items():
            if re.search(pat, cl):
                return orig
    return None


def load_mimic_iv() -> pd.DataFrame:
    """Load MIMIC-IV admissions with diagnoses (lightweight: skip labs for speed)."""
    hosp_dir = MIMIC_DIR / "hosp"
    if not hosp_dir.exists():
        raise FileNotFoundError(f"MIMIC-IV hosp not found: {hosp_dir}")

    print("  Loading admissions + diagnoses...")
    adm = pd.read_csv(hosp_dir / "admissions.csv.gz", low_memory=False)
    adm["admit_dt"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["discharge_dt"] = pd.to_datetime(adm["dischtime"], errors="coerce")

    diag = pd.read_csv(hosp_dir / "diagnoses_icd.csv.gz", low_memory=False)
    diag_text = diag.groupby("hadm_id")["icd_code"].apply(
        lambda x: " ".join(x.dropna().astype(str))
    ).reset_index(name="all_icd_codes")

    df = adm.merge(diag_text, on="hadm_id", how="left")
    df["all_icd_codes"] = df["all_icd_codes"].fillna("")
    df["subject_id"] = df["subject_id"].astype(str)

    # No lab values for this experiment (realistic cross-institution scenario)
    for lab_name in LAB_CANDIDATES:
        df[lab_name] = np.nan

    # Comorbidity + influenza flags
    for group, pattern in COMORBIDITY_PATTERNS_EN.items():
        df[group] = df["all_icd_codes"].str.contains(pattern, case=False, regex=True, na=False).astype(bool)

    flu_pattern = r"487|488|J09|J10|J11"
    df["is_influenza"] = df["all_icd_codes"].str.contains(flu_pattern, case=False, regex=True, na=False).astype(bool)
    df["is_covid"] = df["all_icd_codes"].str.contains(r"U07\.1|COVID|coronavirus", case=False, regex=True, na=False).astype(bool)

    df["los_days"] = (df["discharge_dt"] - df["admit_dt"]).dt.total_seconds() / 86400
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan

    df = df.sort_values(["subject_id", "admit_dt"]).reset_index(drop=True)
    df["visit_order"] = df.groupby("subject_id").cumcount()
    df["prev_discharge"] = df.groupby("subject_id")["discharge_dt"].shift(1)
    df["gap_days"] = (df["admit_dt"] - df["prev_discharge"]).dt.total_seconds() / 86400
    df.loc[(df["gap_days"] < 0) | (df["gap_days"] > 3650), "gap_days"] = np.nan
    df["year"] = df["admit_dt"].dt.year
    df["week_start"] = (df["admit_dt"] - pd.to_timedelta(df["admit_dt"].dt.weekday, unit="D")).dt.normalize()

    return df


def load_nwicu() -> pd.DataFrame:
    """Load NWICU admissions with diagnoses and labs."""
    if not NWICU_DIR.exists():
        raise FileNotFoundError(f"NWICU not found: {NWICU_DIR}")

    # Look for admissions.csv.gz (NWICU uses gzipped CSVs)
    adm_path = NWICU_DIR / "admissions.csv.gz"
    diag_path = NWICU_DIR / "diagnoses_icd.csv.gz"
    lab_path = NWICU_DIR / "labevents.csv.gz"
    pat_path = NWICU_DIR / "patients.csv.gz"

    if not adm_path.exists():
        raise FileNotFoundError(f"admissions.csv.gz not found in {NWICU_DIR}")

    print(f"  Loading admissions from {adm_path.name} ({adm_path.stat().st_size / 1e6:.0f} MB)...")
    adm = pd.read_csv(adm_path, low_memory=False)
    print(f"    Loaded {len(adm)} rows, columns: {list(adm.columns)[:15]}")

    # Find key columns
    id_col = find_column(adm, ["subject_id", "patient_id", "stay_id", "hadm_id"])
    admit_col = find_column(adm, ["admittime", "admit_dt", "admission"])
    discharge_col = find_column(adm, ["dischtime", "discharge_dt", "discharge"])
    diag_col_local = find_column(adm, ["diagnosis", "icd_code"])

    # If no diagnosis in admissions, load diagnoses_icd
    all_icd = pd.Series("", index=adm.index)
    if diag_path.exists() and id_col:
        print(f"  Loading diagnoses from {diag_path.name}...")
        diag = pd.read_csv(diag_path, low_memory=False)
        diag_id = find_column(diag, ["subject_id", "hadm_id"])
        diag_code = find_column(diag, ["icd_code", "icd9_code"])
        if diag_id and diag_code:
            diag_text = diag.groupby(diag_id)[diag_code].apply(
                lambda x: " ".join(x.dropna().astype(str))
            ).reset_index(name="all_icd")
            all_icd = adm[id_col].map(
                diag_text.set_index(diag_id)["all_icd"]
            ).fillna("")

    df = adm.copy()
    df["subject_id"] = df[id_col].astype(str) if id_col else df.index.astype(str)
    df["admit_dt"] = pd.to_datetime(df[admit_col], errors="coerce") if admit_col else pd.NaT
    if discharge_col:
        df["discharge_dt"] = pd.to_datetime(df[discharge_col], errors="coerce")
    else:
        df["discharge_dt"] = df["admit_dt"] + pd.Timedelta(days=7)
    df["all_icd_codes"] = all_icd
    if diag_col_local and diag_col_local in df.columns:
        df["all_icd_codes"] = df["all_icd_codes"].fillna("").astype(str) + " " + df[diag_col_local].fillna("").astype(str)

    df = df.dropna(subset=["admit_dt"]).copy()
    print(f"    {len(df)} rows with valid admit dates")

    # Comorbidity flags
    for group, pattern in COMORBIDITY_PATTERNS_EN.items():
        df[group] = df["all_icd_codes"].str.contains(pattern, case=False, regex=True, na=False).astype(bool)

    df["is_covid"] = df["all_icd_codes"].str.contains(r"U07\.1|COVID|coronavirus|covid", case=False, regex=True, na=False).astype(bool)

    # Labs
    for lab_name, patterns in LAB_CANDIDATES.items():
        lab_col = find_column(df, patterns)
        if lab_col:
            df[lab_name] = pd.to_numeric(df[lab_col], errors="coerce")
        else:
            df[lab_name] = np.nan

    # LOS
    df["los_days"] = (df["discharge_dt"] - df["admit_dt"]).dt.total_seconds() / 86400
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan

    df = df.sort_values(["subject_id", "admit_dt"]).reset_index(drop=True)
    df["visit_order"] = df.groupby("subject_id").cumcount()
    df["prev_discharge"] = df.groupby("subject_id")["discharge_dt"].shift(1)
    df["gap_days"] = (df["admit_dt"] - df["prev_discharge"]).dt.total_seconds() / 86400
    df.loc[(df["gap_days"] < 0) | (df["gap_days"] > 3650), "gap_days"] = np.nan
    df["year"] = df["admit_dt"].dt.year
    df["week_start"] = (df["admit_dt"] - pd.to_timedelta(df["admit_dt"].dt.weekday, unit="D")).dt.normalize()

    return df


# ══════════════════════════════════════════════════════════════════════════
#  XGBoost local training + LGDI
# ══════════════════════════════════════════════════════════════════════════

def build_seasonal_features(admit_dt: pd.Series) -> pd.DataFrame:
    month = admit_dt.dt.month.astype(float)
    dow = admit_dt.dt.weekday.astype(float)
    doy = admit_dt.dt.dayofyear.astype(float)
    return pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=admit_dt.index)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build leak-free features for local XGBoost training."""
    df = data.sort_values(["subject_id", "admit_dt"]).copy()
    g = df.groupby("subject_id", sort=False)

    df["visit_order"] = g.cumcount()
    df["first_admit_dt"] = g["admit_dt"].transform("min")
    df["days_since_first"] = (df["admit_dt"] - df["first_admit_dt"]).dt.total_seconds() / 86400
    df["prior_los_mean"] = g["los_days"].transform(lambda s: s.shift(1).expanding().mean())
    df["prior_los_std"] = g["los_days"].transform(lambda s: s.shift(1).expanding().std())
    df["prior_los_last"] = g["los_days"].shift(1)
    df["prior_gap_mean"] = g["gap_days"].transform(lambda s: s.shift(1).expanding().mean())
    df["prior_gap_std"] = g["gap_days"].transform(lambda s: s.shift(1).expanding().std())
    df["prior_gap_last"] = g["gap_days"].shift(1)
    df["next_admit_dt"] = g["admit_dt"].shift(-1)
    df["next_los_days"] = g["los_days"].shift(-1)
    df["next_gap_days"] = (df["next_admit_dt"] - df["discharge_dt"]).dt.total_seconds() / 86400
    df.loc[(df["next_gap_days"] < 0) | (df["next_gap_days"] > 3650), "next_gap_days"] = np.nan

    seasonal = build_seasonal_features(df["admit_dt"])
    for c in seasonal.columns:
        df[c] = seasonal[c].values

    feature_cols = [
        "visit_order", "days_since_first",
        "prior_los_mean", "prior_los_std", "prior_los_last",
        "prior_gap_mean", "prior_gap_std", "prior_gap_last",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    ]
    feature_cols.extend(GROUPS)
    for lab in LAB_CANDIDATES:
        if lab in df.columns:
            feature_cols.append(lab)

    for c in feature_cols:
        if c in GROUPS:
            df[c] = df[c].astype(bool).astype(float)
        elif c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    return df, feature_cols


def train_local_xgb(
    train_data: pd.DataFrame, feature_cols: list[str], target: str
) -> tuple[xgb.XGBRegressor, dict]:
    subset = train_data.dropna(subset=[target]).copy()
    audit = {"target": target, "n_train": int(len(subset)), "cv_r2": float("nan"), "cv_mae": float("nan")}

    if len(subset) < 200:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  random_state=RANDOM_STATE, tree_method="hist", n_jobs=4)
        if len(subset) > 0:
            X = subset[feature_cols].fillna(0.0).astype(float)
            model.fit(X, subset[target].astype(float))
        return model, audit

    groups = subset["subject_id"].fillna("").astype(str).values
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=max(2, n_splits))
    preds, truths = [], []
    for tr_idx, va_idx in gkf.split(subset[feature_cols], subset[target], groups=groups):
        X_tr = subset.iloc[tr_idx][feature_cols].fillna(0.0).astype(float)
        y_tr = subset.iloc[tr_idx][target].astype(float)
        X_va = subset.iloc[va_idx][feature_cols].fillna(0.0).astype(float)
        y_va = subset.iloc[va_idx][target].astype(float)
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              random_state=RANDOM_STATE, tree_method="hist", n_jobs=4)
        m.fit(X_tr, y_tr)
        preds.extend(m.predict(X_va).tolist())
        truths.extend(y_va.tolist())

    t_arr = np.asarray(truths, dtype=float)
    p_arr = np.asarray(preds, dtype=float)
    audit["cv_r2"] = float(r2_score(t_arr, p_arr)) if t_arr.var() > 1e-9 else float("nan")
    audit["cv_mae"] = float(mean_absolute_error(t_arr, p_arr))

    # Final model
    final = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              random_state=RANDOM_STATE, tree_method="hist", n_jobs=4)
    X_all = subset[feature_cols].fillna(0.0).astype(float)
    final.fit(X_all, subset[target].astype(float))
    return final, audit


def compute_local_lgdi(
    data: pd.DataFrame, feature_cols: list[str],
    los_model: xgb.XGBRegressor, gap_model: xgb.XGBRegressor,
    baseline_start: str, baseline_end: str, label: str
) -> dict:
    """Compute LGDI using locally-trained models, evaluate vs local respiratory endpoint."""
    df = data.copy()
    X = df[feature_cols].fillna(0.0).astype(float)
    df["pred_los"] = los_model.predict(X)
    df["pred_gap"] = gap_model.predict(X)
    df["resid_los"] = df["next_los_days"].fillna(df["pred_los"]) - df["pred_los"]
    df["resid_gap"] = df["next_gap_days"].fillna(df["pred_gap"]) - df["pred_gap"]

    # Baseline residual scale per group
    bl_mask = df["admit_dt"].between(baseline_start, baseline_end)
    baseline = df[bl_mask]
    scales = {}
    for g in GROUPS:
        gb = baseline[baseline[g]]
        scales[g] = {
            "los_mae_scale": float(gb["resid_los"].abs().mean()) if len(gb) > 10 else 1.0,
            "gap_mae_scale": float(gb["resid_gap"].abs().mean()) if len(gb) > 10 else 1.0,
        }
        if not np.isfinite(scales[g]["los_mae_scale"]) or scales[g]["los_mae_scale"] < 0.001:
            scales[g]["los_mae_scale"] = 1.0
        if not np.isfinite(scales[g]["gap_mae_scale"]) or scales[g]["gap_mae_scale"] < 0.001:
            scales[g]["gap_mae_scale"] = 1.0

    # Weekly LGDI
    df["week_start_dt"] = pd.to_datetime(df["week_start"])
    weeks = sorted(df["week_start_dt"].dropna().unique())
    results = []
    for anchor in weeks:
        start = anchor - pd.Timedelta(days=21)
        end = anchor + pd.Timedelta(days=6)
        window = df[(df["admit_dt"] >= start) & (df["admit_dt"] <= end)]
        if len(window) < 50:
            continue

        group_scores = {}
        for g in GROUPS:
            gw = window[window[g]]
            if len(gw) < 10:
                group_scores[g] = float("nan")
                continue
            los_signed = float(gw["resid_los"].mean() / max(scales[g]["los_mae_scale"], 0.001))
            gap_signed = float(-gw["resid_gap"].mean() / max(scales[g]["gap_mae_scale"], 0.001))
            group_scores[g] = (los_signed + gap_signed) / 2.0

        resp_score = group_scores.get(RESP, float("nan"))
        others = [v for g, v in group_scores.items() if g != RESP and np.isfinite(v)]
        mean_other = float(np.mean(others)) if others else float("nan")
        lgdi = resp_score - mean_other if np.isfinite(resp_score) and np.isfinite(mean_other) else float("nan")

        # Local respiratory endpoint
        n_flu = int(window["is_influenza"].sum()) if "is_influenza" in window.columns else 0
        n_covid = int(window["is_covid"].sum()) if "is_covid" in window.columns else 0

        results.append({
            "week_start": str(anchor.date()),
            "n_admissions": int(len(window)),
            "resp_score": resp_score,
            "mean_other": mean_other,
            "lgdi": lgdi,
            "n_influenza_admissions": n_flu,
            "n_covid_admissions": n_covid,
            "group_scores": group_scores,
        })

    timeline = pd.DataFrame(results)
    return {
        "label": label,
        "n_weeks": int(len(timeline)),
        "n_weeks_with_lgdi": int(timeline["lgdi"].notna().sum()),
        "lgdi_mean": float(timeline["lgdi"].mean()) if len(timeline) > 0 else float("nan"),
        "lgdi_std": float(timeline["lgdi"].std()) if len(timeline) > 0 else float("nan"),
        "resp_score_mean": float(timeline["resp_score"].mean()) if len(timeline) > 0 else float("nan"),
        "weekly_data": timeline.to_dict(orient="records")[:20],  # first 20 weeks as sample
    }


# ══════════════════════════════════════════════════════════════════════════
#  Pearson RDI (model-free) for comparison
# ══════════════════════════════════════════════════════════════════════════

def compute_pearson_rdi(
    data: pd.DataFrame, reference_admissions: pd.DataFrame, label: str
) -> dict:
    """Compute Pearson RDI using the reference admissions' profile."""
    # Find available numeric columns
    numeric_cols = []
    for col in data.columns:
        if col in ["los_days"] or col.startswith("lab_"):
            if data[col].notna().sum() > 100:
                numeric_cols.append(col)

    if len(numeric_cols) < 3:
        return {"label": label, "error": f"Only {len(numeric_cols)} numeric columns available"}

    # Baseline stats (all admissions)
    stats = {}
    for col in numeric_cols:
        vals = pd.to_numeric(data[col], errors="coerce").dropna()
        stats[col] = {"mean": float(vals.mean()), "std": float(vals.std()) or 1.0}

    # Reference profile
    ref_profile = []
    for col in numeric_cols:
        vals = pd.to_numeric(reference_admissions[col], errors="coerce").dropna()
        if len(vals) > 0:
            ref_profile.append((float(vals.mean()) - stats[col]["mean"]) / stats[col]["std"])
        else:
            ref_profile.append(0.0)
    ref = np.array(ref_profile)

    # Per-group correlations
    corrs = {}
    for g in GROUPS:
        gdata = data[data[g]]
        if len(gdata) < 50:
            corrs[g] = float("nan")
            continue
        gprof = []
        for col in numeric_cols:
            vals = pd.to_numeric(gdata[col], errors="coerce").dropna()
            if len(vals) > 0:
                gprof.append((float(vals.mean()) - stats[col]["mean"]) / stats[col]["std"])
            else:
                gprof.append(0.0)
        gprof_arr = np.array(gprof)
        norm_ref = np.linalg.norm(ref) or 1.0
        norm_g = np.linalg.norm(gprof_arr) or 1.0
        corrs[g] = float(np.dot(ref, gprof_arr) / (norm_ref * norm_g))

    others = [v for g, v in corrs.items() if g != RESP and np.isfinite(v)]
    mean_other = float(np.mean(others)) if others else float("nan")
    rdi = corrs.get(RESP, float("nan")) - mean_other if np.isfinite(corrs.get(RESP, float("nan"))) else float("nan")

    return {
        "label": label,
        "numeric_cols_used": numeric_cols,
        "n_reference_admissions": int(len(reference_admissions)),
        "group_correlations": corrs,
        "rdi": rdi,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Main experiment
# ══════════════════════════════════════════════════════════════════════════

def main():
    all_results = {}

    # ── MIMIC-IV: local XGBoost + LGDI ─────────────────────────────────
    print("=" * 70)
    print("MIMIC-IV: Local XGBoost → LGDI")
    print("=" * 70)
    try:
        mimic = load_mimic_iv()
        print(f"  Loaded {len(mimic)} admissions, {mimic['subject_id'].nunique()} patients")
        print(f"  Influenza-coded: {mimic['is_influenza'].sum()} admissions")

        # Prepare features
        mimic, feat_cols = prepare_features(mimic)
        available_feats = [c for c in feat_cols if c in mimic.columns]
        print(f"  Available features: {len(available_feats)} / {len(feat_cols)}")

        # Train local XGBoost
        bl_mask = mimic["admit_dt"].between("2008-01-01", "2016-12-31")
        train = mimic[bl_mask]

        los_model, los_audit = train_local_xgb(train, available_feats, "next_los_days")
        gap_model, gap_audit = train_local_xgb(train, available_feats, "next_gap_days")
        print(f"  Local LOS CV: R²={los_audit['cv_r2']:.4f}, MAE={los_audit['cv_mae']:.2f}")
        print(f"  Local Gap CV: R²={gap_audit['cv_r2']:.4f}, MAE={gap_audit['cv_mae']:.2f}")

        # Compute LGDI
        mimic_lgdi = compute_local_lgdi(
            mimic, available_feats, los_model, gap_model,
            "2008-01-01", "2016-12-31", "MIMIC-IV_local"
        )
        print(f"  LGDI weeks: {mimic_lgdi['n_weeks_with_lgdi']}/{mimic_lgdi['n_weeks']}")
        print(f"  LGDI mean±sd: {mimic_lgdi['lgdi_mean']:.4f}±{mimic_lgdi['lgdi_std']:.4f}")

        # Pearson RDI for comparison
        ref_mask = mimic["is_influenza"].astype(bool)
        ref_adm = mimic[ref_mask]
        mimic_pearson = compute_pearson_rdi(mimic[mimic["admit_dt"].between("2017-01-01", "2019-12-31")],
                                             ref_adm, "MIMIC-IV_Pearson")
        print(f"  Pearson RDI: {mimic_pearson.get('rdi', 'N/A')}")
        if "group_correlations" in mimic_pearson:
            for g, r in mimic_pearson["group_correlations"].items():
                print(f"    {g}: r={r:.3f}")

        all_results["MIMIC-IV_local_LGDI"] = {
            "los_cv": los_audit,
            "gap_cv": gap_audit,
            "lgdi_summary": {k: v for k, v in mimic_lgdi.items() if k != "weekly_data"},
            "pearson_rdi": mimic_pearson,
        }

        with open(OUT_DIR / "mimic_localtrain_summary.json", "w") as f:
            json.dump(all_results["MIMIC-IV_local_LGDI"], f, indent=2, default=str)

    except Exception as e:
        print(f"  ❌ MIMIC-IV failed: {e}")
        all_results["MIMIC-IV_local_LGDI"] = {"error": str(e)}

    # ── NWICU: local XGBoost + LGDI ────────────────────────────────────
    print("\n" + "=" * 70)
    print("NWICU: Local XGBoost → LGDI")
    print("=" * 70)
    try:
        nwicu = load_nwicu()
        print(f"  Loaded {len(nwicu)} rows, {nwicu['subject_id'].nunique()} patients")
        print(f"  COVID-coded: {nwicu['is_covid'].sum()} rows")

        nwicu, feat_cols_nw = prepare_features(nwicu)
        available_feats_nw = [c for c in feat_cols_nw if c in nwicu.columns]
        print(f"  Available features: {len(available_feats_nw)} / {len(feat_cols_nw)}")

        bl_mask_nw = nwicu["admit_dt"] <= nwicu["admit_dt"].quantile(0.3)  # earliest 30% as baseline
        train_nw = nwicu[bl_mask_nw]
        baseline_start_nw = str(train_nw["admit_dt"].min().date())
        baseline_end_nw = str(train_nw["admit_dt"].max().date())
        print(f"  Baseline: {baseline_start_nw} to {baseline_end_nw} ({len(train_nw)} rows)")

        if len(train_nw) >= 200:
            los_model_nw, los_audit_nw = train_local_xgb(train_nw, available_feats_nw, "next_los_days")
            gap_model_nw, gap_audit_nw = train_local_xgb(train_nw, available_feats_nw, "next_gap_days")
            print(f"  Local LOS CV: R²={los_audit_nw['cv_r2']:.4f}, MAE={los_audit_nw['cv_mae']:.2f}")
            print(f"  Local Gap CV: R²={gap_audit_nw['cv_r2']:.4f}, MAE={gap_audit_nw['cv_mae']:.2f}")

            nwicu_lgdi = compute_local_lgdi(
                nwicu, available_feats_nw, los_model_nw, gap_model_nw,
                "2018-01-01", "2019-12-31", "NWICU_local"
            )
            print(f"  LGDI weeks: {nwicu_lgdi['n_weeks_with_lgdi']}/{nwicu_lgdi['n_weeks']}")
            print(f"  LGDI mean±sd: {nwicu_lgdi['lgdi_mean']:.4f}±{nwicu_lgdi['lgdi_std']:.4f}")

            # Pearson RDI
            ref_mask_nw = nwicu["is_covid"].astype(bool)
            ref_adm_nw = nwicu[ref_mask_nw]
            nwicu_pearson = compute_pearson_rdi(
                nwicu[~ref_mask_nw], ref_adm_nw, "NWICU_Pearson"
            )
            print(f"  Pearson RDI: {nwicu_pearson.get('rdi', 'N/A')}")
            if "group_correlations" in nwicu_pearson:
                for g, r in nwicu_pearson["group_correlations"].items():
                    print(f"    {g}: r={r:.3f}")

            all_results["NWICU_local_LGDI"] = {
                "los_cv": los_audit_nw,
                "gap_cv": gap_audit_nw,
                "lgdi_summary": {k: v for k, v in nwicu_lgdi.items() if k != "weekly_data"},
                "pearson_rdi": nwicu_pearson,
            }
        else:
            print(f"  ❌ Insufficient training data: {len(train_nw)} rows")
            all_results["NWICU_local_LGDI"] = {"error": f"Insufficient training data: {len(train_nw)} rows"}

        with open(OUT_DIR / "nwicu_localtrain_summary.json", "w") as f:
            json.dump(all_results["NWICU_local_LGDI"], f, indent=2, default=str)

    except Exception as e:
        print(f"  ❌ NWICU failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["NWICU_local_LGDI"] = {"error": str(e)}

    # ── Cross-institution comparison ────────────────────────────────────
    # Load existing results for reference
    existing = {}
    cardiac_lgdi_path = BASE / "lgdi_results" / "lgdi_summary.json"
    if cardiac_lgdi_path.exists():
        existing["cardiac_expanded"] = json.loads(cardiac_lgdi_path.read_text())

    comparison = {
        "experiment": "cross_institution_local_retraining",
        "question": "Does local XGBoost retraining rescue LGDI for cross-institution deployment?",
        "results": all_results,
        "existing_cardiac_baseline": {
            "los_cv_r2": existing.get("cardiac_expanded", {}).get("cv_audit", [{}])[0].get("cv_r2") if existing.get("cardiac_expanded", {}).get("cv_audit") else None,
            "gap_cv_r2": existing.get("cardiac_expanded", {}).get("cv_audit", [{}])[1].get("cv_r2") if existing.get("cardiac_expanded", {}).get("cv_audit") else None,
        } if existing else {},
    }

    with open(OUT_DIR / "cross_institution_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-INSTITUTION SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<20} {'LOS CV R²':>10} {'Gap CV R²':>10} {'LGDI mean':>10} {'Pearson RDI':>12}")
    print("-" * 65)
    # Cardiac baseline
    cardiac_cv = existing.get("cardiac_expanded", {}).get("cv_audit", [])
    print(f"{'Cardiac (existing)':<20} {cardiac_cv[0].get('cv_r2',0) if len(cardiac_cv)>0 else 'N/A':>10} "
          f"{cardiac_cv[1].get('cv_r2',0) if len(cardiac_cv)>1 else 'N/A':>10} {'—':>10} {'—':>12}")

    for key, res in all_results.items():
        name = key.replace("_local_LGDI", "")
        if "error" in res:
            print(f"{name:<20} {'ERROR':>10} {'ERROR':>10}")
            continue
        los_r2 = res.get("los_cv", {}).get("cv_r2", float("nan"))
        gap_r2 = res.get("gap_cv", {}).get("cv_r2", float("nan"))
        lgdi_m = res.get("lgdi_summary", {}).get("lgdi_mean", float("nan"))
        rdi = res.get("pearson_rdi", {}).get("rdi", float("nan"))
        print(f"{name:<20} {los_r2:>10.4f} {gap_r2:>10.4f} {lgdi_m:>10.4f} {rdi:>12.4f}")

    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
