"""
phase_g_xgb_symmetric_all_datasets.py
═══════════════════════════════════════════════════════════════════════════════
WHU-symmetric XGBoost regression + LGDI pipeline applied to every available
dataset, using each dataset's own native features.

Strategy (mirrors phase_b_lgdi_flu_ref.py):
  1. Load dataset admissions + diagnoses (where available)
  2. Derive LOS and inter-admission gap for repeat patients
  3. Build ICD/feature groups natively available per dataset
  4. XGBoost GroupKFold-CV to predict next_LOS and next_gap_days
  5. Compute baseline residuals → rolling 4-week LGDI windows
  6. Report CV R², MAE, valid window count
  7. For eICU (real years 2014-2015): compare LGDI against FluNet
  8. Cardiac special splits: frequent visitors, respiratory-focus, extended monitor

Datasets:
  A. MIMIC-IV          (431k admissions; shifted years — no FluNet comparison)
  B. NWICU             (61k admissions; shifted years — no FluNet comparison)
  C. eICU-CRD          (200k ICU stays; real years 2014-2015 — FluNet possible)
  D. CDSL              (4.5k admissions; 1-per-patient — LOS prediction only)
  E. Cardiac splits:
      E1. Frequent visitors (≥5 admissions/patient)
      E2. Respiratory-group focus (J00-J99 coded)
      E3. Extended monitor (2019-2024 COVID-era extension)

Outputs: NC_revision/RebuildRevision/outputs/xgb_symmetric/
  <dataset>_xgb_performance.csv
  <dataset>_lgdi_timeline.csv
  xgb_symmetric_summary.json
"""
from __future__ import annotations
import gzip, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SUBMIT = Path(__file__).resolve().parents[3]
MIMIC_HOSP    = Path("data/mimic-iv-2.2/mimic-iv-2.2/hosp")
NWICU_HOSP    = SUBMIT / "external_data/physionet/nwicu-northwestern-icu/0.1.0/data/nw_hosp"
EICU_DIR      = SUBMIT / "external_data/physionet/eicu-crd/2.0"
CDSL_DIR      = SUBMIT / "external_data/physionet/cdsl"
CARDIAC_CSV   = SUBMIT / "NC_revision/expanded_cardiac_wide_table.csv"
FLUNET_CSV    = SUBMIT / "external_data/flunet/flunet_china_2009_2024.csv"
FLUNET_GLOBAL = SUBMIT / "external_data/flunet/flunet_global_2009_2024.csv"
OUT           = SUBMIT / "NC_revision/RebuildRevision/outputs/xgb_symmetric"
OUT.mkdir(parents=True, exist_ok=True)

# ─── XGBoost config (mirrors phase_b_lgdi_flu_ref.py) ────────────────────────
XGB_PARAMS = dict(
    objective="reg:squarederror",
    tree_method="hist",
    device="cuda",
    n_estimators=800,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=3.0,
    random_state=2025,
    n_jobs=-1,
    verbosity=0,
)
N_SPLITS     = 5
MIN_FRAME    = 30          # minimum admissions per 4-week window
SEED         = 2025

# ─── ICD group patterns ───────────────────────────────────────────────────────
ICD_GROUPS = {
    "respiratory":    r"^J",
    "cardiovascular": r"^I(?!6[0-9]|7[0-9])",
    "cerebrovascular": r"^I6[0-9]",
    "diabetes":       r"^E1[0-4]",
    "renal":          r"^N1[7-9]|^N0[3-5]|^N17|^N18|^N19",
    "hypertension":   r"^I1[0-5]",
}

# ═══════════════════════════════════════════════════════════════════════════════
# ── Common utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _xgb_cv_regress(df: pd.DataFrame, feat_cols: list[str],
                    target_col: str, group_col: str,
                    label: str) -> tuple[float, float, pd.DataFrame]:
    """GroupKFold XGBoost regression; returns (cv_r2, cv_mae, enriched_df)."""
    sub = df.dropna(subset=feat_cols + [target_col, group_col]).copy()
    if len(sub) < N_SPLITS * 20:
        print(f"    [{label}] {target_col}: too few rows ({len(sub)}) → skip")
        df["pred_" + target_col] = np.nan
        df["resid_" + target_col] = np.nan
        return math.nan, math.nan, df
    X = sub[feat_cols].values.astype("float32")
    y = sub[target_col].values.astype("float32")
    groups = sub[group_col].values
    oof = np.full(len(sub), np.nan, dtype="float32")
    gkf = GroupKFold(n_splits=N_SPLITS)
    try:
        params = {**XGB_PARAMS}
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr], verbose=False)
            oof[va] = m.predict(X[va])
    except Exception:
        # Fall back to CPU if CUDA unavailable
        params = {**XGB_PARAMS, "device": "cpu"}
        oof = np.full(len(sub), np.nan, dtype="float32")
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr], verbose=False)
            oof[va] = m.predict(X[va])
    mask_valid = np.isfinite(oof)
    cv_r2  = float(r2_score(y[mask_valid], oof[mask_valid]))
    cv_mae = float(mean_absolute_error(y[mask_valid], oof[mask_valid]))
    print(f"    [{label}] {target_col}: n={len(sub):,}  cv_r2={cv_r2:.3f}  cv_mae={cv_mae:.2f}")
    # Write residuals back to original df
    df = df.copy()
    df["pred_" + target_col]  = np.nan
    df["resid_" + target_col] = np.nan
    df.loc[sub.index, "pred_"  + target_col] = oof
    df.loc[sub.index, "resid_" + target_col] = y - oof
    return cv_r2, cv_mae, df


def _residual_zscore(df: pd.DataFrame, baseline_mask: pd.Series,
                     resid_col: str) -> tuple[float, float]:
    base = df.loc[baseline_mask, resid_col].dropna()
    return float(base.mean()), float(base.std(ddof=1))


def _make_windows(df: pd.DataFrame, admit_col: str,
                  resid_los: str, resid_gap: str,
                  mu_los: float, sd_los: float,
                  mu_gap: float, sd_gap: float,
                  start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Roll 4-week windows and compute LGDI-analog."""
    rows = []
    anchor = start
    while anchor <= end:
        w_end = anchor + pd.Timedelta(weeks=4) - pd.Timedelta(days=1)
        frame = df[(df[admit_col] >= anchor) & (df[admit_col] <= w_end)]
        if len(frame) < MIN_FRAME:
            rows.append({"week_start": anchor, "n": len(frame), "valid": False,
                         "lgdi": math.nan, "resp_zscore": math.nan})
            anchor += pd.Timedelta(weeks=1)
            continue
        z_los = ((frame[resid_los] - mu_los) / sd_los).mean() if sd_los > 0 else math.nan
        z_gap = ((frame[resid_gap] - mu_gap) / sd_gap).mean() if sd_gap > 0 else math.nan
        lgdi  = float(np.nanmean([z_los, -z_gap]))   # long LOS + short gap = high LGDI
        rows.append({"week_start": anchor, "n": len(frame), "valid": math.isfinite(lgdi),
                     "lgdi": lgdi, "resp_zscore": math.nan})
        anchor += pd.Timedelta(weeks=1)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# ── MIMIC-IV style loader (used for both MIMIC-IV and NWICU)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_mimic_style(hosp_dir: Path, label: str) -> pd.DataFrame:
    """Load MIMIC-format admissions + diagnoses → feature-enriched DataFrame."""
    print(f"  Loading {label} admissions...")
    with gzip.open(hosp_dir / "admissions.csv.gz", "rt") as f:
        adm = pd.read_csv(f, usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                                       "admission_type", "insurance", "race", "hospital_expire_flag"],
                          dtype={"subject_id": "int64", "hadm_id": "int64"})
    adm["admit_dt"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dis_dt"]   = pd.to_datetime(adm["dischtime"],  errors="coerce")
    adm = adm.dropna(subset=["admit_dt"]).copy()
    adm["los_days"] = (adm["dis_dt"] - adm["admit_dt"]).dt.total_seconds() / 86400
    adm.loc[(adm["los_days"] < 0) | (adm["los_days"] > 180), "los_days"] = np.nan
    adm = adm.sort_values(["subject_id", "admit_dt"]).copy()
    adm["prev_dis"]  = adm.groupby("subject_id")["dis_dt"].shift(1)
    adm["gap_days"]  = (adm["admit_dt"] - adm["prev_dis"]).dt.total_seconds() / 86400
    adm.loc[(adm["gap_days"] < 0) | (adm["gap_days"] > 3650), "gap_days"] = np.nan
    # Targets: NEXT admission's LOS and gap
    adm["next_los_days"] = adm.groupby("subject_id")["los_days"].shift(-1)
    adm["next_gap_days"] = adm.groupby("subject_id")["gap_days"].shift(-1)
    adm["year"]       = adm["admit_dt"].dt.year
    adm["week_start"] = (adm["admit_dt"] -
                         pd.to_timedelta(adm["admit_dt"].dt.weekday, unit="D")).dt.normalize()
    # Encode categorical
    for col in ["admission_type", "insurance", "race"]:
        adm[col + "_enc"] = pd.Categorical(adm[col]).codes
    adm["died_inhosp"] = adm["hospital_expire_flag"].fillna(0).astype(int)
    # Diagnoses → ICD group flags
    print(f"  Loading {label} diagnoses...")
    diag_path = hosp_dir / "diagnoses_icd.csv.gz"
    if diag_path.exists():
        with gzip.open(diag_path, "rt") as f:
            diag = pd.read_csv(f, usecols=["hadm_id", "icd_code", "seq_num"],
                               dtype={"hadm_id": "int64", "icd_code": str})
        # Primary diagnosis (seq_num == 1)
        prim = diag[diag["seq_num"] == 1].drop_duplicates("hadm_id")
        for grp, pat in ICD_GROUPS.items():
            flag = prim[prim["icd_code"].str.match(pat, na=False)]["hadm_id"]
            adm[grp] = adm["hadm_id"].isin(flag).astype(int)
    else:
        for grp in ICD_GROUPS:
            adm[grp] = 0
    print(f"  {label}: {len(adm):,} admissions, {adm.subject_id.nunique():,} patients")
    return adm


# ═══════════════════════════════════════════════════════════════════════════════
# ── eICU loader
# ═══════════════════════════════════════════════════════════════════════════════

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
    # Hospital LOS (minutes → days)
    df["hosp_los_days"] = (df["hospitaldischargeoffset"] - df["hospitaladmitoffset"]) / 60 / 24
    df.loc[(df["hosp_los_days"] < 0) | (df["hosp_los_days"] > 365), "hosp_los_days"] = np.nan
    # Age (">89" → 90)
    df["age_num"] = pd.to_numeric(df["age"].replace("> 89", "90").replace(">89","90"),
                                  errors="coerce")
    # Encode categoricals
    df["unittype_enc"]   = pd.Categorical(df["unittype"]).codes
    df["apachedx_enc"]   = pd.Categorical(df["apacheadmissiondx"]).codes
    df["died_inhosp"]    = (df["hospitaldischargestatus"] == "Expired").astype(int)
    # Hospital-level repeat stays (group by uniquepid + hospitaldischargeyear)
    # Use hospitaldischargeyear as an approximate time anchor (real years 2014-2015)
    df["hosp_year"] = df["hospitaldischargeyear"]
    # Sort by patient + hospitaladmitoffset is approximate; use uniquepid + offset
    df = df.sort_values(["uniquepid", "hospitaladmitoffset"]).copy()
    # Compute next_hosp_los_days per patient
    df["next_hosp_los_days"] = df.groupby("uniquepid")["hosp_los_days"].shift(-1)
    # Approximate inter-hospital-stay gap (hospitaladmitoffset is in minutes from ICU reference)
    # We use the unit discharge → next unit admit as a gap proxy
    df["unit_gap_h"] = (df.groupby("uniquepid")["hospitaladmitoffset"].shift(-1) -
                        df["hospitaldischargeoffset"])
    df["unit_gap_h"] = df["unit_gap_h"].clip(lower=0) / 60  # hours
    df["next_gap_days"] = df["unit_gap_h"] / 24
    df.loc[df["next_gap_days"] > 3650, "next_gap_days"] = np.nan
    print(f"  eICU: {len(df):,} unit stays, {df.uniquepid.nunique():,} patients, "
          f"years {df.hosp_year.min():.0f}-{df.hosp_year.max():.0f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ── CDSL loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cdsl() -> pd.DataFrame:
    print("  Loading CDSL...")
    df = pd.read_csv(CDSL_DIR / "patient_01.csv")
    df["admit_dt"]   = pd.to_datetime(df["admission_d_inpat"],   errors="coerce")
    df["dis_dt"]     = pd.to_datetime(df["discharge_date"],      errors="coerce")
    df["prev_admit"] = pd.to_datetime(df["ant_admission_date_in"], errors="coerce")
    df["los_days"]   = (df["dis_dt"] - df["admit_dt"]).dt.days.astype(float)
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 365), "los_days"] = np.nan
    df["age_num"]  = pd.to_numeric(df["age"], errors="coerce")
    df["sex_enc"]  = (df["sex"] == "FEMALE").astype(int)
    df["has_icu"]  = df["icu_days"].notna().astype(int)
    df["icu_days_filled"] = df["icu_days"].fillna(0)
    for col in ["bp_max_first_emerg", "bp_min_first_emerg", "temp_first_emerg",
                "hr_first_emerg", "sat_02_first_emerg", "glu_first_emerg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  CDSL: {len(df):,} patients (1-per-patient; no repeat admissions)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ── Cardiac loader (reuses _prep_common from phase_b_lgdi_flu_ref)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cardiac() -> pd.DataFrame:
    print("  Loading cardiac wide table...")
    df = pd.read_csv(CARDIAC_CSV, encoding="utf-8-sig", low_memory=False, dtype={"病案号": str})
    df["mrn"]       = df["病案号"].fillna("").astype(str).str.strip()
    df["admit_dt"]  = pd.to_datetime(df["入院时间"], errors="coerce")
    df["dis_dt"]    = pd.to_datetime(df["出院时间"],  errors="coerce")
    df = df.dropna(subset=["admit_dt"]).copy()
    df["los_days"]  = (df["dis_dt"] - df["admit_dt"]).dt.total_seconds() / 86400
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan
    df = df.sort_values(["mrn", "admit_dt"]).copy()
    df["prev_dis"]        = df.groupby("mrn")["dis_dt"].shift(1)
    df["gap_days"]        = (df["admit_dt"] - df["prev_dis"]).dt.total_seconds() / 86400
    df.loc[(df["gap_days"] < 0) | (df["gap_days"] > 3650), "gap_days"] = np.nan
    df["next_los_days"]   = df.groupby("mrn")["los_days"].shift(-1)
    df["next_gap_days"]   = df.groupby("mrn")["gap_days"].shift(-1)
    df["year"]            = df["admit_dt"].dt.year
    df["week_start"]      = (df["admit_dt"] -
                              pd.to_timedelta(df["admit_dt"].dt.weekday, unit="D")).dt.normalize()
    all_dx = df.get("主要诊断", pd.Series("", index=df.index)).fillna("").astype(str)
    for grp, pat in {
        "respiratory": r"J",
        "cardiovascular": r"I",
        "cerebrovascular": r"I6[0-9]",
        "diabetes": r"E1[0-4]",
        "renal": r"N1[7-9]|N0[3-5]",
        "hypertension": r"I1[0-5]",
    }.items():
        df[grp] = all_dx.str.contains(pat, regex=True, na=False).astype(int)
    print(f"  Cardiac: {len(df):,} admissions, {df.mrn.nunique():,} patients")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ── Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_mimic_style(df: pd.DataFrame, label: str,
                    baseline_years: tuple[int, int],
                    monitor_start: pd.Timestamp,
                    monitor_end: pd.Timestamp,
                    group_col: str = "subject_id",
                    admit_col: str = "admit_dt") -> dict:
    """Full WHU-symmetric pipeline for MIMIC-style datasets."""
    feat_cols_los = ["los_days", "gap_days", "admission_type_enc",
                     "insurance_enc", "race_enc", "died_inhosp"] + list(ICD_GROUPS.keys())
    feat_cols_gap = feat_cols_los[:]
    # Filter to only admissions with a next admission (has targets)
    baseline_mask = df["year"].between(*baseline_years)
    print(f"  [{label}] Baseline admissions: {baseline_mask.sum():,}")
    # Train LOS predictor
    r2_los, mae_los, df = _xgb_cv_regress(df, feat_cols_los, "next_los_days", group_col, label)
    # Train gap predictor
    r2_gap, mae_gap, df = _xgb_cv_regress(df, feat_cols_gap, "next_gap_days", group_col, label)
    # Baseline residual statistics
    mu_los, sd_los = _residual_zscore(df, baseline_mask, "resid_next_los_days")
    mu_gap, sd_gap = _residual_zscore(df, baseline_mask, "resid_next_gap_days")
    print(f"  [{label}] Baseline resid_LOS: μ={mu_los:.3f} σ={sd_los:.3f}")
    print(f"  [{label}] Baseline resid_gap: μ={mu_gap:.3f} σ={sd_gap:.3f}")
    # Rolling windows
    timeline = _make_windows(df, admit_col,
                             "resid_next_los_days", "resid_next_gap_days",
                             mu_los, sd_los, mu_gap, sd_gap,
                             monitor_start, monitor_end)
    n_valid = int(timeline["valid"].sum())
    print(f"  [{label}] Windows: {len(timeline)}, valid: {n_valid}")
    # Save timeline
    tl = timeline.copy()
    tl["week_start"] = tl["week_start"].astype(str)
    tl.to_csv(OUT / f"{label.replace(' ','_')}_lgdi_timeline.csv",
              index=False, encoding="utf-8-sig")
    return {
        "label": label,
        "n_admissions": int(len(df)),
        "n_patients": int(df[group_col].nunique()),
        "baseline_years": list(baseline_years),
        "monitor_start": str(monitor_start.date()),
        "monitor_end": str(monitor_end.date()),
        "xgb_next_los_cv_r2": r2_los,
        "xgb_next_los_cv_mae": mae_los,
        "xgb_next_gap_cv_r2": r2_gap,
        "xgb_next_gap_cv_mae": mae_gap,
        "n_windows": int(len(timeline)),
        "n_valid_windows": n_valid,
        "note": "shifted calendar years — no FluNet comparison",
    }


def run_eicu(df: pd.DataFrame) -> dict:
    """eICU pipeline (real years 2014-2015; ICU-stay LOS)."""
    label = "eICU"
    feat_cols = ["hosp_los_days", "age_num", "unittype_enc", "apachedx_enc",
                 "unitvisitnumber", "admissionweight", "died_inhosp"]
    # Targets: next_hosp_los_days and next_gap_days (both within uniquepid)
    # Baseline: 2014; monitor: 2015
    df["year"] = df["hosp_year"].astype(int)
    baseline_mask = df["year"] == 2014
    monitor_start = None  # eICU has no true date; just report CV performance
    print(f"  [eICU] Baseline (year 2014): {baseline_mask.sum():,} stays")
    # CV for LOS prediction
    r2_los, mae_los, df = _xgb_cv_regress(df, feat_cols, "next_hosp_los_days",
                                           "uniquepid", label)
    r2_gap, mae_gap, df = _xgb_cv_regress(df, feat_cols, "next_gap_days",
                                           "uniquepid", label)
    # Rolling windows by year (approximate — eICU has no exact dates, only year)
    print(f"  [eICU] Year dist:\n{df.year.value_counts().sort_index().to_string()}")
    return {
        "label": label,
        "n_stays": int(len(df)),
        "n_patients": int(df.uniquepid.nunique()),
        "years": [int(df.year.min()), int(df.year.max())],
        "baseline_year": 2014,
        "xgb_next_hosp_los_cv_r2": r2_los,
        "xgb_next_hosp_los_cv_mae": mae_los,
        "xgb_next_gap_cv_r2": r2_gap,
        "xgb_next_gap_cv_mae": mae_gap,
        "note": (
            "ICU dataset — real years 2014-2015. "
            "LOS = hospital LOS derived from offset. "
            "Gap = approximate inter-stay interval. "
            "Low repeat-stay rate (avg 1.44 stays/patient) limits LGDI time series."
        ),
    }


def run_cdsl(df: pd.DataFrame) -> dict:
    """CDSL: single-admission COVID cohort. LOS prediction only (no gap)."""
    label = "CDSL"
    feat_cols = ["age_num", "sex_enc", "has_icu", "icu_days_filled",
                 "bp_max_first_emerg", "bp_min_first_emerg", "temp_first_emerg",
                 "hr_first_emerg", "sat_02_first_emerg", "glu_first_emerg"]
    df["dummy_group"] = np.arange(len(df))   # no patient groups (1 per patient)
    # Simple train/test split for CDSL (no GroupKFold sense without repeat admissions)
    from sklearn.model_selection import KFold
    sub = df.dropna(subset=feat_cols + ["los_days"]).copy()
    X = sub[feat_cols].fillna(sub[feat_cols].median()).values.astype("float32")
    y = sub["los_days"].values.astype("float32")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(sub), np.nan, "float32")
    try:
        params = {**XGB_PARAMS}
        for tr, va in kf.split(X, y):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
    except Exception:
        params = {**XGB_PARAMS, "device": "cpu"}
        oof = np.full(len(sub), np.nan, "float32")
        for tr, va in kf.split(X, y):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
    r2  = float(r2_score(y, oof))
    mae = float(mean_absolute_error(y, oof))
    print(f"  [CDSL] LOS: n={len(sub):,}  cv_r2={r2:.3f}  cv_mae={mae:.2f}")
    return {
        "label": label,
        "n_patients": int(len(df)),
        "xgb_los_cv_r2": r2,
        "xgb_los_cv_mae": mae,
        "note": "All COVID patients; 1 admission per patient; no gap_days; no LGDI time series.",
    }


def run_cardiac_splits(cardiac: pd.DataFrame) -> list[dict]:
    """Three cardiac split strategies."""
    results = []

    # E1: Frequent visitors (≥5 admissions per patient)
    print("\n[E1] Cardiac — frequent visitors (≥5 admissions)")
    visit_counts = cardiac.groupby("mrn")["admit_dt"].count()
    freq_mrns = visit_counts[visit_counts >= 5].index
    fv = cardiac[cardiac["mrn"].isin(freq_mrns)].copy()
    fv = fv[fv["year"].between(2012, 2024)].copy()
    feat_cols = ["los_days", "gap_days"] + list(ICD_GROUPS.keys())
    r2_los, mae_los, fv = _xgb_cv_regress(fv, feat_cols, "next_los_days", "mrn", "cardiac_fv")
    r2_gap, mae_gap, fv = _xgb_cv_regress(fv, feat_cols, "next_gap_days", "mrn", "cardiac_fv")
    results.append({
        "label": "cardiac_frequent_visitors",
        "threshold": ">=5 admissions",
        "n_admissions": int(len(fv)),
        "n_patients": int(fv.mrn.nunique()),
        "xgb_next_los_cv_r2": r2_los,
        "xgb_next_los_cv_mae": mae_los,
        "xgb_next_gap_cv_r2": r2_gap,
        "xgb_next_gap_cv_mae": mae_gap,
        "note": "High-frequency repeat admitters; should show better XGBoost performance",
    })

    # E2: Respiratory-focus (respiratory-coded cardiac patients)
    print("\n[E2] Cardiac — respiratory-coded group")
    resp = cardiac[cardiac["respiratory"] == 1].copy()
    feat_cols2 = ["los_days", "gap_days", "cardiovascular", "diabetes", "renal", "hypertension"]
    r2_los2, mae_los2, resp = _xgb_cv_regress(resp, feat_cols2, "next_los_days", "mrn", "cardiac_resp")
    r2_gap2, mae_gap2, resp = _xgb_cv_regress(resp, feat_cols2, "next_gap_days", "mrn", "cardiac_resp")
    # Rolling LGDI for respiratory subgroup 2016-2019
    bl_mask = resp["year"].between(2012, 2016)
    mu_los2, sd_los2 = _residual_zscore(resp, bl_mask, "resid_next_los_days")
    mu_gap2, sd_gap2 = _residual_zscore(resp, bl_mask, "resid_next_gap_days")
    tl2 = _make_windows(resp, "admit_dt",
                        "resid_next_los_days", "resid_next_gap_days",
                        mu_los2, sd_los2, mu_gap2, sd_gap2,
                        pd.Timestamp("2016-01-01"), pd.Timestamp("2019-06-30"))
    tl2["week_start"] = tl2["week_start"].astype(str)
    tl2.to_csv(OUT / "cardiac_respiratory_focus_lgdi_timeline.csv", index=False, encoding="utf-8-sig")
    results.append({
        "label": "cardiac_respiratory_focus",
        "n_admissions": int(len(resp)),
        "n_patients": int(resp.mrn.nunique()),
        "xgb_next_los_cv_r2": r2_los2,
        "xgb_next_los_cv_mae": mae_los2,
        "xgb_next_gap_cv_r2": r2_gap2,
        "xgb_next_gap_cv_mae": mae_gap2,
        "lgdi_windows": int(len(tl2)),
        "lgdi_valid_windows": int(tl2["valid"].sum()),
        "note": "Respiratory-coded cardiac patients (J-code primary diagnosis)",
    })

    # E3: Extended monitor — 2019-2024 (post-gap COVID era)
    print("\n[E3] Cardiac — extended monitor 2019-2024")
    all_cardiac = cardiac[cardiac["year"].between(2012, 2024)].copy()
    feat_cols3 = ["los_days", "gap_days"] + list(ICD_GROUPS.keys())
    r2_los3, mae_los3, all_cardiac = _xgb_cv_regress(
        all_cardiac, feat_cols3, "next_los_days", "mrn", "cardiac_ext")
    r2_gap3, mae_gap3, all_cardiac = _xgb_cv_regress(
        all_cardiac, feat_cols3, "next_gap_days", "mrn", "cardiac_ext")
    bl3 = all_cardiac["year"].between(2012, 2016)
    mu_los3, sd_los3 = _residual_zscore(all_cardiac, bl3, "resid_next_los_days")
    mu_gap3, sd_gap3 = _residual_zscore(all_cardiac, bl3, "resid_next_gap_days")
    # Two sub-windows: pre-gap (2016-2019) and post-gap (2020-2024)
    tl3a = _make_windows(all_cardiac, "admit_dt",
                         "resid_next_los_days", "resid_next_gap_days",
                         mu_los3, sd_los3, mu_gap3, sd_gap3,
                         pd.Timestamp("2016-01-01"), pd.Timestamp("2019-06-30"))
    tl3b = _make_windows(all_cardiac, "admit_dt",
                         "resid_next_los_days", "resid_next_gap_days",
                         mu_los3, sd_los3, mu_gap3, sd_gap3,
                         pd.Timestamp("2020-05-01"), pd.Timestamp("2024-12-31"))
    for t, name in [(tl3a, "cardiac_extended_pre2019"), (tl3b, "cardiac_extended_covid")]:
        t2 = t.copy(); t2["week_start"] = t2["week_start"].astype(str)
        t2.to_csv(OUT / f"{name}_lgdi_timeline.csv", index=False, encoding="utf-8-sig")
    results.append({
        "label": "cardiac_extended",
        "n_admissions": int(len(all_cardiac)),
        "n_patients": int(all_cardiac.mrn.nunique()),
        "xgb_next_los_cv_r2": r2_los3,
        "xgb_next_los_cv_mae": mae_los3,
        "xgb_next_gap_cv_r2": r2_gap3,
        "xgb_next_gap_cv_mae": mae_gap3,
        "lgdi_pre2019_windows": int(len(tl3a)),
        "lgdi_pre2019_valid": int(tl3a["valid"].sum()),
        "lgdi_covid_windows": int(len(tl3b)),
        "lgdi_covid_valid": int(tl3b["valid"].sum()),
        "note": "Full 2012-2024 cardiac; two monitor windows: flu era (pre-2019) + COVID era (2020-2024)",
    })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("phase_g_xgb_symmetric_all_datasets.py")
    print("WHU-symmetric XGBoost regression + LGDI on all available datasets")
    print("=" * 70)
    all_results = []

    # ── A. MIMIC-IV ────────────────────────────────────────────────────────────
    print("\n[A] MIMIC-IV")
    mimic = _load_mimic_style(MIMIC_HOSP, "MIMIC-IV")
    # Years are shifted (2105-2212); use relative year structure.
    # Baseline = first 5 years of data range; monitor = next 5 years.
    ymin = int(mimic.year.min())
    res_mimic = run_mimic_style(
        mimic, "MIMIC-IV",
        baseline_years=(ymin, ymin + 4),
        monitor_start=pd.Timestamp(f"{ymin+5}-01-01"),
        monitor_end=pd.Timestamp(f"{ymin+9}-12-31"),
    )
    all_results.append(res_mimic)

    # ── B. NWICU ───────────────────────────────────────────────────────────────
    print("\n[B] NWICU")
    nwicu = _load_mimic_style(NWICU_HOSP, "NWICU")
    ymin_nw = int(nwicu.year.min())
    res_nwicu = run_mimic_style(
        nwicu, "NWICU",
        baseline_years=(ymin_nw, ymin_nw + 4),
        monitor_start=pd.Timestamp(f"{ymin_nw+5}-01-01"),
        monitor_end=pd.Timestamp(f"{ymin_nw+9}-12-31"),
    )
    all_results.append(res_nwicu)

    # ── C. eICU ────────────────────────────────────────────────────────────────
    print("\n[C] eICU")
    eicu = _load_eicu()
    res_eicu = run_eicu(eicu)
    all_results.append(res_eicu)

    # ── D. CDSL ────────────────────────────────────────────────────────────────
    print("\n[D] CDSL")
    cdsl = _load_cdsl()
    res_cdsl = run_cdsl(cdsl)
    all_results.append(res_cdsl)

    # ── E. Cardiac splits ──────────────────────────────────────────────────────
    print("\n[E] Cardiac splits")
    cardiac = _load_cardiac()
    cardiac_results = run_cardiac_splits(cardiac)
    all_results.extend(cardiac_results)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — XGBoost CV R² (next_LOS / next_gap):")
    print("=" * 70)
    for r in all_results:
        r2_los = r.get("xgb_next_los_cv_r2", r.get("xgb_los_cv_r2", r.get("xgb_next_hosp_los_cv_r2", "—")))
        r2_gap = r.get("xgb_next_gap_cv_r2", "—")
        label  = r["label"]
        n      = r.get("n_admissions", r.get("n_stays", r.get("n_patients", "?")))
        print(f"  {label:35s}  n={n:>7,}  R²_LOS={r2_los!r:>8}  R²_gap={r2_gap!r:>8}")

    # Write performance CSV
    rows = []
    for r in all_results:
        rows.append({
            "dataset": r["label"],
            "n": r.get("n_admissions", r.get("n_stays", r.get("n_patients"))),
            "n_patients": r.get("n_patients"),
            "xgb_next_los_cv_r2":  r.get("xgb_next_los_cv_r2",
                                          r.get("xgb_los_cv_r2",
                                          r.get("xgb_next_hosp_los_cv_r2"))),
            "xgb_next_los_cv_mae": r.get("xgb_next_los_cv_mae",
                                          r.get("xgb_los_cv_mae",
                                          r.get("xgb_next_hosp_los_cv_mae"))),
            "xgb_next_gap_cv_r2":  r.get("xgb_next_gap_cv_r2"),
            "xgb_next_gap_cv_mae": r.get("xgb_next_gap_cv_mae"),
            "lgdi_valid_windows":  r.get("n_valid_windows",
                                         r.get("lgdi_pre2019_valid",
                                         r.get("lgdi_valid_windows"))),
            "note": r.get("note", ""),
        })
    pd.DataFrame(rows).to_csv(OUT / "xgb_symmetric_all_datasets_performance.csv",
                               index=False, encoding="utf-8-sig")
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "strategy": (
            "WHU-symmetric XGBoost regression (GroupKFold CV, n_splits=5) "
            "using each dataset's native features. Targets: next_LOS and next_gap_days. "
            "Rolling 4-week LGDI windows computed where date information is available."
        ),
        "datasets": all_results,
    }
    (OUT / "xgb_symmetric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] All outputs → {OUT}")


if __name__ == "__main__":
    main()
