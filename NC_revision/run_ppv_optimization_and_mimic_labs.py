#!/usr/bin/env python3
r"""Three-in-one experiment:
  A. MIMIC labevents → XGBoost (does adding labs rescue R²?)
  B. PPV optimization sweep on cardiac COVID data (can we improve PPV?)
  C. Patient-count analysis (does window sample size explain low PPV?)

Outputs: NC_revision/cross_institution_lgdi_results/
  covid_ppv_optimization.json +.csv
  patient_count_analysis.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "cross_institution_lgdi_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ───────────────────────────────────────────────────────────────
MIMIC_HOSP = BASE.parent / "external_data" / "physionet" / "mimic-iv-2.2" / "mimic-iv-2.2" / "hosp"
CARDIAC_LGDI = BASE / "lgdi_results" / "lgdi_rolling4_weekly.csv"
CARDIAC_RDI = BASE / "weekly_rdi_42k_results" / "weekly_rdi_42k_rolling4_weekly.csv"
CARDIAC_EXPANDED = BASE / "expanded_cardiac_wide_table.csv"

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]

# ══════════════════════════════════════════════════════════════════════════
#  PART A — MIMIC labevents + XGBoost (lightweight)
# ══════════════════════════════════════════════════════════════════════════

def load_mimic_lab_itemids() -> dict[str, list[int]]:
    """Find MIMIC itemids for common lab tests."""
    items = pd.read_csv(MIMIC_HOSP / "d_labitems.csv.gz", low_memory=False)
    print(f"  Loaded {len(items)} lab items")

    target_labs = {
        "lab_WBC": ["white blood", "leukocyte", "wbc"],
        "lab_HGB": ["hemoglobin", "haemoglobin", "hgb"],
        "lab_CREA": ["creatinine", "creat"],
        "lab_GLU": ["glucose", "glu"],
        "lab_K": ["potassium", "^k$"],
        "lab_Na": ["sodium", "^na$"],
        "lab_CRP": ["c reactive", "crp"],
        "lab_ALB": ["albumin"],
    }

    result = {}
    for lab_name, patterns in target_labs.items():
        pattern = "|".join(patterns)
        match = items[items["label"].str.lower().str.contains(pattern, na=False, regex=True)]
        if len(match) > 0:
            itemids = match["itemid"].unique().tolist()
            result[lab_name] = itemids
            print(f"    {lab_name}: {len(itemids)} itemids — sample: {match['label'].iloc[:3].tolist()}")
    return result


def aggregate_mimic_labs(lab_itemids: dict[str, list[int]]) -> pd.DataFrame:
    """Chunk-read labevents, filter to target itemids, aggregate per hadm_id."""
    all_ids = set()
    for ids in lab_itemids.values():
        all_ids.update(ids)
    print(f"\n  Processing labevents.csv.gz (1.8GB) — keeping {len(all_ids)} target itemids...")

    lab_file = MIMIC_HOSP / "labevents.csv.gz"
    chunks = []
    total_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(pd.read_csv(lab_file, low_memory=False, chunksize=500000,
                                           usecols=["hadm_id", "itemid", "valuenum"])):
        total_rows += len(chunk)
        filtered = chunk[chunk["itemid"].isin(all_ids)]
        kept_rows += len(filtered)
        if len(filtered) > 0:
            chunks.append(filtered)
        if (i + 1) % 20 == 0:
            print(f"    Chunk {i+1}: {total_rows/1e6:.1f}M rows scanned, {kept_rows/1e3:.1f}K kept")

    if not chunks:
        print("  ❌ No lab rows matched target itemids!")
        return pd.DataFrame()

    all_labs = pd.concat(chunks, ignore_index=True)
    print(f"  Total: {total_rows/1e6:.1f}M rows scanned, {kept_rows/1e3:.1f}K kept")

    # Map itemid to lab_name
    id_to_lab = {}
    for lab_name, ids in lab_itemids.items():
        for iid in ids:
            id_to_lab[iid] = lab_name
    all_labs["lab_name"] = all_labs["itemid"].map(id_to_lab)

    # Aggregate per hadm_id per lab
    agg = all_labs.groupby(["hadm_id", "lab_name"])["valuenum"].mean().reset_index()
    pivoted = agg.pivot(index="hadm_id", columns="lab_name", values="valuenum").reset_index()
    pivoted.columns.name = None

    print(f"  Aggregated: {len(pivoted)} admissions with labs")
    return pivoted


def mimic_xgboost_quick() -> dict:
    """Train XGBoost on MIMIC WITH lab values, compare to without-labs baseline."""
    print("\n" + "=" * 70)
    print("PART A: MIMIC labevents → XGBoost")
    print("=" * 70)

    # Load admissions + diagnoses (lightweight)
    print("  Loading admissions + diagnoses...")
    adm = pd.read_csv(MIMIC_HOSP / "admissions.csv.gz", low_memory=False)
    adm["admit_dt"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["discharge_dt"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    diag = pd.read_csv(MIMIC_HOSP / "diagnoses_icd.csv.gz", low_memory=False)
    diag_text = diag.groupby("hadm_id")["icd_code"].apply(
        lambda x: " ".join(x.dropna().astype(str))
    ).reset_index(name="all_icd_codes")
    df = adm.merge(diag_text, on="hadm_id", how="left")
    df["all_icd_codes"] = df["all_icd_codes"].fillna("")
    df["subject_id"] = df["subject_id"].astype(str)
    df["los_days"] = (df["discharge_dt"] - df["admit_dt"]).dt.total_seconds() / 86400
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan
    print(f"  {len(df)} admissions, {df['subject_id'].nunique()} patients")

    # Get lab itemids
    lab_itemids = load_mimic_lab_itemids()
    if not lab_itemids:
        return {"error": "No lab itemids found", "r2_no_labs": 0.124, "r2_with_labs": None}

    # Aggregate labs
    labs_pivoted = aggregate_mimic_labs(lab_itemids)
    if len(labs_pivoted) > 0:
        df = df.merge(labs_pivoted, on="hadm_id", how="left")

    # Build features
    df = df.sort_values(["subject_id", "admit_dt"]).reset_index(drop=True)
    g = df.groupby("subject_id", sort=False)
    df["visit_order"] = g.cumcount()
    df["first_admit_dt"] = g["admit_dt"].transform("min")
    df["days_since_first"] = (df["admit_dt"] - df["first_admit_dt"]).dt.total_seconds() / 86400
    df["prior_los_mean"] = g["los_days"].transform(lambda s: s.shift(1).expanding().mean())
    df["prior_los_std"] = g["los_days"].transform(lambda s: s.shift(1).expanding().std())
    df["prior_los_last"] = g["los_days"].shift(1)
    df["prior_gap_mean"] = g["gap_days"].transform(lambda s: s.shift(1).expanding().mean()) if "gap_days" in df.columns else np.nan
    df["prev_discharge"] = g["discharge_dt"].shift(1)
    df["gap_days"] = (df["admit_dt"] - df["prev_discharge"]).dt.total_seconds() / 86400
    df.loc[(df["gap_days"] < 0) | (df["gap_days"] > 3650), "gap_days"] = np.nan
    df["next_admit_dt"] = g["admit_dt"].shift(-1)
    df["next_los_days"] = g["los_days"].shift(-1)
    df["next_gap_days"] = (df["next_admit_dt"] - df["discharge_dt"]).dt.total_seconds() / 86400
    df.loc[(df["next_gap_days"] < 0) | (df["next_gap_days"] > 3650), "next_gap_days"] = np.nan

    # Features: temporal only (no-labs baseline) vs temporal + labs
    temporal_feats = ["visit_order", "days_since_first", "prior_los_mean", "prior_los_std", "prior_los_last",
                       "month_sin", "month_cos", "dow_sin", "dow_cos"]
    month = df["admit_dt"].dt.month.astype(float)
    dow = df["admit_dt"].dt.weekday.astype(float)
    for col_name, vals in [("month_sin", np.sin(2*np.pi*month/12)), ("month_cos", np.cos(2*np.pi*month/12)),
                            ("dow_sin", np.sin(2*np.pi*dow/7)), ("dow_cos", np.cos(2*np.pi*dow/7))]:
        df[col_name] = vals

    lab_feats = [f for f in lab_itemids if f in df.columns]
    all_feats = temporal_feats + lab_feats
    available_temporal = [f for f in temporal_feats if f in df.columns]
    available_all = [f for f in all_feats if f in df.columns]

    print(f"  Temporal features: {len(available_temporal)}")
    print(f"  Lab features: {len([f for f in lab_feats if f in df.columns])}")
    print(f"  Total features: {len(available_all)}")

    # Train/eval on gap prediction (most relevant for visit rhythm)
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error

    target = "next_gap_days"
    subset = df.dropna(subset=[target] + available_all).copy()
    if len(subset) < 500:
        return {"error": f"Only {len(subset)} trainable rows", "r2_no_labs": 0.124}

    X_no_labs = subset[available_temporal].fillna(0.0).astype(float)
    X_with_labs = subset[available_all].fillna(0.0).astype(float)
    y = subset[target].astype(float)
    groups = subset["subject_id"].fillna("").astype(str).values

    results = {"n_train": int(len(subset)), "n_patients": int(len(set(groups)))}

    for label, X in [("no_labs", X_no_labs), ("with_labs", X_with_labs)]:
        n_splits = min(5, max(2, len(set(groups))))
        gkf = GroupKFold(n_splits=n_splits)
        preds, truths = [], []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                      random_state=20260516, tree_method="hist", n_jobs=4)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds.extend(model.predict(X.iloc[va_idx]).tolist())
            truths.extend(y.iloc[va_idx].tolist())

        r2 = float(r2_score(truths, preds)) if np.var(truths) > 1e-9 else float("nan")
        mae = float(mean_absolute_error(truths, preds))
        results[f"r2_{label}"] = r2
        results[f"mae_{label}"] = mae
        print(f"  Gap R² ({label}): {r2:.4f}, MAE: {mae:.1f} days")

    results["r2_delta_from_labs"] = results.get("r2_with_labs", float("nan")) - results.get("r2_no_labs", float("nan"))
    return results


# ══════════════════════════════════════════════════════════════════════════
#  PART B — PPV optimization sweep on cardiac COVID data
# ══════════════════════════════════════════════════════════════════════════

def safe_div(a, b):
    return float(a / b) if b else float("nan")


def compute_oc(tp, fp, fn, tn):
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": safe_div(tp, tp + fn),
        "ppv": safe_div(tp, tp + fp),
        "far": safe_div(fp, fp + tn),
    }


def ppv_optimization_sweep() -> dict:
    """Systematic PPV sweep on cardiac COVID data."""
    print("\n" + "=" * 70)
    print("PART B: PPV Optimization Sweep on Cardiac COVID")
    print("=" * 70)

    # Load cardiac LGDI weekly data (has event_window flag)
    lgdi = pd.read_csv(CARDIAC_LGDI, low_memory=False)
    lgdi["week_start_dt"] = pd.to_datetime(lgdi["window_anchor"])

    # COVID event: Dec 2022 - Jan 2023
    covid_start = pd.Timestamp("2022-12-01")
    covid_end = pd.Timestamp("2023-01-31")
    lgdi["is_covid"] = lgdi["week_start_dt"].between(covid_start, covid_end)

    # Only valid weeks
    valid = lgdi[lgdi["valid"].eq(True)].copy()
    print(f"  Valid weeks: {len(valid)}, COVID weeks: {valid['is_covid'].sum()}")

    # Baseline 2016-2018
    bl = valid[valid["week_start_dt"].dt.year.between(2016, 2018)]
    lgdi_mean = float(bl["lgdi"].mean())
    lgdi_sd = float(bl["lgdi"].std()) or 0.001
    resp_mean = float(bl["resp_score"].mean())
    resp_sd = float(bl["resp_score"].std()) or 0.001

    # Also load weekly RDI for comparison
    rdi_df = None
    if CARDIAC_RDI.exists():
        rdi_df = pd.read_csv(CARDIAC_RDI, low_memory=False)
        rdi_df["week_start_dt"] = pd.to_datetime(rdi_df["window_start"])
        rdi_bl = rdi_df[rdi_df["week_start_dt"].dt.year.between(2016, 2018)]
        rdi_mean = float(rdi_bl["rdi"].mean())
        rdi_sd = float(rdi_bl["rdi"].std()) or 0.001

    # Compute per-group z-scores
    for g in GROUPS:
        col = f"score_{g}"
        if col in valid.columns:
            g_bl = bl[col].dropna()
            g_mean = float(g_bl.mean()) if len(g_bl) > 0 else 0.0
            g_sd = float(g_bl.std()) if len(g_bl) > 1 else 0.001
            valid[f"z_{g}"] = (valid[col] - g_mean) / g_sd
            valid[f"above_{g}"] = valid[f"z_{g}"] > 1.5
    above_cols = [f"above_{g}" for g in GROUPS if f"above_{g}" in valid.columns]
    if above_cols:
        valid["n_groups_above"] = valid[above_cols].sum(axis=1)

    # Define strategies to test
    strategies = []

    # A. LGDI thresholds
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for base in ["lgdi", "resp_score"]:
            if base == "lgdi":
                mean_v, sd_v = lgdi_mean, lgdi_sd
            else:
                mean_v, sd_v = resp_mean, resp_sd
            thr = mean_v + mult * sd_v
            col_name = f"alert_{base}_mean_{mult}sd"
            valid[col_name] = valid[base] > thr
            strategies.append((col_name, f"{base} > mean+{mult}SD (thr={thr:.4f})"))

    # B. RDI thresholds
    if rdi_df is not None:
        rdi_merged = valid[["week_start_dt"]].merge(
            rdi_df[["week_start_dt", "rdi"]], on="week_start_dt", how="left"
        )
        valid["rdi"] = rdi_merged["rdi"].values
        valid["rdi"] = pd.to_numeric(valid["rdi"], errors="coerce")
        for mult in [1.0, 1.5, 2.0, 2.5]:
            thr = rdi_mean + mult * rdi_sd
            col_name = f"alert_rdi_mean_{mult}sd"
            valid[col_name] = valid["rdi"] > thr
            strategies.append((col_name, f"RDI > mean+{mult}SD (thr={thr:.4f})"))

    # C. Multi-group consensus with different k
    if "n_groups_above" in valid.columns:
        for k in [1, 2, 3, 4, 5]:
            col_name = f"alert_consensus_k{k}"
            valid[col_name] = valid["n_groups_above"] >= k
            strategies.append((col_name, f"Consensus >= {k} groups"))

    # D. Sustained rule: alert must persist 2+ consecutive weeks
    valid = valid.sort_values("week_start_dt").reset_index(drop=True)
    for base_col in ["alert_lgdi_mean_1.5sd", "alert_resp_score_mean_1.5sd"]:
        if base_col not in valid.columns:
            continue
        col_name = f"{base_col}_sustained2"
        prev = valid[base_col].shift(1, fill_value=False)
        nxt = valid[base_col].shift(-1, fill_value=False)
        valid[col_name] = valid[base_col] & (prev | nxt)
        strategies.append((col_name, f"{base_col} + sustained 2 weeks"))

    # E. Season-restricted: Nov-Mar
    valid["in_core_season"] = valid["week_start_dt"].dt.month.isin([11, 12, 1, 2, 3])
    for base_col in ["alert_lgdi_mean_1.5sd", "alert_resp_score_mean_1.5sd"]:
        if base_col not in valid.columns:
            continue
        col_name = f"{base_col}_season"
        valid[col_name] = valid[base_col] & valid["in_core_season"]
        strategies.append((col_name, f"{base_col} + Nov-Mar season"))

    # F. Combined: sustained + season + consensus
    if "n_groups_above" in valid.columns:
        for k in [1, 2]:
            cons_col = f"_cons{k}_tmp"
            valid[cons_col] = valid["n_groups_above"] >= k
            for base_col in ["alert_lgdi_mean_1.5sd", "alert_resp_score_mean_1.5sd"]:
                if base_col not in valid.columns:
                    continue
                col_name = f"{base_col}_cons{k}_season"
                valid[col_name] = valid[base_col] & valid[cons_col] & valid["in_core_season"]
                strategies.append((col_name, f"{base_col} + k={k} consensus + season"))

    # G. Extreme z-score (>4, >5, >6 SD)
    if "z_Respiratory" in valid.columns:
        for z_thr in [3, 4, 5, 6]:
            col_name = f"alert_zresp_{z_thr}sd"
            valid[col_name] = valid["z_Respiratory"] > z_thr
            strategies.append((col_name, f"z_resp > {z_thr}SD"))

    # Evaluate all strategies
    results = []
    y_true = valid["is_covid"].astype(bool)
    for col_name, description in strategies:
        if col_name not in valid.columns:
            continue
        y_pred = valid[col_name].astype(bool)
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())
        tn = int((~y_pred & ~y_true).sum())
        n_alerts = int(y_pred.sum())
        results.append({
            "strategy": description,
            "n_alerts": n_alerts,
            **compute_oc(tp, fp, fn, tn),
        })

    df_r = pd.DataFrame(results)
    df_r = df_r.sort_values("ppv", ascending=False)

    print(f"\n  Tested {len(df_r)} strategies")
    print(f"\n  Top 10 by PPV:")
    print(df_r.head(10).to_string(index=False))

    print(f"\n  Top 10 by Sensitivity:")
    print(df_r.sort_values("sensitivity", ascending=False).head(10).to_string(index=False))

    return {
        "n_covid_weeks": int(y_true.sum()),
        "n_monitor_weeks": int(len(valid)),
        "n_strategies_tested": len(df_r),
        "top_by_ppv": df_r.head(10).to_dict(orient="records"),
        "top_by_sensitivity": df_r.sort_values("sensitivity", ascending=False).head(10).to_dict(orient="records"),
        "all_results": df_r.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════════════════
#  PART C — Patient count per window analysis
# ══════════════════════════════════════════════════════════════════════════

def patient_count_analysis():
    """Check if per-window sample size explains low PPV."""
    print("\n" + "=" * 70)
    print("PART C: Patient Count per Window Analysis")
    print("=" * 70)

    # Load cardiac wide table
    print("  Loading cardiac expanded wide table...")
    ct = pd.read_csv(CARDIAC_EXPANDED, low_memory=False, dtype={"病案号": str},
                      usecols=["病案号", "入院时间", "主要诊断"])
    ct["admit_dt"] = pd.to_datetime(ct["入院时间"], errors="coerce")
    ct = ct.dropna(subset=["admit_dt"])

    # Simple respiratory flag from diagnosis
    resp_pattern = r"肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染|呼吸道感染"
    ct["is_resp"] = ct["主要诊断"].fillna("").str.contains(resp_pattern, case=False, regex=True, na=False)
    ct["week_start"] = (ct["admit_dt"] - pd.to_timedelta(ct["admit_dt"].dt.weekday, unit="D")).dt.normalize()

    # Per-week counts
    weekly = ct.groupby("week_start").agg(
        n_total=("病案号", "count"),
        n_unique=("病案号", "nunique"),
        n_resp=("is_resp", "sum"),
    ).reset_index()

    # Flag COVID window
    covid_start = pd.Timestamp("2022-12-01")
    covid_end = pd.Timestamp("2023-01-31")
    weekly["is_covid_window"] = weekly["week_start"].between(covid_start, covid_end)

    print(f"\n  Weekly admission counts:")
    print(f"    All weeks: mean={weekly['n_total'].mean():.0f}, median={weekly['n_total'].median():.0f}, "
          f"p10={weekly['n_total'].quantile(0.1):.0f}, p90={weekly['n_total'].quantile(0.9):.0f}")
    print(f"    COVID weeks: mean={weekly[weekly['is_covid_window']]['n_total'].mean():.0f}, "
          f"median={weekly[weekly['is_covid_window']]['n_total'].median():.0f}")
    print(f"    Non-COVID weeks: mean={weekly[~weekly['is_covid_window']]['n_total'].mean():.0f}")

    print(f"\n  Respiratory admission counts:")
    print(f"    All weeks: mean={weekly['n_resp'].mean():.1f}, median={weekly['n_resp'].median():.0f}")
    print(f"    COVID weeks: mean={weekly[weekly['is_covid_window']]['n_resp'].mean():.1f}, "
          f"median={weekly[weekly['is_covid_window']]['n_resp'].median():.0f}")

    # Weekly with zero admissions
    zero_weeks = weekly[weekly["n_total"] == 0]
    print(f"\n  Weeks with ZERO admissions: {len(zero_weeks)} / {len(weekly)}")
    if len(zero_weeks) > 0:
        print(f"    Date range: {zero_weeks['week_start'].min().date()} to {zero_weeks['week_start'].max().date()}")

    # Compare monthly vs weekly
    weekly["month"] = weekly["week_start"].dt.to_period("M")
    monthly = weekly.groupby("month").agg(
        n_total=("n_total", "sum"),
        n_unique=("n_unique", "sum"),
        n_weeks=("week_start", "count"),
    ).reset_index()

    print(f"\n  Monthly admission counts (4-5 weeks aggregated):")
    print(f"    Mean: {monthly['n_total'].mean():.0f}, Median: {monthly['n_total'].median():.0f}, "
          f"Min: {monthly['n_total'].min():.0f}, Max: {monthly['n_total'].max():.0f}")

    # Key insight: per-group counts at weekly level
    print(f"\n  Estimated per-group weekly counts (assuming uniform 6-group split):")
    for label, weekly_n in [("Min week", weekly["n_total"].min()),
                              ("P10 week", weekly["n_total"].quantile(0.1)),
                              ("Median week", weekly["n_total"].median()),
                              ("COVID week mean", weekly[weekly["is_covid_window"]]["n_total"].mean())]:
        per_group = int(weekly_n / 6)
        print(f"    {label}: {int(weekly_n)} total → ~{per_group} per group")

    return {
        "weekly": {
            "n_mean": float(weekly["n_total"].mean()),
            "n_median": float(weekly["n_total"].median()),
            "n_p10": float(weekly["n_total"].quantile(0.1)),
            "resp_mean": float(weekly["n_resp"].mean()),
            "covid_week_mean": float(weekly[weekly["is_covid_window"]]["n_total"].mean()),
            "zero_weeks": int((weekly["n_total"] == 0).sum()),
        },
        "monthly": {
            "n_mean": float(monthly["n_total"].mean()),
            "n_median": float(monthly["n_total"].median()),
        },
        "interpretation": "Low per-group weekly sample sizes may contribute to profile instability and low PPV at weekly granularity."
    }


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    all_results = {}

    # Part A: MIMIC labs + XGBoost
    try:
        mimic_r = mimic_xgboost_quick()
        all_results["mimic_lab_xgboost"] = mimic_r
        with open(OUT_DIR / "mimic_lab_xgboost.json", "w") as f:
            json.dump(mimic_r, f, indent=2, default=str)
    except Exception as e:
        print(f"  Part A failed: {e}")
        import traceback
        traceback.print_exc()

    # Part B: PPV sweep
    try:
        ppv_r = ppv_optimization_sweep()
        all_results["ppv_sweep"] = ppv_r
        with open(OUT_DIR / "covid_ppv_optimization.json", "w") as f:
            json.dump(ppv_r, f, indent=2, default=str)
        # Save CSV of all results
        pd.DataFrame(ppv_r["all_results"]).to_csv(OUT_DIR / "covid_ppv_optimization.csv", index=False)
    except Exception as e:
        print(f"  Part B failed: {e}")
        import traceback
        traceback.print_exc()

    # Part C: Patient counts
    try:
        pc_r = patient_count_analysis()
        all_results["patient_counts"] = pc_r
        with open(OUT_DIR / "patient_count_analysis.json", "w") as f:
            json.dump(pc_r, f, indent=2, default=str)
    except Exception as e:
        print(f"  Part C failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "mimic_lab_xgboost" in all_results:
        mr = all_results["mimic_lab_xgboost"]
        print(f"MIMIC XGBoost: R²_no_labs={mr.get('r2_no_labs','N/A')}, R²_with_labs={mr.get('r2_with_labs','N/A')}, Δ={mr.get('r2_delta_from_labs','N/A')}")
    if "ppv_sweep" in all_results:
        pr = all_results["ppv_sweep"]
        print(f"PPV sweep: {pr['n_strategies_tested']} strategies on {pr['n_covid_weeks']} COVID weeks")
        if pr.get("top_by_ppv"):
            best = pr["top_by_ppv"][0]
            print(f"  Best PPV: {best['ppv']:.3f} ({best['strategy']})")
            print(f"  Best Se:  {best['sensitivity']:.3f}")
    if "patient_counts" in all_results:
        pc = all_results["patient_counts"]
        print(f"Patient counts: weekly median={pc['weekly']['n_median']:.0f}, monthly median={pc['monthly']['n_median']:.0f}")


if __name__ == "__main__":
    main()
