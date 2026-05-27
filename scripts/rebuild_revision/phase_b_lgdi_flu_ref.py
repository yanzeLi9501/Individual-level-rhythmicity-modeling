"""
Phase B – LGDI pipeline rerun with FluNet-anchored historical flu reference.

Replaces the circular COVID-event design (Dec 2022–Jan 2023) with a
pre-specifiable historical influenza-season reference.

  LGDI component (XGBoost residual monitoring)
  ─────────────────────────────────────────────
  Input data   : NC_revision/expanded_cardiac_wide_table.csv
                 (cardiac specialty; 299 728 admissions, 2007-2024)
                 — used because dense repeated-admission data is required
                   for reliable XGBoost LOS/gap modelling
  Baseline     : 2012-2016  (~35 k admissions, pre-flu-event)
  Monitor      : 2016-01-01 → 2019-06-30  (stops before 2019-07 policy gap)
  Event weeks  : FluNet China positivity > 0.30
                 (28 weeks in 2016–2019: Mar 2016 × 5, Dec 2017–Feb 2018 × 10,
                 Jan–Apr 2019 × 13)
  Lag tests    : Spearman ρ at lags 0–4 with Bonferroni α = 0.01

  Pearson reference vector (flu-season profile correlation)
  ──────────────────────────────────────────────────────────
  Input data   : NC_revision/_tmp_whu_primary_for_lgdi.csv
                 (WHU primary general hospital; 50 781 admissions, 2012-2020)
  Reference    : respiratory-coded admissions in Jan–Feb 2018 (n ≈ 292)
                 — FluNet peak season, pre-specifiable without COVID knowledge
  Comparison   : Q4 2017 group profiles vs flu reference

Outputs → NC_revision/RebuildRevision/outputs/lgdi_flu_ref/
"""
from __future__ import annotations

import json
import math
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
SUBMIT      = ROOT.parents[1]  # .../Submit
CARDIAC_CSV = SUBMIT / "NC_revision" / "expanded_cardiac_wide_table.csv"
WHU_CSV     = SUBMIT / "NC_revision" / "_tmp_whu_primary_for_lgdi.csv"
FLUNET_CSV  = SUBMIT / "external_data" / "flunet" / "flunet_china_2009_2024.csv"
OUT         = ROOT / "outputs" / "lgdi_flu_ref"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260513

# ── Cohort / group config ────────────────────────────────────────────────────
COMORBIDITIES = ["Cardiovascular", "Hypertension", "Diabetes",
                 "Cerebrovascular", "Renal", "Respiratory"]
RESPIRATORY = "Respiratory"
COMORBIDITY_PATTERNS = {
    "Cardiovascular":  r"冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病",
    "Hypertension":    r"高血压",
    "Diabetes":        r"糖尿病|血糖",
    "Cerebrovascular": r"脑梗|脑出血|脑血管|脑卒中|中风|腔隙性",
    "Renal":           r"肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏",
    "Respiratory":     r"肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染|呼吸道感染",
}
LAB_COLS_MAP = {
    "白细胞":   "lab_WBC",
    "超敏C反应蛋白": "lab_CRP",
    "血红蛋白":  "lab_HGB",
    "白蛋白":   "lab_ALB",
    "肌酐":     "lab_CREA",
    "空腹血糖":  "lab_GLU",
    "钾":       "lab_K",
    "钠":       "lab_Na",
}
FLU_POSITIVITY_THRESHOLD = 0.30   # FluNet China positivity > this → event week

# ── 1. Load & prep data ──────────────────────────────────────────────────────
def _prep_common(df: pd.DataFrame, mrn_col: str, admit_col: str, dis_col: str,
                 diag_col: str | None, prev_diag_col: str | None) -> pd.DataFrame:
    """Shared admission prep for both cardiac and WHU primary."""
    df = df.copy()
    df["病案号_norm"]  = df[mrn_col].fillna("").astype(str).str.strip()
    df["admit_dt"]    = pd.to_datetime(df[admit_col], errors="coerce")
    df["discharge_dt"] = pd.to_datetime(df[dis_col],  errors="coerce")
    df = df.dropna(subset=["admit_dt"]).copy()
    df["los_days"] = (df["discharge_dt"] - df["admit_dt"]).dt.total_seconds() / 86400
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan
    df = df.sort_values(["病案号_norm", "admit_dt", "discharge_dt"]).copy()
    df["prev_dis"]  = df.groupby("病案号_norm", dropna=False)["discharge_dt"].shift(1)
    df["gap_days"]  = (df["admit_dt"] - df["prev_dis"]).dt.total_seconds() / 86400
    df.loc[(df["gap_days"] < 0) | (df["gap_days"] > 3650), "gap_days"] = np.nan
    df["year"]       = df["admit_dt"].dt.year
    df["week_start"] = (df["admit_dt"] -
                        pd.to_timedelta(df["admit_dt"].dt.weekday, unit="D")).dt.normalize()
    all_dx = (df[diag_col].fillna("").astype(str) if diag_col else pd.Series("", index=df.index))
    if prev_diag_col and prev_diag_col in df.columns:
        all_dx = all_dx + " " + df[prev_diag_col].fillna("").astype(str)
    for g, pat in COMORBIDITY_PATTERNS.items():
        df[g] = all_dx.str.contains(pat, case=False, regex=True, na=False).astype(bool)
    for src, tgt in LAB_COLS_MAP.items():
        if src in df.columns:
            df[tgt] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[tgt] = np.nan
    return df


def load_cardiac() -> pd.DataFrame:
    df = pd.read_csv(CARDIAC_CSV, encoding="utf-8-sig", low_memory=False,
                     dtype={"病案号": str})
    return _prep_common(df, "病案号", "入院时间", "出院时间", "主要诊断", "上次诊断")


def load_whu_primary() -> pd.DataFrame:
    df = pd.read_csv(WHU_CSV, encoding="utf-8-sig", low_memory=False,
                     dtype={"病案号": str})
    return _prep_common(df, "病案号", "入院时间", "出院时间", "主要诊断", "上次诊断")


# ── 2. Feature engineering ───────────────────────────────────────────────────
def _seasonal(admit_dt: pd.Series) -> pd.DataFrame:
    m  = admit_dt.dt.month
    dw = admit_dt.dt.weekday
    dy = admit_dt.dt.dayofyear
    return pd.DataFrame({
        "year":      admit_dt.dt.year.astype(float),
        "month_sin": np.sin(2 * np.pi * m / 12.0),
        "month_cos": np.cos(2 * np.pi * m / 12.0),
        "dow_sin":   np.sin(2 * np.pi * dw / 7.0),
        "dow_cos":   np.cos(2 * np.pi * dw / 7.0),
        "doy_sin":   np.sin(2 * np.pi * dy / 365.25),
        "doy_cos":   np.cos(2 * np.pi * dy / 365.25),
    }, index=admit_dt.index)


def build_features(df: pd.DataFrame):
    w = df.sort_values(["病案号_norm", "admit_dt", "discharge_dt"]).copy()
    g = w.groupby("病案号_norm", dropna=False, sort=False)
    w["visit_order"]        = g.cumcount()
    w["first_admit_dt"]     = g["admit_dt"].transform("min")
    w["days_since_first"]   = (w["admit_dt"] - w["first_admit_dt"]).dt.total_seconds() / 86400
    w["prior_los_mean"]     = g["los_days"].transform(lambda s: s.shift(1).expanding().mean())
    w["prior_los_std"]      = g["los_days"].transform(lambda s: s.shift(1).expanding().std())
    w["prior_los_last"]     = g["los_days"].shift(1)
    w["prior_gap_mean"]     = g["gap_days"].transform(lambda s: s.shift(1).expanding().mean())
    w["prior_gap_std"]      = g["gap_days"].transform(lambda s: s.shift(1).expanding().std())
    w["prior_gap_last"]     = g["gap_days"].shift(1)
    w["next_admit_dt"]      = g["admit_dt"].shift(-1)
    w["next_los_days"]      = g["los_days"].shift(-1)
    next_gap = (w["next_admit_dt"] - w["discharge_dt"]).dt.total_seconds() / 86400
    w["next_gap_days"]      = next_gap.where((next_gap >= 0) & (next_gap <= 3650))
    for col, vals in _seasonal(w["admit_dt"]).items():
        w[col] = vals.values
    feat = ["visit_order", "days_since_first",
            "prior_los_mean", "prior_los_std", "prior_los_last",
            "prior_gap_mean", "prior_gap_std", "prior_gap_last",
            "year", "month_sin", "month_cos", "dow_sin", "dow_cos",
            "doy_sin", "doy_cos"] + COMORBIDITIES + list(LAB_COLS_MAP.values())
    for c in feat:
        if c in COMORBIDITIES:
            w[c] = w[c].astype(bool).astype(float)
        else:
            w[c] = pd.to_numeric(w[c], errors="coerce").astype(float)
    return w, feat


# ── 3. XGBoost training ──────────────────────────────────────────────────────
def _xgb():
    return xgb.XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.02, subsample=0.7,
        colsample_bytree=0.6, min_child_weight=15, reg_alpha=1.0, reg_lambda=3.0,
        gamma=0.1, tree_method="hist", device="cpu", random_state=RANDOM_STATE,
        n_jobs=4, objective="reg:squarederror",
    )


def train_models(enriched: pd.DataFrame, baseline_mask: pd.Series, feat: list[str]):
    base = enriched[baseline_mask.reindex(enriched.index, fill_value=False)].copy()
    models, audits = {}, {}
    for tgt in ["next_los_days", "next_gap_days"]:
        sub = base.dropna(subset=[tgt]).copy()
        m = _xgb()
        if len(sub) >= 200:
            gkf = GroupKFold(n_splits=5)
            groups = sub["病案号_norm"].fillna("").astype(str)
            maes, r2s = [], []
            for tr_idx, te_idx in gkf.split(sub, groups=groups):
                Xtr = sub.iloc[tr_idx][feat].fillna(np.nan)
                ytr = sub.iloc[tr_idx][tgt].astype(float)
                Xte = sub.iloc[te_idx][feat].fillna(np.nan)
                yte = sub.iloc[te_idx][tgt].astype(float)
                m_fold = _xgb()
                m_fold.fit(Xtr, ytr)
                pred = m_fold.predict(Xte)
                maes.append(mean_absolute_error(yte, pred))
                r2s.append(r2_score(yte, pred))
            audits[tgt] = {"cv_mae": float(np.mean(maes)), "cv_r2": float(np.mean(r2s)), "n": len(sub)}
        else:
            audits[tgt] = {"cv_mae": math.nan, "cv_r2": math.nan, "n": len(sub)}
        m.fit(sub[feat].fillna(np.nan), sub[tgt].astype(float))
        models[tgt] = m
        print(f"  [XGB] {tgt}: n={len(sub)}  cv_mae={audits[tgt]['cv_mae']:.2f}  cv_r2={audits[tgt]['cv_r2']:.3f}")
    # Predict on full dataset
    for tgt, m in models.items():
        valid_mask = enriched[feat].notna().all(axis=1)
        enriched[f"pred_{tgt}"] = np.nan
        enriched.loc[valid_mask, f"pred_{tgt}"] = m.predict(enriched.loc[valid_mask, feat].fillna(np.nan))
    return models, audits, enriched


# ── 4. Baseline residual scale ────────────────────────────────────────────────
def residual_scale(enriched: pd.DataFrame, baseline_mask: pd.Series) -> dict:
    base = enriched[baseline_mask.reindex(enriched.index, fill_value=False)].copy()
    stats = {}
    for g in COMORBIDITIES:
        stats[g] = {}
        sub = base[base[g].astype(bool)]
        for tgt in ["next_los_days", "next_gap_days"]:
            valid = sub.dropna(subset=[tgt, f"pred_{tgt}"])
            mae = float(np.mean(np.abs(valid[tgt] - valid[f"pred_{tgt}"]))) if len(valid) else 1.0
            stats[g][tgt] = {"mae_scale": max(mae, 0.01)}
    return stats


# ── 5. Rolling 4-week windows ─────────────────────────────────────────────────
def group_score(frame: pd.DataFrame, stats: dict, min_n: int = 10) -> dict:
    scores = {}
    for g in COMORBIDITIES:
        sub = frame[frame[g].astype(bool)].dropna(subset=["next_los_days", f"pred_next_los_days"])
        if len(sub) < min_n:
            scores[g] = math.nan; continue
        los_res = sub["next_los_days"] - sub[f"pred_next_los_days"]
        gap_sub = frame[frame[g].astype(bool)].dropna(subset=["next_gap_days", f"pred_next_gap_days"])
        gap_res = gap_sub["next_gap_days"] - gap_sub[f"pred_next_gap_days"]
        mase_los  = float(np.mean(np.abs(los_res)))  / max(stats[g]["next_los_days"]["mae_scale"], 0.01)
        mase_gap  = float(np.mean(np.abs(gap_res)))  / max(stats[g]["next_gap_days"]["mae_scale"], 0.01)
        # direction: +1 for longer LOS (sicker), -1 for shorter gap (returning sooner)
        sign_los = float(np.mean(los_res)) / max(abs(float(np.mean(los_res))), 1e-9)
        sign_gap = float(np.mean(gap_res)) / max(abs(float(np.mean(gap_res))), 1e-9)
        signed_los = sign_los * mase_los
        signed_gap = -1.0 * sign_gap * mase_gap
        scores[g] = float(np.mean([signed_los, signed_gap]))
    return scores


def make_windows(enriched: pd.DataFrame, stats: dict,
                 monitor_start: pd.Timestamp, monitor_end: pd.Timestamp):
    rows = []
    for anchor in pd.date_range(monitor_start, monitor_end, freq="W-MON"):
        start = anchor - pd.Timedelta(days=21)
        end   = anchor + pd.Timedelta(days=6)
        frame = enriched[(enriched["admit_dt"] >= start) & (enriched["admit_dt"] <= end)]
        if len(frame) < 30:
            rows.append({"week_start": anchor, "n": len(frame), "valid": False,
                         "lgdi": math.nan, "resp_score": math.nan,
                         "mean_other_score": math.nan})
            continue
        sc = group_score(frame, stats)
        resp = sc.get(RESPIRATORY, math.nan)
        others = [v for k, v in sc.items() if k != RESPIRATORY and math.isfinite(v if v is not None else math.nan)]
        lgdi = resp - float(np.mean(others)) if others and math.isfinite(resp) else math.nan
        row = {"week_start": anchor, "n": len(frame), "valid": math.isfinite(lgdi),
               "lgdi": lgdi, "resp_score": resp, "mean_other_score": float(np.mean(others)) if others else math.nan}
        for g in COMORBIDITIES:
            row[f"score_{g}"] = sc.get(g, math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ── 6. FluNet event weeks ────────────────────────────────────────────────────
def load_flunet_events(monitor_start: pd.Timestamp, monitor_end: pd.Timestamp):
    flu = pd.read_csv(FLUNET_CSV, low_memory=False)
    flu["week_start"] = pd.to_datetime(flu["ISO_WEEKSTARTDATE"], errors="coerce")
    flu["spec"]       = pd.to_numeric(flu.get("SPEC_PROCESSED_NB"), errors="coerce")
    flu["inf_all"]    = pd.to_numeric(flu.get("INF_ALL"), errors="coerce")
    flu["pos_rate"]   = flu["inf_all"] / flu["spec"].replace(0, np.nan)
    flu = flu.dropna(subset=["week_start", "pos_rate"]).copy()
    flu = flu[flu["week_start"].between(monitor_start, monitor_end)].copy()
    flu["event_week"] = flu["pos_rate"] > FLU_POSITIVITY_THRESHOLD
    return flu[["week_start", "pos_rate", "event_week"]].reset_index(drop=True)


# ── 7. S1–S4 multi-group consensus rules ────────────────────────────────────
def compute_s_rules(timeline: pd.DataFrame, flunet: pd.DataFrame):
    """Evaluate S1–S4 strategies against FluNet event weeks."""
    merged = timeline[timeline["valid"].astype(bool)].merge(
        flunet[["week_start", "event_week", "pos_rate"]],
        on="week_start", how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    # Baseline thresholds (2016-2017, pre-peak)
    baseline_weeks = merged[merged["week_start"] < pd.Timestamp("2017-12-01")]
    thr_resp  = float(baseline_weeks["resp_score"].quantile(0.75)) if len(baseline_weeks) > 4 else 0.5
    thr_other = {g: float(baseline_weeks[f"score_{g}"].quantile(0.75)) if len(baseline_weeks) > 4 else 0.5
                 for g in COMORBIDITIES}
    # Count groups above baseline threshold per week
    for g in COMORBIDITIES:
        merged[f"above_{g}"] = merged[f"score_{g}"] > thr_other[g]
    merged["groups_above"] = merged[[f"above_{g}" for g in COMORBIDITIES]].sum(axis=1)
    merged["month"] = merged["week_start"].dt.month
    # Strategy definitions
    strategies = {
        "S1": merged["resp_score"] > thr_resp,
        "S2": merged["groups_above"] >= 2,
        "S3": (merged["groups_above"] >= 2) & merged["month"].isin([1,2,3,4,10,11,12]),
        "S4": (merged["groups_above"] >= 2) & merged["month"].isin([1,2,3,11,12]) &
              (merged["groups_above"].rolling(2, min_periods=2).min() >= 2),
    }
    rows = []
    event = merged["event_week"].astype(bool)
    n_total = len(merged)
    n_event = int(event.sum())
    n_nonevent = n_total - n_event
    for name, alert_mask in strategies.items():
        alert = alert_mask.astype(bool)
        tp = int((alert & event).sum())
        fp = int((alert & ~event).sum())
        fn = int((~alert & event).sum())
        tn = int((~alert & ~event).sum())
        sens = tp / n_event      if n_event else math.nan
        ppv  = tp / (tp + fp)   if (tp + fp) else math.nan
        far  = fp / n_nonevent   if n_nonevent else math.nan
        rows.append({"strategy": name, "n_monitor": n_total, "n_event": n_event,
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "sensitivity": round(sens, 4), "PPV": round(ppv, 4),
                     "FAR": round(far, 4)})
    return pd.DataFrame(rows)


# ── 8. Bonferroni-corrected lag–Spearman ────────────────────────────────────
def lag_spearman_bonferroni(timeline: pd.DataFrame, flunet: pd.DataFrame, n_lags: int = 5):
    merged = timeline[timeline["valid"].astype(bool)].merge(
        flunet[["week_start", "pos_rate"]], on="week_start", how="inner"
    ).sort_values("week_start").copy()
    rows = []
    alpha_corrected = 0.05 / n_lags
    for lag in range(n_lags):
        test = merged.copy()
        test["flu_future"] = test["pos_rate"].shift(-lag)
        test = test.dropna(subset=["lgdi", "flu_future"])
        if len(test) >= 8:
            rho, p = spearmanr(test["lgdi"], test["flu_future"])
        else:
            rho, p = math.nan, math.nan
        p_corr = min(float(p) * n_lags, 1.0) if math.isfinite(float(p) if p else math.nan) else math.nan
        rows.append({"lag": lag, "n": len(test), "rho": round(float(rho), 4),
                     "p_raw": round(float(p), 6) if math.isfinite(float(p) if p else math.nan) else math.nan,
                     "p_bonferroni": round(p_corr, 6) if math.isfinite(p_corr) else math.nan,
                     "alpha_bonferroni": alpha_corrected,
                     "significant": bool(p_corr < alpha_corrected) if math.isfinite(p_corr) else False})
    return pd.DataFrame(rows)


# ── 9. Pearson reference vector (flu peak Jan–Feb 2018) ──────────────────────
def build_flu_reference_pearson(df: pd.DataFrame):
    """
    Compute 13-dim Pearson reference vector from respiratory-coded WHU
    admissions in Jan–Feb 2018 (FluNet peak season).  Also compute
    Q4 2017 group profiles and correlation against this reference.
    """
    LAB_FEATS = list(LAB_COLS_MAP.values())
    ALL_FEATS  = ["los_days", "gap_days"] + LAB_FEATS   # 10 available; extend if utilization cols present

    ref_mask = (df["admit_dt"].between("2018-01-01", "2018-02-28") &
                df[RESPIRATORY].astype(bool))
    ref_adm  = df[ref_mask].copy()
    n_ref    = len(ref_adm)
    ref_vec  = ref_adm[ALL_FEATS].mean().to_dict()

    def pearson_corr(group_vec: dict, ref: dict) -> float:
        keys = [k for k in ref if math.isfinite(ref.get(k, math.nan)) and math.isfinite(group_vec.get(k, math.nan))]
        if len(keys) < 3:
            return math.nan
        x = np.array([group_vec[k] for k in keys])
        y = np.array([ref[k] for k in keys])
        x -= x.mean(); y -= y.mean()
        denom = np.sqrt((x**2).sum() * (y**2).sum())
        return float(np.dot(x, y) / denom) if denom > 0 else math.nan

    # Q4 2017 group profiles
    q4_mask = df["admit_dt"].between("2017-10-01", "2017-12-31")
    rows = []
    for g in COMORBIDITIES:
        gm = q4_mask & df[g].astype(bool)
        gvec = df[gm][ALL_FEATS].mean().to_dict()
        r = pearson_corr(gvec, ref_vec)
        rows.append({"group": g, "n_q4_2017": int(gm.sum()), "pearson_r_vs_flu_ref": round(r, 4)})
    corr_df = pd.DataFrame(rows)
    return ref_vec, n_ref, corr_df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    MONITOR_START  = pd.Timestamp("2016-01-01")
    MONITOR_END    = pd.Timestamp("2019-06-30")   # stop before 2019-07 policy gap
    BASELINE_START = 2012
    BASELINE_END   = 2016

    print("[1/7] Loading cardiac wide table (LGDI substrate)...")
    cardiac = load_cardiac()
    print(f"  Cardiac: {len(cardiac):,} admissions, {cardiac['病案号_norm'].nunique():,} patients")
    print(f"  Date range: {cardiac['admit_dt'].min().date()} → {cardiac['admit_dt'].max().date()}")
    yc = cardiac["year"].value_counts().sort_index()
    print("  Year counts (2012-2019):", dict(yc[(yc.index >= 2012) & (yc.index <= 2019)]))

    print("\n[2/7] Building features...")
    enriched, feat = build_features(cardiac)

    print("[3/7] Training XGBoost on 2012–2016 baseline...")
    baseline_mask = enriched["year"].between(BASELINE_START, BASELINE_END)
    models, audits, enriched = train_models(enriched, baseline_mask, feat)
    print(f"  Baseline admissions: {baseline_mask.sum():,}")

    print("[4/7] Computing baseline residual scale...")
    stats = residual_scale(enriched, baseline_mask)

    print("[5/7] Building rolling 4-week windows (2016–2019)...")
    timeline = make_windows(enriched, stats, MONITOR_START, MONITOR_END)
    print(f"  Windows: {len(timeline)}, valid: {timeline['valid'].sum()}")

    print("[6/7] Loading FluNet event weeks (pos > 0.30, 2016-2019)...")
    flunet = load_flunet_events(MONITOR_START, MONITOR_END)
    n_event   = int(flunet["event_week"].sum())
    n_monitor = len(flunet)
    print(f"  FluNet event weeks: {n_event}/{n_monitor} (threshold >{FLU_POSITIVITY_THRESHOLD})")
    print("  Event weeks:")
    print(flunet[flunet["event_week"]][["week_start","pos_rate"]].to_string(index=False))

    # S1–S4 operating characteristics
    s_rules = compute_s_rules(timeline, flunet)
    print("\nS1–S4 operating characteristics vs FluNet event weeks:")
    print(s_rules[["strategy","n_monitor","n_event","sensitivity","PPV","FAR"]].to_string(index=False))

    # Bonferroni lag–Spearman
    lag_df = lag_spearman_bonferroni(timeline, flunet)
    print("\nBonferroni lag–Spearman (resp_score vs FluNet):")
    print(lag_df.to_string(index=False))

    print("\n[7/7] Building flu-reference Pearson vector from WHU primary (Jan–Feb 2018 respiratory)...")
    whu = load_whu_primary()
    ref_vec, n_ref, corr_df = build_flu_reference_pearson(whu)
    print(f"  WHU primary: {len(whu):,} admissions. Reference n={n_ref}")
    print(corr_df.to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────────
    timeline_out = timeline.copy()
    timeline_out["week_start"] = timeline_out["week_start"].astype(str)
    timeline_out.to_csv(OUT / "lgdi_flu_ref_timeline.csv", index=False, encoding="utf-8-sig")
    flunet.to_csv(OUT / "flunet_event_weeks_2016_2019.csv", index=False, encoding="utf-8-sig")
    s_rules.to_csv(OUT / "lgdi_flu_ref_s1s4_operating_characteristics.csv",
                   index=False, encoding="utf-8-sig")
    lag_df.to_csv(OUT / "lgdi_flu_ref_lag_spearman_bonferroni.csv",
                  index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUT / "lgdi_flu_ref_pearson_profile_q4_2017.csv",
                   index=False, encoding="utf-8-sig")

    sig_lags = lag_df[lag_df["significant"].astype(bool)][["lag","rho","p_bonferroni"]].to_dict(orient="records")
    best_s   = s_rules.sort_values("PPV", ascending=False).head(1).to_dict(orient="records")
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "design": "FluNet-anchored historical flu reference (pre-specifiable, non-circular)",
        "lgdi_input_data": str(CARDIAC_CSV),
        "lgdi_n_admissions": int(len(cardiac)),
        "lgdi_n_patients": int(cardiac["病案号_norm"].nunique()),
        "baseline_window": f"{BASELINE_START}-{BASELINE_END}",
        "monitor_window": f"{MONITOR_START.date()} to {MONITOR_END.date()}",
        "flunet_threshold": FLU_POSITIVITY_THRESHOLD,
        "n_event_weeks": n_event,
        "n_monitor_weeks": n_monitor,
        "xgb_audit": audits,
        "s_rules_results": s_rules.to_dict(orient="records"),
        "bonferroni_significant_lags": sig_lags,
        "s_rules_best_ppv": best_s[0] if best_s else {},
        "flu_reference_vector_n": n_ref,
        "flu_reference_period": "2018-01-01 to 2018-02-28 (WHU primary respiratory admissions)",
        "pearson_results": corr_df.to_dict(orient="records"),
        "note": (
            "LGDI component uses cardiac wide table (dense repeat admissions for XGBoost). "
            "Event weeks defined by FluNet China positivity > 0.30 — pre-specifiable "
            "without COVID-era foreknowledge. Monitor stops at 2019-06-30 to avoid the "
            "documented 2019-07 to 2020-04 policy-induced no-admission gap. "
            f"Pearson reference vector uses n={n_ref} respiratory WHU primary admissions "
            "in Jan-Feb 2018 (FluNet peak), replacing the n=19 post-hoc COVID reference."
        ),
    }
    (OUT / "lgdi_flu_ref_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] All outputs → {OUT}")
    print(json.dumps({k: summary[k] for k in [
        "lgdi_n_admissions", "n_event_weeks", "n_monitor_weeks",
        "bonferroni_significant_lags", "flu_reference_vector_n",
        "s_rules_best_ppv"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
