#!/usr/bin/env python3
r"""Two-stage alert strategy: exclude influenza-explained signals → evaluate residual COVID PPV.

Stage 1 (Flu filter):
  Apply the LGDI multi-group consensus rule (S2 ≥2 groups or S4 season-sustained)
  to flag weeks that are "flu-explained" by the healthcare-utilization residual.

Stage 2 (Residual COVID check):
  Among remaining non-flu-explained alert weeks, compute overlap with COVID-coded
  event windows and report PPV / sensitivity / false-alarm rate.

Cohorts evaluated:
  A. WHU primary 32k (2016-2020): FluNet-anchored, COVID window Jan–Apr 2020 (n=19 ref)
  B. Cardiac expanded 42k (2016-2024): FluNet merged by ISO week, COVID window Dec 2022–Jan 2023

Outputs:
  NC_revision/two_stage_results/
    two_stage_whu_summary.json
    two_stage_whu_week_audit.csv
    two_stage_cardiac_summary.json
    two_stage_cardiac_week_audit.csv
    two_stage_combined_report.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "two_stage_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Inputs ──────────────────────────────────────────────────────────────

# WHU primary: already has FluNet labels + alert_s1..s4 flags
WHU_VAL_CSV = BASE / "lgdi_results" / "lgdi_whu_influenza_validation.csv"

# Cardiac expanded: LGDI weekly timeline (COVID event_window True for Dec 2022–Jan 2023)
CARDIAC_LGDI_CSV = BASE / "lgdi_results" / "lgdi_rolling4_weekly.csv"

# FluNet China: for merging with cardiac cohort
FLUNET_CSV = BASE.parent / "external_data" / "flunet" / "flunet_china_2009_2024.csv"

# ── Event windows ───────────────────────────────────────────────────────

# WHU COVID period (Wuhan outbreak, early 2020)
WHU_COVID_START = pd.Timestamp("2020-01-01")
WHU_COVID_END = pd.Timestamp("2020-04-30")

# Cardiac COVID period (China reopening wave, Dec 2022–Jan 2023)
CARDIAC_COVID_START = pd.Timestamp("2022-12-01")
CARDIAC_COVID_END = pd.Timestamp("2023-01-31")

# ── Helper functions ────────────────────────────────────────────────────

def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def compute_operating_chars(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "sensitivity": safe_div(tp, tp + fn),
        "ppv": safe_div(tp, tp + fp),
        "specificity": safe_div(tn, tn + fp),
        "false_alarm_rate": safe_div(fp, fp + tn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


# ══════════════════════════════════════════════════════════════════════════
#  COHORT A — WHU Primary 32k (FluNet-anchored)
# ══════════════════════════════════════════════════════════════════════════

def analyse_whu_two_stage() -> dict:
    whu = pd.read_csv(WHU_VAL_CSV, low_memory=False)
    whu["week_start_dt"] = pd.to_datetime(whu["week_start"])

    # ── Stage 1: Flu detection ──────────────────────────────────────────
    # S2 = multi-group consensus (≥2 chronic-disease groups above baseline)
    # S4 = S2 + sustained 2 weeks + Nov–Mar core season
    # flu_event_window = FluNet ground truth (positivity > threshold, Sep–Apr)

    # Flu-explained = S4 alert fires (optimized rule)
    whu["flu_explained_s4"] = whu["alert_s4"].astype(bool)
    whu["flu_explained_s2"] = whu["alert_s2"].astype(bool)

    # ── Stage 2: Residual COVID check ───────────────────────────────────
    # Define COVID event window
    whu["covid_event_window"] = whu["week_start_dt"].between(
        WHU_COVID_START, WHU_COVID_END
    )

    # Define Stage-2 alert: resp_score exceeds baseline mean+1.5SD AND not flu-explained
    # (using the same threshold as the existing S1 single-series rule)
    whu["raw_alert"] = whu["alert_s1"].astype(bool)  # resp_score mean+1.5SD

    # Two-stage alert (S4 flu filter)
    whu["alert_two_stage_s4"] = whu["raw_alert"] & ~whu["flu_explained_s4"]
    # Two-stage alert (S2 flu filter)
    whu["alert_two_stage_s2"] = whu["raw_alert"] & ~whu["flu_explained_s2"]

    # ── Counts ──────────────────────────────────────────────────────────
    results = {
        "cohort": "WHU_primary_32k",
        "n_weeks_total": int(len(whu)),
        "date_range": f"{whu['week_start_dt'].min().date()} to {whu['week_start_dt'].max().date()}",
        "flu_ground_truth_weeks": int(whu["flu_event_window"].sum()),
        "covid_event_weeks": int(whu["covid_event_window"].sum()),
    }

    for flu_rule, flu_col in [("S2_consensus", "flu_explained_s2"), ("S4_season_sustained", "flu_explained_s4")]:
        for alert_col, alert_label in [
            ("raw_alert", "single_stage_resp_threshold"),
            (f"alert_two_stage_{'s4' if 's4' in flu_col else 's2'}", "two_stage_residual"),
        ]:
            # Evaluate against FluNet ground truth (flu validation)
            flu_true = whu["flu_event_window"].astype(bool)
            flu_pred = whu[alert_col].astype(bool)
            flu_tp = int((flu_pred & flu_true).sum())
            flu_fp = int((flu_pred & ~flu_true).sum())
            flu_fn = int((~flu_pred & flu_true).sum())
            flu_tn = int((~flu_pred & ~flu_true).sum())
            key_flu = f"{alert_label}_vs_FluNet__flu_filter_{flu_rule}"
            results[key_flu] = compute_operating_chars(flu_tp, flu_fp, flu_fn, flu_tn)

            # Evaluate against COVID event window
            covid_true = whu["covid_event_window"].astype(bool)
            covid_pred = whu[alert_col].astype(bool)
            covid_tp = int((covid_pred & covid_true).sum())
            covid_fp = int((covid_pred & ~covid_true).sum())
            covid_fn = int((~covid_pred & covid_true).sum())
            covid_tn = int((~covid_pred & ~covid_true).sum())
            key_covid = f"{alert_label}_vs_COVID__flu_filter_{flu_rule}"
            results[key_covid] = compute_operating_chars(covid_tp, covid_fp, covid_fn, covid_tn)

        # ── Also check: among ALL flu-explained weeks, how many overlap FluNet? ──
        flu_pred_only = whu[flu_col].astype(bool)
        flu_true_only = whu["flu_event_window"].astype(bool)
        ftp = int((flu_pred_only & flu_true_only).sum())
        ffp = int((flu_pred_only & ~flu_true_only).sum())
        ffn = int((~flu_pred_only & flu_true_only).sum())
        ftn = int((~flu_pred_only & ~flu_true_only).sum())
        results[f"flu_filter_{flu_rule}_self_evaluation"] = compute_operating_chars(ftp, ffp, ffn, ftn)

    # ── Export week-level audit ─────────────────────────────────────────
    audit_cols = [
        "week_start", "label", "season_id",
        "positivity", "flu_event_window",
        "resp_score", "z_resp_score", "lgdi",
        "alert_s1", "alert_s2", "alert_s4",
        "flu_explained_s2", "flu_explained_s4",
        "raw_alert", "alert_two_stage_s2", "alert_two_stage_s4",
        "covid_event_window",
    ]
    audit = whu[[c for c in audit_cols if c in whu.columns]].copy()
    audit.to_csv(OUT_DIR / "two_stage_whu_week_audit.csv", index=False)

    return results


# ══════════════════════════════════════════════════════════════════════════
#  COHORT B — Cardiac Expanded 42k (FluNet-merged)
# ══════════════════════════════════════════════════════════════════════════

def load_flunet_weekly() -> pd.DataFrame:
    """Compute weekly FluNet China positivity, return [week_start, positivity, flu_season]."""
    fn = pd.read_csv(FLUNET_CSV, low_memory=False)
    fn = fn[fn["COUNTRY_CODE"] == "CHN"].copy()
    fn["week_start"] = pd.to_datetime(fn["ISO_WEEKSTARTDATE"])
    fn = fn[fn["week_start"].between("2009-01-01", "2025-01-01")]
    agg = fn.groupby("week_start", as_index=False).agg(
        inf_all=("INF_ALL", "sum"),
        spec_processed=("SPEC_PROCESSED_NB", "sum"),
    )
    agg["positivity"] = agg["inf_all"] / agg["spec_processed"].replace({0: float("nan")})
    agg["positivity"] = agg["positivity"].fillna(0.0)

    # FluNet ground-truth event weeks: same definition as lgdi_influenza_validation.py
    pre = agg[agg["week_start"].dt.year.between(2012, 2019)]
    mu = pre["positivity"].mean()
    sd = pre["positivity"].std()
    threshold = mu + sd
    month = agg["week_start"].dt.month
    in_season = (month >= 9) | (month <= 4)
    agg["flu_event_window"] = (agg["positivity"] > threshold) & in_season
    agg["flu_threshold"] = threshold
    agg["flu_baseline_mean"] = mu
    agg["flu_baseline_sd"] = sd
    return agg


def define_cardiac_flu_filter(cardiac: pd.DataFrame, flunet: pd.DataFrame) -> pd.DataFrame:
    """
    For the cardiac cohort, we don't have pre-computed alert_s2/s4.
    Instead we approximate a flu filter using the cardiac LGDI z-scores:
      - Compute cardiac-group z-scores from cardiac baseline (2016-2018)
      - Apply multi-group consensus: ≥2 groups above each group's mean+1.5SD
      - Apply season restriction (Nov-Mar)
      - Apply sustained 2-week requirement

    Returns cardiac with added flu-filter columns.
    """
    df = cardiac.copy()
    groups = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
    score_cols = [f"score_{g}" for g in groups]
    available_score_cols = [c for c in score_cols if c in df.columns]
    available_groups = [c.replace("score_", "") for c in available_score_cols]

    # Compute per-group baseline mean / SD from 2016-2018
    baseline_mask = df["window_anchor"].str.startswith(("2016", "2017", "2018"))
    baseline = df[baseline_mask]
    group_stats: dict[str, dict[str, float]] = {}
    for g in available_groups:
        col = f"score_{g}"
        vals = pd.to_numeric(baseline[col], errors="coerce").dropna()
        group_stats[g] = {
            "mean": float(vals.mean()) if len(vals) else 0.0,
            "sd": float(vals.std()) if len(vals) > 1 else 0.001,
        }
        if group_stats[g]["sd"] < 0.001:
            group_stats[g]["sd"] = 0.001

    # Compute per-group z-score
    for g in available_groups:
        col = f"score_{g}"
        mean = group_stats[g]["mean"]
        sd = group_stats[g]["sd"]
        df[f"z_{g}"] = (pd.to_numeric(df[col], errors="coerce") - mean) / sd

    # Multi-group consensus: ≥2 groups above baseline mean+1.5SD
    above_cols = []
    for g in available_groups:
        above_cols.append(f"above_{g}")
        df[f"above_{g}"] = df[f"z_{g}"] > 1.5
    df["n_groups_above"] = df[above_cols].sum(axis=1)
    df["consensus_2plus"] = df["n_groups_above"] >= 2

    # Month filter
    df["_month"] = pd.to_datetime(df["window_anchor"]).dt.month
    df["_in_core_season"] = df["_month"].between(11, 12) | df["_month"].between(1, 3)

    # Sustained 2-week requirement
    df = df.sort_values("window_anchor").copy()
    df["_consensus_prev"] = df["consensus_2plus"].shift(1, fill_value=False)
    df["_consensus_next"] = df["consensus_2plus"].shift(-1, fill_value=False)
    df["sustained_consensus"] = df["consensus_2plus"] & (df["_consensus_prev"] | df["_consensus_next"])

    # Flu filter S2 (basic consensus, no season/sustain)
    df["flu_filter_s2"] = df["consensus_2plus"]
    # Flu filter S4 (consensus + season + sustained)
    df["flu_filter_s4"] = df["consensus_2plus"] & df["_in_core_season"] & df["sustained_consensus"]

    # Drop temporary columns
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def analyse_cardiac_two_stage() -> dict:
    cardiac = pd.read_csv(CARDIAC_LGDI_CSV, low_memory=False)
    flunet = load_flunet_weekly()

    # Merge FluNet with cardiac by ISO week start
    cardiac["week_start_dt"] = pd.to_datetime(cardiac["window_anchor"])
    # Round cardiac week to nearest Monday to align with FluNet
    cardiac["week_start_dt"] = cardiac["week_start_dt"] - pd.to_timedelta(
        cardiac["week_start_dt"].dt.weekday, unit="D"
    )

    # Build flu filters on cardiac
    cardiac = define_cardiac_flu_filter(cardiac, flunet)

    # Merge FluNet labels
    cardiac = cardiac.merge(
        flunet[["week_start", "positivity", "flu_event_window", "flu_threshold"]],
        left_on="week_start_dt", right_on="week_start", how="left"
    )
    cardiac["positivity"] = cardiac["positivity"].fillna(0.0)
    cardiac["flu_event_window"] = cardiac["flu_event_window"].fillna(False).astype(bool)

    # ── Stage 2: COVID residual ─────────────────────────────────────────
    # COVID event window
    cardiac["covid_event_window"] = cardiac["week_start_dt"].between(
        CARDIAC_COVID_START, CARDIAC_COVID_END
    )

    # Raw alert: lgdi exceeds baseline mean+1.5SD
    baseline_lgdi = cardiac[cardiac["week_start_dt"].dt.year.between(2016, 2018)]
    lgdi_mean = float(baseline_lgdi["lgdi"].mean())
    lgdi_sd = float(baseline_lgdi["lgdi"].std())
    lgdi_threshold = lgdi_mean + 1.5 * lgdi_sd
    cardiac["raw_alert_lgdi"] = cardiac["lgdi"] > lgdi_threshold

    # Also define alert based on resp_score (more like the WHU approach)
    baseline_resp = cardiac[cardiac["week_start_dt"].dt.year.between(2016, 2018)]
    resp_mean = float(baseline_resp["resp_score"].mean())
    resp_sd = float(baseline_resp["resp_score"].std())
    if resp_sd < 0.001:
        resp_sd = 0.001
    resp_threshold = resp_mean + 1.5 * resp_sd
    cardiac["raw_alert_resp"] = cardiac["resp_score"] > resp_threshold

    # Two-stage alerts
    for flu_col, flu_name in [("flu_filter_s2", "S2"), ("flu_filter_s4", "S4")]:
        cardiac[f"alert_two_stage_lgdi_{flu_name}"] = cardiac["raw_alert_lgdi"] & ~cardiac[flu_col]
        cardiac[f"alert_two_stage_resp_{flu_name}"] = cardiac["raw_alert_resp"] & ~cardiac[flu_col]

    # ── Counts ──────────────────────────────────────────────────────────
    results = {
        "cohort": "cardiac_expanded_42k",
        "n_weeks_total": int(len(cardiac)),
        "date_range": f"{cardiac['week_start_dt'].min().date()} to {cardiac['week_start_dt'].max().date()}",
        "flu_ground_truth_weeks_in_range": int(cardiac["flu_event_window"].sum()),
        "covid_event_weeks": int(cardiac["covid_event_window"].sum()),
        "lgdi_baseline_mean_2016_2018": round(lgdi_mean, 6),
        "lgdi_baseline_sd_2016_2018": round(lgdi_sd, 6),
        "lgdi_alert_threshold_mean_plus_1_5sd": round(lgdi_threshold, 6),
        "resp_score_baseline_mean_2016_2018": round(resp_mean, 6),
        "resp_score_baseline_sd_2016_2018": round(resp_sd, 6),
        "resp_alert_threshold_mean_plus_1_5sd": round(resp_threshold, 6),
    }

    for flu_name, flu_col in [("S2", "flu_filter_s2"), ("S4", "flu_filter_s4")]:
        for alert_col, alert_label in [
            ("raw_alert_lgdi", "single_stage_LGDI"),
            ("raw_alert_resp", "single_stage_resp_score"),
            (f"alert_two_stage_lgdi_{flu_name}", "two_stage_LGDI_residual"),
            (f"alert_two_stage_resp_{flu_name}", "two_stage_resp_residual"),
        ]:
            if alert_col not in cardiac.columns:
                continue

            # vs FluNet
            flu_true = cardiac["flu_event_window"].astype(bool)
            flu_pred = cardiac[alert_col].astype(bool)
            flu_tp = int((flu_pred & flu_true).sum())
            flu_fp = int((flu_pred & ~flu_true).sum())
            flu_fn = int((~flu_pred & flu_true).sum())
            flu_tn = int((~flu_pred & ~flu_true).sum())
            key_flu = f"{alert_label}_vs_FluNet__flu_filter_{flu_name}"
            results[key_flu] = compute_operating_chars(flu_tp, flu_fp, flu_fn, flu_tn)

            # vs COVID
            covid_true = cardiac["covid_event_window"].astype(bool)
            covid_pred = cardiac[alert_col].astype(bool)
            covid_tp = int((covid_pred & covid_true).sum())
            covid_fp = int((covid_pred & ~covid_true).sum())
            covid_fn = int((~covid_pred & covid_true).sum())
            covid_tn = int((~covid_pred & ~covid_true).sum())
            key_covid = f"{alert_label}_vs_COVID__flu_filter_{flu_name}"
            results[key_covid] = compute_operating_chars(covid_tp, covid_fp, covid_fn, covid_tn)

        # Flu filter self-evaluation
        flu_pred_only = cardiac[flu_col].astype(bool)
        flu_true_only = cardiac["flu_event_window"].astype(bool)
        ftp = int((flu_pred_only & flu_true_only).sum())
        ffp = int((flu_pred_only & ~flu_true_only).sum())
        ffn = int((~flu_pred_only & flu_true_only).sum())
        ftn = int((~flu_pred_only & ~flu_true_only).sum())
        results[f"flu_filter_{flu_name}_self_evaluation"] = compute_operating_chars(ftp, ffp, ffn, ftn)

    # ── Export week-level audit ─────────────────────────────────────────
    audit_cols = [
        "window_anchor", "week_start_dt",
        "n_admissions", "valid",
        "resp_score", "lgdi",
        "covid_event_window", "event_window",
        "flu_event_window", "positivity",
        "flu_filter_s2", "flu_filter_s4",
        "raw_alert_lgdi", "raw_alert_resp",
        "alert_two_stage_lgdi_S2", "alert_two_stage_lgdi_S4",
        "alert_two_stage_resp_S2", "alert_two_stage_resp_S4",
        "n_groups_above",
    ]
    for g in ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]:
        for prefix in ["score_", "z_", "above_"]:
            col = f"{prefix}{g}"
            if col in cardiac.columns:
                audit_cols.append(col)
    audit = cardiac[[c for c in audit_cols if c in cardiac.columns]].copy()
    audit.to_csv(OUT_DIR / "two_stage_cardiac_week_audit.csv", index=False)

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Two-Stage Alert Strategy: Flu Filter → Residual COVID PPV")
    print("=" * 70)

    # Cohort A: WHU primary
    print("\n── Cohort A: WHU Primary 32k ──")
    whu_results = analyse_whu_two_stage()
    with open(OUT_DIR / "two_stage_whu_summary.json", "w", encoding="utf-8") as f:
        json.dump(whu_results, f, indent=2, ensure_ascii=False, default=str)

    # Cohort B: Cardiac expanded
    print("\n── Cohort B: Cardiac Expanded 42k ──")
    cardiac_results = analyse_cardiac_two_stage()
    with open(OUT_DIR / "two_stage_cardiac_summary.json", "w", encoding="utf-8") as f:
        json.dump(cardiac_results, f, indent=2, ensure_ascii=False, default=str)

    # ── Combined report ──
    report_lines = [
        "# Two-Stage Alert Strategy Report",
        "",
        "## Strategy",
        "",
        "**Stage 1 (Flu filter)**: Apply LGDI multi-group consensus rule (≥2 chronic-disease",
        "groups simultaneously above each group's 2016–2018 baseline mean+1.5 SD) plus",
        "season restriction (Nov–Mar) and sustained 2-week requirement (S4 rule) to identify",
        "influenza-like healthcare-utilization residual signals.",
        "",
        "**Stage 2 (Residual COVID check)**: Among remaining non-flu-explained alert weeks,",
        "compute overlap with COVID-coded event windows. PPV = TP / (TP + FP) where",
        "TP = alert firing during COVID event window AND not flu-explained.",
        "",
        "---",
        "",
        "## Cohort A: WHU Primary 32k (2016–2020)",
        "",
        f"- Total weeks: {whu_results['n_weeks_total']}",
        f"- FluNet ground-truth event weeks: {whu_results['flu_ground_truth_weeks']}",
        f"- COVID event weeks (Jan–Apr 2020): {whu_results['covid_event_weeks']}",
        "",
    ]

    # Extract key metrics
    def fmt_oc(d: dict) -> str:
        return f"Se={d['sensitivity']:.3f}  PPV={d['ppv']:.3f}  FAR={d['false_alarm_rate']:.3f}  (TP={d['true_positives']}, FP={d['false_positives']}, FN={d['false_negatives']})"

    for cohort_label, results_dict in [
        ("WHU Primary 32k", whu_results),
        ("Cardiac Expanded 42k", cardiac_results),
    ]:
        report_lines.append(f"## Cohort B: {cohort_label}" if "Cardiac" in cohort_label else "")
        report_lines.append("")
        for key, val in results_dict.items():
            if isinstance(val, dict) and "ppv" in val:
                report_lines.append(f"### {key}")
                report_lines.append(f"```")
                report_lines.append(fmt_oc(val))
                report_lines.append(f"```")
                report_lines.append("")
            elif not isinstance(val, dict):
                if "date_range" in key or "weeks" in key.lower() or "threshold" in key.lower() or "baseline" in key.lower():
                    report_lines.append(f"- **{key}**: {val}")

    report_lines.extend([
        "---",
        "",
        "## Interpretation",
        "",
        "The two-stage strategy asks: after removing weeks that the LGDI framework",
        "identifies as influenza-like (Stage 1), do the remaining alerts capture",
        "COVID-specific signals better than the raw single-stage approach?",
        "",
        "Key caveats:",
        "- WHU COVID window (Jan–Apr 2020) overlaps with normal influenza season;",
        "  the flu filter may inadvertently remove true COVID weeks.",
        "- Cardiac COVID window (Dec 2022–Jan 2023) also falls in NH influenza season.",
        "- FluNet is national aggregate positivity, not patient-level infection status.",
        "- The cardiac flu filter is approximate (uses cardiac-group z-scores mimicking",
        "  the WHU consensus rule, not a separately trained model).",
        "- Both cohorts have very few COVID event weeks (WHU ~17, cardiac 13),",
        "  so PPV estimates are unstable.",
    ])

    with open(OUT_DIR / "two_stage_combined_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Outputs written to {OUT_DIR}")
    print(f"   two_stage_whu_summary.json")
    print(f"   two_stage_whu_week_audit.csv")
    print(f"   two_stage_cardiac_summary.json")
    print(f"   two_stage_cardiac_week_audit.csv")
    print(f"   two_stage_combined_report.md")


if __name__ == "__main__":
    main()
