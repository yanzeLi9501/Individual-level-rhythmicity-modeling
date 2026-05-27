#!/usr/bin/env python3
"""LGDI optimization strategies (Task 2).

Operates on existing LGDI weekly outputs in `lgdi_results/` to evaluate alert
strategies that go beyond the prespecified `mean+1.5SD` rolling-4-week
threshold. Produces `lgdi_optimized_performance.csv` and an updated
`lgdi_optimized_summary.json` with five families:

  1. multi_scale_2w_or_4w  — dual-window OR rule (4-week LGDI || 2-week LGDI)
  2. cusum                 — one-sided CUSUM on respiratory group_score residual
  3. ewma                  — EWMA control chart on LGDI
  4. seasonal_adjusted     — month-of-year baseline threshold (mean+1.5SD per ISO month)
  5. joint_los_gap         — composite trigger requiring both LOS and gap residual elevation

The cardiac-cohort Dec 2022-Jan 2023 event window is reused as ground truth.
Outputs are also synced to `resubmission_package_*/analysis_outputs/`.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
LGDI_DIR = BASE / "lgdi_results"
PACKAGE_ANALYSIS = BASE / "resubmission_package_20260512" / "analysis_outputs"

EVENT_START = pd.Timestamp("2022-12-01")
EVENT_END = pd.Timestamp("2023-01-31")
LEAD_START = EVENT_START - pd.Timedelta(days=28)
MONITOR_START = pd.Timestamp("2019-01-01")
MONITOR_END = pd.Timestamp("2024-12-31")
BASELINE_START = pd.Timestamp("2016-01-01")
BASELINE_END = pd.Timestamp("2018-12-31")

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
RESP = "Respiratory"


# ---------- helpers ----------

def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else math.nan


def first_alert_lead(monitor: pd.DataFrame, alert_col: str) -> tuple[str | None, int | None]:
    sub = monitor[monitor[alert_col] & monitor["window_anchor_dt"].between(LEAD_START, EVENT_END)]
    if sub.empty:
        return None, None
    first = pd.Timestamp(sub["window_anchor_dt"].min())
    return first.date().isoformat(), int((EVENT_START - first).days)


def score_alerts(monitor: pd.DataFrame, alert_col: str, rule: str, threshold_value: float | None) -> dict:
    event = monitor["event_window"].astype(bool)
    alert = monitor[alert_col].astype(bool)
    tp = int((alert & event).sum())
    fp = int((alert & ~event).sum())
    fn = int((~alert & event).sum())
    tn = int((~alert & ~event).sum())
    first, lead = first_alert_lead(monitor, alert_col)
    return {
        "threshold_rule": rule,
        "threshold_value": threshold_value,
        "monitor_windows": int(len(monitor)),
        "event_windows": int(event.sum()),
        "alert_windows": int(alert.sum()),
        "true_positive_event_windows": tp,
        "false_positive_event_strict": fp,
        "false_negative_event_windows": fn,
        "true_negative_event_strict": tn,
        "sensitivity_event_week": safe_div(tp, tp + fn),
        "precision_ppv_event_strict": safe_div(tp, tp + fp),
        "false_alarm_rate_event_strict": safe_div(fp, fp + tn),
        "first_alert_in_lead_or_event": first,
        "lead_time_days": lead,
    }


# ---------- strategy implementations ----------

def add_event_flags(timeline: pd.DataFrame) -> pd.DataFrame:
    df = timeline.copy()
    df["window_anchor_dt"] = pd.to_datetime(df["window_anchor"])
    df["event_window"] = (df["window_anchor_dt"] >= EVENT_START) & (df["window_anchor_dt"] <= EVENT_END)
    return df


def strategy_multi_scale(timeline_4w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """4-week LGDI series + 2-week derived series (rolling mean width=2 of weekly LGDI)."""
    df = add_event_flags(timeline_4w)
    df = df[df["valid"].eq(True)].sort_values("window_anchor_dt").copy()
    # Approximate "2-week" series as 2-week rolling mean of the per-window LGDI values
    df["lgdi_2w"] = df["lgdi"].rolling(window=2, min_periods=1).mean()
    df["lgdi_4w"] = df["lgdi"]
    base_mask = df["window_anchor_dt"].between(BASELINE_START, BASELINE_END)
    base_4w = df.loc[base_mask, "lgdi_4w"].dropna()
    base_2w = df.loc[base_mask, "lgdi_2w"].dropna()
    th_4w = float(base_4w.mean() + 1.5 * base_4w.std())
    th_2w = float(base_2w.mean() + 1.5 * base_2w.std())
    monitor = df[df["window_anchor_dt"].between(MONITOR_START, MONITOR_END)].copy()
    monitor["alert"] = (monitor["lgdi_4w"] >= th_4w) | (monitor["lgdi_2w"] >= th_2w)
    perf = score_alerts(monitor, "alert", "multi_scale_2w_or_4w", th_4w)
    perf["secondary_threshold_2w"] = th_2w
    return monitor[["window_anchor_dt", "lgdi_4w", "lgdi_2w", "alert", "event_window"]], perf


def strategy_cusum(timeline_4w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One-sided CUSUM with k = 0.5 sigma, h = 4 sigma on baseline residuals."""
    df = add_event_flags(timeline_4w)
    df = df[df["valid"].eq(True)].sort_values("window_anchor_dt").copy()
    base_mask = df["window_anchor_dt"].between(BASELINE_START, BASELINE_END)
    base = df.loc[base_mask, "lgdi"].dropna()
    mu = float(base.mean())
    sd = float(base.std())
    if sd <= 1e-9:
        sd = 1.0
    k = 0.5 * sd
    h = 4.0 * sd
    cusum = 0.0
    cusum_series: list[float] = []
    for value in df["lgdi"].fillna(mu).values:
        cusum = max(0.0, cusum + (float(value) - mu) - k)
        cusum_series.append(cusum)
    df["cusum"] = cusum_series
    monitor = df[df["window_anchor_dt"].between(MONITOR_START, MONITOR_END)].copy()
    monitor["alert"] = monitor["cusum"] >= h
    perf = score_alerts(monitor, "alert", "cusum_k0_5sd_h4sd", h)
    perf["cusum_mu"] = mu
    perf["cusum_sigma"] = sd
    return monitor[["window_anchor_dt", "lgdi", "cusum", "alert", "event_window"]], perf


def strategy_ewma(timeline_4w: pd.DataFrame, lam: float = 0.2, L: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """One-sided EWMA control chart on LGDI."""
    df = add_event_flags(timeline_4w)
    df = df[df["valid"].eq(True)].sort_values("window_anchor_dt").copy()
    base_mask = df["window_anchor_dt"].between(BASELINE_START, BASELINE_END)
    base = df.loc[base_mask, "lgdi"].dropna()
    mu = float(base.mean())
    sd = float(base.std())
    if sd <= 1e-9:
        sd = 1.0
    z = mu
    z_series: list[float] = []
    for value in df["lgdi"].fillna(mu).values:
        z = lam * float(value) + (1 - lam) * z
        z_series.append(z)
    df["ewma"] = z_series
    sigma_z = sd * math.sqrt(lam / (2 - lam))
    threshold = mu + L * sigma_z
    monitor = df[df["window_anchor_dt"].between(MONITOR_START, MONITOR_END)].copy()
    monitor["alert"] = monitor["ewma"] >= threshold
    perf = score_alerts(monitor, "alert", f"ewma_lambda{lam}_L{L}", threshold)
    return monitor[["window_anchor_dt", "lgdi", "ewma", "alert", "event_window"]], perf


def strategy_seasonal(timeline_4w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-ISO-month baseline threshold (mean + 1.5SD by month-of-year)."""
    df = add_event_flags(timeline_4w)
    df = df[df["valid"].eq(True)].sort_values("window_anchor_dt").copy()
    df["month"] = df["window_anchor_dt"].dt.month
    base_mask = df["window_anchor_dt"].between(BASELINE_START, BASELINE_END)
    base = df.loc[base_mask].dropna(subset=["lgdi"])
    month_thr: dict[int, float] = {}
    for m in range(1, 13):
        s = base[base["month"].eq(m)]["lgdi"]
        if len(s) >= 3:
            month_thr[m] = float(s.mean() + 1.5 * s.std())
        else:
            month_thr[m] = float(base["lgdi"].mean() + 1.5 * base["lgdi"].std())
    df["threshold_month"] = df["month"].map(month_thr)
    monitor = df[df["window_anchor_dt"].between(MONITOR_START, MONITOR_END)].copy()
    monitor["alert"] = monitor["lgdi"] >= monitor["threshold_month"]
    perf = score_alerts(monitor, "alert", "seasonal_month_mean_plus_1_5sd", float(np.mean(list(month_thr.values()))))
    perf["per_month_thresholds"] = month_thr
    return monitor[["window_anchor_dt", "lgdi", "threshold_month", "alert", "event_window"]], perf


def strategy_joint_los_gap(group_metrics: pd.DataFrame, timeline_4w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Composite trigger: respiratory next_los_days residual high AND next_gap_days residual high.

    Both signed in the LGDI direction (positive = consistent with extra burden).
    """
    gm = group_metrics[group_metrics["group"].eq(RESP)].copy()
    gm["window_anchor_dt"] = pd.to_datetime(gm["window_anchor"])
    pivot = gm.pivot_table(
        index=["label", "window_anchor_dt"],
        columns="outcome",
        values="signed_mase_residual",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    base_mask = pivot["window_anchor_dt"].between(BASELINE_START, BASELINE_END)
    los_base = pivot.loc[base_mask, "next_los_days"].dropna()
    gap_base = pivot.loc[base_mask, "next_gap_days"].dropna()
    th_los = float(los_base.mean() + 1.5 * los_base.std()) if len(los_base) else 0.0
    th_gap = float(gap_base.mean() + 1.5 * gap_base.std()) if len(gap_base) else 0.0
    pivot["alert"] = (pivot["next_los_days"] >= th_los) & (pivot["next_gap_days"] >= th_gap)
    pivot["event_window"] = (pivot["window_anchor_dt"] >= EVENT_START) & (pivot["window_anchor_dt"] <= EVENT_END)
    monitor = pivot[pivot["window_anchor_dt"].between(MONITOR_START, MONITOR_END)].copy()
    perf = score_alerts(monitor, "alert", "joint_los_gap_resp_mean_plus_1_5sd", th_los)
    perf["secondary_threshold_gap"] = th_gap
    return monitor[["window_anchor_dt", "next_los_days", "next_gap_days", "alert", "event_window"]], perf


# ---------- driver ----------

def main() -> None:
    timeline_4w = pd.read_csv(LGDI_DIR / "lgdi_rolling4_weekly.csv")
    group_metrics = pd.read_csv(LGDI_DIR / "lgdi_group_window_metrics.csv")

    rows: list[dict] = []
    series_dump: dict[str, list[dict]] = {}
    for fn in (strategy_multi_scale, strategy_cusum, strategy_ewma, strategy_seasonal):
        ser, perf = fn(timeline_4w)
        rows.append(perf)
        series_dump[perf["threshold_rule"]] = ser.assign(window_anchor_dt=ser["window_anchor_dt"].astype(str)).to_dict(orient="records")
    ser, perf = strategy_joint_los_gap(group_metrics, timeline_4w)
    rows.append(perf)
    series_dump[perf["threshold_rule"]] = ser.assign(window_anchor_dt=ser["window_anchor_dt"].astype(str)).to_dict(orient="records")

    # Reference: prior `mean+1.5SD` from the existing performance file for direct comparison
    ref = pd.read_csv(LGDI_DIR / "lgdi_performance.csv")
    ref_row = ref[ref["threshold_rule"].eq("lgdi_mean_plus_1_5sd")].iloc[0].to_dict()
    rows.insert(0, {
        "threshold_rule": "REFERENCE_lgdi_mean_plus_1_5sd",
        "threshold_value": float(ref_row["threshold_value"]),
        "monitor_windows": int(ref_row["monitor_windows"]),
        "event_windows": int(ref_row["event_windows"]),
        "alert_windows": int(ref_row["alert_windows"]),
        "true_positive_event_windows": int(ref_row["true_positive_event_windows"]),
        "false_positive_event_strict": int(ref_row["false_positive_event_strict"]),
        "false_negative_event_windows": int(ref_row["false_negative_event_windows"]),
        "true_negative_event_strict": int(ref_row["true_negative_event_strict"]),
        "sensitivity_event_week": float(ref_row["sensitivity_event_week"]),
        "precision_ppv_event_strict": float(ref_row["precision_ppv_event_strict"]),
        "false_alarm_rate_event_strict": float(ref_row["false_alarm_rate_event_strict"]),
        "first_alert_in_lead_or_event": ref_row["first_alert_in_lead_or_event"],
        "lead_time_days": ref_row["lead_time_days"],
    })

    out_df = pd.DataFrame(rows)
    out_csv = LGDI_DIR / "lgdi_optimized_performance.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(out_df[["threshold_rule", "sensitivity_event_week", "precision_ppv_event_strict", "false_alarm_rate_event_strict", "first_alert_in_lead_or_event", "lead_time_days"]].to_string(index=False))

    summary = {
        "scope": "Cardiac expanded cohort, 2019-01-01 to 2024-12-31 monitoring window",
        "event_definition": {"start": EVENT_START.date().isoformat(), "end": EVENT_END.date().isoformat(),
                              "lead_start": LEAD_START.date().isoformat()},
        "baseline_window": {"start": BASELINE_START.date().isoformat(), "end": BASELINE_END.date().isoformat()},
        "strategies": rows,
    }
    out_json = LGDI_DIR / "lgdi_optimized_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=lambda o: None if pd.isna(o) else o))
    print(f"Wrote {out_json}")

    PACKAGE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    for src in [out_csv, out_json]:
        shutil.copy2(src, PACKAGE_ANALYSIS / src.name)


if __name__ == "__main__":
    main()
