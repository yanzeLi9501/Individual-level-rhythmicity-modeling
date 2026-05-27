"""Unified-metric experiment: combine LGDI (respiratory-vs-baseline residual)
and per-week respiratory cosine similarity into a single composite indicator,
then compare alert-operating characteristics against FluNet event-week labels
on the WHU primary cohort (32,056 patient record numbers).

Inputs (already on disk):
  NC_revision/lgdi_results/lgdi_whu_rolling4_weekly.csv
  NC_revision/lgdi_results/lgdi_whu_influenza_validation.csv

Per row available there:
  resp_score          = weekly cosine-style respiratory similarity to the 2016-2018 baseline
                        respiratory profile (the original "early cosine similarity" metric)
  mean_other_score    = mean cosine similarity of the five non-respiratory groups
  lgdi                = resp_score - mean_other_score (residual / dominance indicator)
  flu_event_window    = ground-truth label drawn from FluNet 2015/16-2019/20 seasons
  positivity          = WHU-internal influenza positivity rate (proxy outcome)
  positivity_threshold= season-specific high-positivity cutoff

Candidate strategies compared (one threshold rule per metric: baseline mean + 1.5 SD,
where the baseline is calendar years 2016-2018):
  1. Cosine-only       alert when resp_score >= mean(resp_score|2016-2018)+1.5SD
  2. LGDI-only         alert when lgdi       >= mean(lgdi|2016-2018)+1.5SD
  3. Unified-Z-sum     alert when (z_resp + z_lgdi)/2 >= 1.5
  4. Unified-Z-max     alert when max(z_resp, z_lgdi) >= 1.5
  5. Unified-AND       alert when BOTH z_resp >= 1.5 AND z_lgdi >= 1.5
  6. Unified-OR        alert when EITHER z_resp >= 1.5 OR z_lgdi >= 1.5

(z_x is computed with 2016-2018 baseline mean/std; the +1.5 SD level keeps the
threshold rule identical to the published Methods text.)

Outputs:
  NC_revision/unified_metric_results/
    metric_comparison.csv        # rows = strategy, cols = sens/PPV/FAR/AUC/lead-time
    per_week_scores.csv          # per-week z-scores and alerts for every strategy
    summary.json                 # headline numbers + thresholds
    unified_metric_panel.{png,pdf,svg}  # 2-panel figure (timeline + ROC)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BASE_DIR = Path(__file__).resolve().parent
LGDI_DIR = BASE_DIR / "lgdi_results"
OUT_DIR = BASE_DIR / "unified_metric_results"
OUT_DIR.mkdir(exist_ok=True)

VALIDATION_CSV = LGDI_DIR / "lgdi_whu_influenza_validation.csv"

BASELINE_YEARS = (2016, 2018)
ZTHRESH = 1.5

COLORS = {
    "cosine": "#1F77B4",  # blue
    "lgdi": "#D62728",    # red
    "unified": "#2CA02C", # green
    "event": "#FFD580",   # peach
    "alert": "#9467BD",   # purple
    "grid": "#DDDDDD",
}


def load_data() -> pd.DataFrame:
    if not VALIDATION_CSV.exists():
        raise SystemExit(f"Missing {VALIDATION_CSV}")
    df = pd.read_csv(VALIDATION_CSV)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["year"] = df["week_start"].dt.year
    df["lgdi"] = pd.to_numeric(df["lgdi"], errors="coerce")
    df["resp_score"] = pd.to_numeric(df["resp_score"], errors="coerce")
    df["flu_event_window"] = df["flu_event_window"].astype(bool)
    df["valid"] = df.get("valid", True).astype(bool) if "valid" in df.columns else True
    return df.dropna(subset=["lgdi", "resp_score"])


def baseline_z(series: pd.Series, base_mask: pd.Series) -> tuple[pd.Series, float, float]:
    base = series[base_mask].dropna()
    mu = float(base.mean())
    sd = float(base.std())
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (series - mu) / sd, mu, sd


def confusion(alert: np.ndarray, event: np.ndarray) -> dict[str, float]:
    a = alert.astype(bool); e = event.astype(bool)
    tp = int((a & e).sum()); fp = int((a & ~e).sum())
    fn = int((~a & e).sum()); tn = int((~a & ~e).sum())
    sens = tp / (tp + fn) if (tp + fn) else math.nan
    ppv = tp / (tp + fp) if (tp + fp) else math.nan
    far = fp / (fp + tn) if (fp + tn) else math.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alerts": int(a.sum()), "events": int(e.sum()),
            "sensitivity": sens, "ppv": ppv, "false_alarm_rate": far}


def lead_time(week_start: pd.Series, alert: np.ndarray, event: np.ndarray) -> dict[str, object]:
    e_mask = event.astype(bool); a_mask = alert.astype(bool)
    if not e_mask.any() or not a_mask.any():
        return {"first_alert": None, "first_event": None, "lead_weeks": None}
    first_event = pd.Timestamp(week_start[e_mask].min())
    pre = (week_start < first_event) & a_mask
    if not pre.any():
        return {"first_alert": None, "first_event": first_event.date().isoformat(), "lead_weeks": 0}
    first_alert = pd.Timestamp(week_start[pre].min())
    lead = int((first_event - first_alert).days // 7)
    return {"first_alert": first_alert.date().isoformat(),
            "first_event": first_event.date().isoformat(),
            "lead_weeks": lead}


def evaluate(df: pd.DataFrame, score: np.ndarray, alert: np.ndarray, name: str) -> dict[str, object]:
    row = {"strategy": name, **confusion(alert, df["flu_event_window"].values)}
    try:
        row["auc"] = float(roc_auc_score(df["flu_event_window"].values.astype(int), score))
    except ValueError:
        row["auc"] = math.nan
    row.update(lead_time(df["week_start"], alert, df["flu_event_window"].values))
    return row


def main() -> None:
    df = load_data().reset_index(drop=True)
    base_mask = (df["year"] >= BASELINE_YEARS[0]) & (df["year"] <= BASELINE_YEARS[1])
    print(f"Loaded {len(df)} weekly rows; baseline weeks={int(base_mask.sum())}, "
          f"event weeks={int(df['flu_event_window'].sum())}")

    z_cosine, mu_c, sd_c = baseline_z(df["resp_score"], base_mask)
    z_lgdi, mu_l, sd_l = baseline_z(df["lgdi"], base_mask)
    unified_avg = (z_cosine + z_lgdi) / 2.0
    unified_max = np.maximum(z_cosine, z_lgdi)
    print(f"  baseline cosine  mean={mu_c:.4f} sd={sd_c:.4f}")
    print(f"  baseline lgdi    mean={mu_l:.4f} sd={sd_l:.4f}")

    strategies: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("cosine_only",          z_cosine.values,                z_cosine.values >= ZTHRESH),
        ("lgdi_only",            z_lgdi.values,                  z_lgdi.values   >= ZTHRESH),
        ("unified_zsum",         unified_avg.values,             unified_avg.values >= ZTHRESH),
        ("unified_zmax",         unified_max,                    unified_max     >= ZTHRESH),
        ("unified_and",          np.minimum(z_cosine, z_lgdi),   (z_cosine >= ZTHRESH).values & (z_lgdi >= ZTHRESH).values),
        ("unified_or",           unified_max,                    (z_cosine >= ZTHRESH).values | (z_lgdi >= ZTHRESH).values),
    ]
    rows: list[dict[str, object]] = []
    for name, score, alert in strategies:
        rows.append(evaluate(df, score, alert, name))
    perf = pd.DataFrame(rows)
    perf.to_csv(OUT_DIR / "metric_comparison.csv", index=False)
    print("\nPerformance comparison:")
    print(perf[["strategy", "alerts", "events", "tp", "fp", "fn", "tn",
                "sensitivity", "ppv", "false_alarm_rate", "auc", "lead_weeks"]].to_string(index=False))

    # per-week dump
    weekly = df[["week_start", "label", "valid", "n_admissions", "resp_score", "lgdi",
                 "flu_event_window", "season_id"]].copy()
    weekly["z_cosine"] = z_cosine.values
    weekly["z_lgdi"] = z_lgdi.values
    weekly["unified_zsum"] = unified_avg.values
    weekly["unified_zmax"] = unified_max
    for name, _, alert in strategies:
        weekly[f"alert_{name}"] = alert.astype(int)
    weekly.to_csv(OUT_DIR / "per_week_scores.csv", index=False)

    # summary
    summary = {
        "n_weeks": int(len(df)),
        "baseline_window": {"start_year": BASELINE_YEARS[0], "end_year": BASELINE_YEARS[1], "weeks": int(base_mask.sum())},
        "event_weeks": int(df["flu_event_window"].sum()),
        "baseline_stats": {"cosine_mean": mu_c, "cosine_sd": sd_c, "lgdi_mean": mu_l, "lgdi_sd": sd_l},
        "z_threshold": ZTHRESH,
        "performance": rows,
        "scripts_used": {"validation_csv": str(VALIDATION_CSV.relative_to(BASE_DIR))},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Figure: 2 panels (timeline + ROC) ──
    fig = plt.figure(figsize=(12.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.28, left=0.06, right=0.98, top=0.92, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    ax.plot(df["week_start"], z_cosine, color=COLORS["cosine"], lw=1.1, label="z(Pearson profile-corr.)")
    ax.plot(df["week_start"], z_lgdi, color=COLORS["lgdi"], lw=1.1, label="z(LGDI)")
    ax.plot(df["week_start"], unified_avg, color=COLORS["unified"], lw=1.5, label="z(unified average)")
    ax.axhline(ZTHRESH, color="#888", ls="--", lw=0.8, label=f"z = {ZTHRESH}")
    # shade event windows
    label_added = False
    for _, r in df[df["flu_event_window"]].iterrows():
        ax.axvspan(r["week_start"], r["week_start"] + pd.Timedelta(days=7),
                   color=COLORS["event"], alpha=0.35, lw=0, label="FluNet event week" if not label_added else None)
        label_added = True
    ax.set_ylabel("Standardised score (baseline-z)")
    ax.set_title("A. Weekly Pearson profile-corr., LGDI and unified score on WHU primary cohort (n = 32 056)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, framealpha=0.92, loc="upper left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5, alpha=0.6)

    ax_r = fig.add_subplot(gs[1])
    y_true = df["flu_event_window"].values.astype(int)
    for label, score, color in [
        ("cosine-only",  z_cosine.values,        COLORS["cosine"]),
        ("LGDI-only",    z_lgdi.values,          COLORS["lgdi"]),
        ("unified avg",  unified_avg.values,     COLORS["unified"]),
    ]:
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        ax_r.plot(fpr, tpr, lw=1.4, color=color, label=f"{label} (AUC = {auc:.3f})")
    ax_r.plot([0, 1], [0, 1], color="#888", lw=0.7, ls="--")
    ax_r.set_xlabel("False-positive rate")
    ax_r.set_ylabel("Sensitivity")
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1.02); ax_r.set_aspect("equal")
    ax_r.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax_r.set_title("B. ROC curves (FluNet event-week ground truth)", fontsize=10.5, fontweight="bold")
    ax_r.grid(color=COLORS["grid"], lw=0.5, alpha=0.6)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"unified_metric_panel.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote panel: {OUT_DIR / 'unified_metric_panel.png'}")


if __name__ == "__main__":
    main()
