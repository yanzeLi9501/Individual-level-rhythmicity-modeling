#!/usr/bin/env python3
"""Generate Figure 3: Influenza season validation of the LGDI framework.

Two panels:
  A — Strategy progression (4-step optimisation ladder, 3 metrics as connected
      dot-line chart):  S1 = Pearson reference; S2 = +Consensus ≥2 groups;
      S3 = +Season Oct–Apr; S4 = +Season Nov–Mar + Sustained 2-wk.
  B — November 2019 anomaly timeline: weekly respiratory z-score across all
      NH-season (Nov–Mar) weeks, with FluNet background shading, alert markers
      for the optimised strategy, and special annotation for the 4 pre-COVID
      false-positive weeks in Nov 2019.

Outputs written to NC_revision/:
  Figure3_influenza_validation.png / .pdf / .svg / .tif
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "lgdi_results"
VALIDATION_CSV = RESULTS / "lgdi_whu_influenza_validation.csv"
SUMMARY_JSON = RESULTS / "lgdi_whu_influenza_summary.json"
UNIFIED_DIR = BASE / "unified_metric_results"
PER_WEEK_CSV = UNIFIED_DIR / "per_week_scores.csv"
METRIC_COMPARE_CSV = UNIFIED_DIR / "metric_comparison.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2c7bb6",   # blue  (Sensitivity)
    "orange":    "#d7191c",   # red → used for FAR (alarm)
    "ppv":       "#fdae61",   # orange (PPV)
    "sens":      "#2c7bb6",   # blue (Sensitivity)
    "far":       "#a6a6a6",   # gray (FAR dashed)
    "alert_fp":  "#d73027",   # red alert bars
    "alert_tp":  "#4dac26",   # green alert bars
    "flu_pos":   "#a1d99b",   # light green FluNet+ background
    "flu_neg":   "#e8e8e8",   # light gray non-event season background
    "nov19":     "#fc8d59",   # annotation box Nov 2019
}

DPI = 300
FIG_FORMATS = ["png", "pdf", "svg", "tif"]


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")


def save_figure(fig: plt.Figure, stem: str) -> None:
    # R1 compliance: do NOT clear panel-level ax.set_title(); only suppress the
    # figure-level suptitle so it is not embedded in the saved image.
    fig.suptitle("")
    for fmt in FIG_FORMATS:
        path = BASE / f"{stem}.{fmt}"
        kwargs = {"dpi": DPI, "bbox_inches": "tight"}
        if fmt == "tif":
            kwargs["format"] = "tiff"
        fig.savefig(path, **kwargs)
        print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel C: ROC curves (Pearson vs LGDI vs Unified)
# ─────────────────────────────────────────────────────────────────────────────
def build_panel_c(ax: plt.Axes) -> None:
    """ROC curves for Pearson profile correlation, LGDI scalar, unified avg."""
    from sklearn.metrics import roc_auc_score, roc_curve

    pw = pd.read_csv(PER_WEEK_CSV)
    y_true = pw["flu_event_window"].astype(int).values
    metrics = [
        ("Pearson Profile Corr.", "z_cosine",    "#2c7bb6"),
        ("LGDI scalar",          "z_lgdi",      "#d7191c"),
        ("Unified avg",          "unified_zsum", "#1a9641"),
    ]
    for label, col, color in metrics:
        score = pw[col].values.astype(float)
        mask = np.isfinite(score)
        fpr, tpr, _ = roc_curve(y_true[mask], score[mask])
        auc = roc_auc_score(y_true[mask], score[mask])
        ax.plot(fpr, tpr, lw=1.6, color=color, label=f"{label}\n(AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--", label="Chance (0.5)")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (Sensitivity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_title(
        "Receiver-operating characteristic\n"
        "(WHU-32k vs FluNet, 213 monitor weeks)",
        fontsize=9,
    )
    ax.grid(alpha=0.22)


# ─────────────────────────────────────────────────────────────────────────────
# Panel D: AUC bar comparison across 3 core strategies
# ─────────────────────────────────────────────────────────────────────────────
def build_panel_d(ax: plt.Axes) -> None:
    """Bar chart of ROC-AUC for Pearson, LGDI, Unified-avg strategies."""
    mc = pd.read_csv(METRIC_COMPARE_CSV)
    keep = ["cosine_only", "lgdi_only", "unified_zsum"]
    labels = ["Pearson Profile Corr.", "LGDI scalar", "Unified avg"]
    colors_bar = ["#2c7bb6", "#d7191c", "#1a9641"]
    mc_sub = mc[mc["strategy"].isin(keep)].set_index("strategy").reindex(keep)
    aucs = mc_sub["auc"].values
    x = np.arange(len(keep))
    bars = ax.bar(x, aucs, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#888888", lw=0.9, ls="--", label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0.38, 0.70)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "ROC-AUC by metric type\n"
        "(Pearson outperforms scalar index)",
        fontsize=9,
    )
    for i, v in enumerate(aucs):
        if np.isfinite(v):
            ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8,
                    fontweight="bold", color=colors_bar[i])
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Strategy progression
# ─────────────────────────────────────────────────────────────────────────────
def build_panel_a(ax: plt.Axes, summary: dict) -> None:
    """Connected dot-line chart for 4 strategies × 3 metrics."""
    strat_map = {r["strategy"]: r for r in summary["per_strategy_metrics"]}

    # The 4-step ladder keys, in order
    step_keys = [
        "REFERENCE_resp_mean_plus_1_5sd",       # S1
        "consensus_2groups_mean_plus_1_5sd",    # S2
        "consensus_2groups_season_oct_apr",     # S3
        "season_sustained_consensus2grp_nov_mar",  # S4
    ]
    step_labels = [
        "S1\nPearson ≥θ",
        "S2\n+Consensus\n≥2 groups",
        "S3\n+Season\nOct–Apr",
        "S4\n+Season Nov–Mar\n+Sustained 2-wk",
    ]
    x = np.arange(len(step_keys))
    sens_vals = [strat_map.get(k, {}).get("sensitivity", np.nan) for k in step_keys]
    ppv_vals  = [strat_map.get(k, {}).get("ppv", np.nan) for k in step_keys]
    far_vals  = [strat_map.get(k, {}).get("false_alarm_rate", np.nan) for k in step_keys]

    # Plot lines + markers
    lw = 1.8
    ms = 8
    ax.plot(x, sens_vals, "o-", color=COLORS["sens"], linewidth=lw, markersize=ms,
            label="Sensitivity", zorder=3)
    ax.plot(x, ppv_vals,  "s-", color=COLORS["ppv"],  linewidth=lw, markersize=ms,
            label="PPV", zorder=3)
    ax.plot(x, far_vals,  "^--", color=COLORS["far"], linewidth=lw, markersize=ms,
            label="FAR", zorder=3)

    # Value annotations (offset alternately to avoid overlap)
    offsets_sens = [0.04, 0.04, 0.04, -0.06]
    offsets_ppv  = [-0.06, -0.06, -0.06, 0.04]
    offsets_far  = [0.04, 0.04, -0.06, 0.04]
    for i, v in enumerate(sens_vals):
        if np.isfinite(v):
            ax.text(x[i], v + offsets_sens[i], f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS["sens"])
    for i, v in enumerate(ppv_vals):
        if np.isfinite(v):
            ax.text(x[i], v + offsets_ppv[i], f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS["ppv"])
    for i, v in enumerate(far_vals):
        if np.isfinite(v):
            ax.text(x[i], v + offsets_far[i], f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS["far"])

    # S4 special annotation
    ppv_s4 = ppv_vals[-1]
    if np.isfinite(ppv_s4):
        ax.annotate(
            f"PPV = {ppv_s4:.3f}\n5 FP remaining",
            xy=(x[-1], ppv_s4), xytext=(x[-1] - 0.55, ppv_s4 + 0.13),
            fontsize=7.5, color=COLORS["ppv"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["ppv"], lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["ppv"], alpha=0.85),
        )

    # PPV = 0.80 reference line
    ax.axhline(0.80, color=COLORS["ppv"], lw=0.9, ls=":", alpha=0.65, zorder=1)
    ax.text(3.08, 0.80, "0.80", va="center", ha="left", fontsize=7, color=COLORS["ppv"], alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Rate")
    ax.set_title("Detection strategy optimisation\n(WHU-32k vs FluNet, 51 influenza-event weeks)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.4, 3.4)

    # Add sensitivity trade-off annotation (S2→S4 drop)
    ax.annotate(
        "Sensitivity trade-off:\n0.549→0.412 (↓25%)\n(sustained rule eliminates\nsingle-week spikes)",
        xy=(x[1], sens_vals[1]), xytext=(x[2] + 0.05, 0.60),
        fontsize=7, color=COLORS["sens"], alpha=0.85,
        arrowprops=dict(arrowstyle="-", color=COLORS["sens"], lw=0.7, linestyle="dashed"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel B: Nov 2019 timeline
# ─────────────────────────────────────────────────────────────────────────────
def build_panel_b(ax: plt.Axes) -> None:
    """Respiratory z-score for NH-season weeks; alert markers; Nov 2019 box."""
    df = pd.read_csv(VALIDATION_CSV, parse_dates=["week_start"])

    # Filter to NH-season weeks (Nov–Mar) to create the compact timeline
    season_df = df[df["in_nov_mar"] == 1].sort_values("week_start").reset_index(drop=True)

    # Use sequential integer x-axis positions
    xs = np.arange(len(season_df))
    z = season_df["z_resp_score"].values
    event = season_df["flu_event_window"].astype(bool).values
    alert_s4 = season_df["alert_s4"].values
    weeks = season_df["week_start"].values  # numpy datetime64

    # ── Background shading: alternate by season ──
    # Identify season transitions (new season year)
    season_ids = season_df["season_id"].values
    prev_sid = None
    season_start_idx = []
    for i, sid in enumerate(season_ids):
        if sid != prev_sid:
            season_start_idx.append((i, sid))
            prev_sid = sid

    for k, (start_i, sid) in enumerate(season_start_idx):
        end_i = season_start_idx[k + 1][0] if k + 1 < len(season_start_idx) else len(xs)
        fc = "#f7fbff" if k % 2 == 0 else "#fff5f0"
        ax.axvspan(start_i - 0.5, end_i - 0.5, facecolor=fc, alpha=0.45, zorder=0)
        # Season label at top
        mid = (start_i + end_i) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 6.5,
                sid, ha="center", va="bottom", fontsize=7, color="#666666",
                transform=ax.transData, clip_on=True)

    # ── FluNet positive background (green) ──
    for i, is_event in enumerate(event):
        if is_event:
            ax.axvspan(i - 0.5, i + 0.5, facecolor=COLORS["flu_pos"], alpha=0.55, zorder=1)

    # ── z-score bars ──
    bar_colors = [COLORS["alert_fp"] if (alert_s4[i] and not event[i])
                  else COLORS["alert_tp"] if (alert_s4[i] and event[i])
                  else "#a0aec0"
                  for i in range(len(xs))]
    ax.bar(xs, z, color=bar_colors, width=0.85, edgecolor="none", zorder=2, alpha=0.80)

    # ── Alert border for S4-triggered weeks ──
    for i in range(len(xs)):
        if alert_s4[i]:
            color = COLORS["alert_fp"] if not event[i] else COLORS["alert_tp"]
            ax.axvline(i, color=color, lw=1.2, alpha=0.7, zorder=3, linestyle="-")

    # ── Nov 2019 FP box annotation ──
    # FP weeks: 2019-11-11, 2019-11-18, 2019-11-25, 2019-12-02
    nov19_weeks = pd.to_datetime(["2019-11-11", "2019-11-18", "2019-11-25", "2019-12-02"])
    nov19_idx = [i for i, w in enumerate(weeks)
                 if pd.Timestamp(w).normalize() in nov19_weeks.normalize()]
    if nov19_idx:
        x_lo = min(nov19_idx) - 0.6
        x_hi = max(nov19_idx) + 0.6
        y_lo = -0.5
        y_hi = max(z[nov19_idx]) + 0.6
        rect = mpatches.FancyBboxPatch(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            boxstyle="round,pad=0.2", linewidth=1.5,
            edgecolor=COLORS["nov19"], facecolor="none", linestyle="--", zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            (x_lo + x_hi) / 2, y_hi + 0.35,
            "4 weeks: FluNet-negative\n(pre-COVID anomaly)",
            ha="center", va="bottom", fontsize=7.5,
            color=COLORS["nov19"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=COLORS["nov19"], alpha=0.90, linewidth=1.2),
            zorder=6,
        )

    # ── Dec 31 2019 WHO notification arrow ──
    who_date = pd.Timestamp("2019-12-31")
    who_idx_list = [i for i, w in enumerate(weeks)
                    if abs((pd.Timestamp(w) - who_date).days) <= 10]
    if who_idx_list:
        who_x = who_idx_list[0]
        ax.annotate(
            "WHO COVID-19\nnotification\nDec 31, 2019",
            xy=(who_x, z[who_x] + 0.4),
            xytext=(who_x + 2.8, z[who_x] + 2.5),
            fontsize=7, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888888", alpha=0.85),
            zorder=6,
        )

    # ── X-axis: show season separators and approximate month labels ──
    # Place major ticks at season boundaries
    tick_pos = []
    tick_lbl = []
    for start_i, sid in season_start_idx:
        tick_pos.append(start_i)
        tick_lbl.append(f"Nov\n{sid[:4]}")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=8)
    for tp in tick_pos[1:]:
        ax.axvline(tp - 0.5, color="#aaaaaa", lw=0.7, ls=":", zorder=0)

    ax.set_xlim(-0.7, len(xs) - 0.3)
    ax.set_ylim(-1.0, 10.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(1.5, color="#888888", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_ylabel("Respiratory profile z-score\n(resp_score, 4-wk rolling)")
    ax.set_xlabel("Nov–Mar season weeks (2016–2020)")
    ax.set_title(
        "WHU-32k alert timeline (NH core season weeks)\n"
        "Optimised strategy: Season Nov–Mar + Sustained 2-wk + Consensus ≥2 groups",
        fontsize=9,
    )

    # Legend
    tp_patch  = mpatches.Patch(color=COLORS["alert_tp"], label="Alert = TP (FluNet+)")
    fp_patch  = mpatches.Patch(color=COLORS["alert_fp"], label="Alert = FP (FluNet−)")
    flu_patch = mpatches.Patch(color=COLORS["flu_pos"],  alpha=0.55, label="FluNet+ weeks")
    ax.legend(handles=[tp_patch, fp_patch, flu_patch], frameon=False, fontsize=7.5,
              loc="upper left")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(
        2, 2, wspace=0.42, hspace=0.50,
        left=0.06, right=0.97, top=0.92, bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    build_panel_a(ax_a, summary)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    build_panel_b(ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    build_panel_c(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "D")
    build_panel_d(ax_d)

    print("[generate_fig3] Saving...")
    save_figure(fig, "Figure3_influenza_validation")
    plt.close(fig)
    print("[generate_fig3] Done.")


if __name__ == "__main__":
    main()
