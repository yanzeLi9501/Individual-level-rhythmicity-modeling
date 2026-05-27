#!/usr/bin/env python3
"""Standalone figure-only regeneration for FigureS10_weekly_rdi_42k.

Reads pre-computed CSVs from weekly_rdi_42k_results/ and the cached
summary JSON to reproduce the 4-panel figure without re-running the full
analysis.  Panel labels A-D placed INSIDE axes at top-left (0.02, 0.97),
fontsize=12, bold — consistent with spring/summer colour scheme used
across all other revised supplementary figures.

Saves to:
  NC_revision/DeepseekRevision/figures/FigureS10_weekly_rdi_42k.{png,pdf,svg,tif}
  NC_revision/DeepseekRevision/output/figures/FigureS10_weekly_rdi_42k.{png,pdf,svg,tif}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent          # NC_revision/
DEEP     = HERE / "DeepseekRevision"
RES_DIR  = HERE / "weekly_rdi_42k_results"
LGDI_DIR = HERE / "lgdi_results"
FIG_CANONICAL = DEEP / "figures"
FIG_OUTPUT    = DEEP / "output" / "figures"

# ── Constants (must match run_weekly_rdi_42k.py) ────────────────────────────
EVENT_START   = pd.Timestamp("2022-12-01")
EVENT_END     = pd.Timestamp("2023-01-31")
LEAD_START    = EVENT_START - pd.Timedelta(days=28)
SOURCE_GAP_START = pd.Timestamp("2019-07-01")
SOURCE_GAP_END   = pd.Timestamp("2020-04-26")
POLICY_LOSS_LABEL = "Policy-induced data loss"

# ── Spring/summer palette consistent with other revised figures ─────────────
COLORS = {
    "primary":   "#5DADE2",   # sky blue  (was #396AB1)
    "secondary": "#F4A261",   # warm orange (was #DA7C30)
    "accent":    "#52B788",   # sage green
    "warn":      "#F4845F",   # coral  (was #CC2529)
    "text":      "#222222",
    "grid":      "#DDDDDD",
}

COVID_CARDIAC_SUMMARY  = HERE / "external_positive_control_results" / "whu_covid_cardiac_summary.json"
CARMEN_CARDIAC_SUMMARY = HERE / "external_positive_control_results" / "carmen_i_cardiac_summary.json"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "svg.fonttype": "none",
})


def _panel_label(ax: plt.Axes, letter: str) -> None:
    """Place bold panel letter inside top-left corner of axes (style matches other figures)."""
    ax.text(0.02, 0.97, letter,
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            va="top", ha="left",
            clip_on=False)


def _add_policy_loss_band(ax: plt.Axes) -> None:
    ax.axvspan(SOURCE_GAP_START, SOURCE_GAP_END,
               color="#7f7f7f", alpha=0.18, linewidth=0,
               label=POLICY_LOSS_LABEL)
    midpoint = SOURCE_GAP_START + (SOURCE_GAP_END - SOURCE_GAP_START) / 2
    ax.text(midpoint, 0.52, POLICY_LOSS_LABEL,
            transform=ax.get_xaxis_transform(),
            rotation=90, ha="center", va="center",
            fontsize=8, color="#4d4d4d", alpha=0.85)


def _plot_with_policy_break(ax: plt.Axes, dates: pd.Series, values: pd.Series, **kw) -> None:
    dates  = pd.to_datetime(dates)
    values = pd.to_numeric(values, errors="coerce")
    for i, seg in enumerate([dates < SOURCE_GAP_START, dates > SOURCE_GAP_END]):
        kw2 = dict(kw)
        if i > 0:
            kw2["label"] = None
        ax.plot(dates[seg], values[seg], **kw2)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_covid_cardiac_mechanism(ax: plt.Axes) -> None:
    whu    = _load_json(COVID_CARDIAC_SUMMARY)
    carmen = _load_json(CARMEN_CARDIAC_SUMMARY)
    if not whu or not carmen:
        ax.text(0.5, 0.5, "COVID cardiac mechanism\nsummary unavailable",
                ha="center", va="center")
        ax.axis("off")
        return

    labels = ["De novo", "Exacerbated\npre-existing", "Pre-existing\nonly", "No cardiac"]
    whu_v  = [float(whu.get("de_novo_covid_cardiac_pct",         np.nan)),
               float(whu.get("exacerbation_existing_pct",         np.nan)),
               float(whu.get("preexisting_only_pct",              np.nan)),
               float(whu.get("no_cardiac_pct",                    np.nan))]
    car_v  = [float(carmen.get("de_novo_covid_cardiac_pct",       np.nan)),
               float(carmen.get("covid_exacerbation_of_preexisting_pct", np.nan)),
               float(carmen.get("preexisting_only_no_acute_pct",  np.nan)),
               float(carmen.get("no_cardiac_pct",                 np.nan))]
    x, w = np.arange(len(labels)), 0.36
    ax.bar(x - w/2, whu_v,  w, color=COLORS["primary"],   label="WHU cardiac COVID patients")
    ax.bar(x + w/2, car_v,  w, color=COLORS["secondary"], label="CARMEN-I benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(80, np.nanmax(whu_v + car_v) + 10))
    ax.set_ylabel("Percent of COVID patients")
    valid = [v for v in whu_v if not np.isnan(v)]
    if valid:
        ax.text(0.10, 0.93,
                f"WHU de novo: {whu_v[0]:.1f}% ({int(whu.get('de_novo_covid_cardiac_n',0))}"
                f"/{int(whu.get('n_unique_covid_patients',0))})",
                transform=ax.transAxes, fontsize=8, color=COLORS["text"])
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)


def build_figure(rolling: pd.DataFrame, performance: pd.DataFrame,
                 threshold: float) -> plt.Figure:
    rolling = rolling.copy()
    # CSV uses 'window_start'; add an alias for plotting
    date_col = "window_start_dt" if "window_start_dt" in rolling.columns else "window_start"
    rolling["window_start_dt"] = pd.to_datetime(rolling[date_col])

    rolling_plot = rolling.copy()
    for col in ["rdi", "resp_sim", "mean_other"]:
        if col in rolling_plot.columns:
            rolling_plot.loc[~rolling_plot.get("valid", pd.Series(True, index=rolling_plot.index)).eq(True), col] = np.nan

    valid_rolling = rolling[rolling.get("valid", pd.Series(True, index=rolling.index)).eq(True)].copy()
    valid_rolling["alert_primary"] = valid_rolling["rdi"] >= threshold
    zoom = valid_rolling[valid_rolling["window_start_dt"].between("2021-01-01", "2024-12-31")]

    fig = plt.figure(figsize=(16, 9.8))
    grid = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34,
                            left=0.07, right=0.98, top=0.97, bottom=0.10)

    # ── Panel A: full-width weekly RDI time series ───────────────────────────
    ax_a = fig.add_subplot(grid[0, :])
    _plot_with_policy_break(ax_a, rolling_plot["window_start_dt"], rolling_plot["rdi"],
                            color=COLORS["primary"], linewidth=1.25,
                            label="4-week rolling weekly RDI")
    _add_policy_loss_band(ax_a)
    ax_a.axhline(threshold, color=COLORS["warn"], linestyle="--", linewidth=1.0,
                 label="Baseline mean + 1.5 SD")
    ax_a.axvspan(EVENT_START, EVENT_END, color=COLORS["secondary"], alpha=0.16,
                 label="Dec 2022–Jan 2023 event window")
    ax_a.scatter(valid_rolling.loc[valid_rolling["alert_primary"], "window_start_dt"],
                 valid_rolling.loc[valid_rolling["alert_primary"], "rdi"],
                 s=18, color=COLORS["warn"], zorder=3, label="Alert weeks")
    ax_a.set_ylabel("Respiratory Dominance Index")
    ax_a.legend(fontsize=7, ncol=3, framealpha=0.9, loc="upper right")
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    _panel_label(ax_a, "A")

    # ── Panel B: resp vs non-resp similarity (zoomed 2021-2025) ─────────────
    ax_b = fig.add_subplot(grid[1, 0])
    if "resp_sim" in zoom.columns and "mean_other" in zoom.columns:
        ax_b.plot(zoom["window_start_dt"], zoom["resp_sim"],
                  color=COLORS["secondary"], linewidth=1.4, label="Respiratory similarity")
        ax_b.plot(zoom["window_start_dt"], zoom["mean_other"],
                  color=COLORS["primary"], linewidth=1.2, label="Mean non-respiratory similarity")
    ax_b.axvspan(EVENT_START, EVENT_END, color=COLORS["secondary"], alpha=0.16)
    ax_b.set_ylabel("Pearson profile correlation")
    ax_b.legend(fontsize=7, framealpha=0.9)
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    _panel_label(ax_b, "B")

    # ── Panel C: threshold operating characteristics ─────────────────────────
    ax_c = fig.add_subplot(grid[1, 1])
    perf = performance[performance["window_type"].eq("rolling4_weekly")].copy()
    order = ["rdi_mean_plus_1_5sd", "rdi_mean_plus_2sd", "rdi_p97_5"]
    perf_ordered = []
    for o in order:
        rows = perf[perf["threshold_rule"].eq(o)]
        if not rows.empty:
            perf_ordered.append(rows.iloc[0])
    if perf_ordered:
        perf = pd.DataFrame(perf_ordered)
        x = np.arange(len(perf))
        w = 0.25
        ax_c.bar(x - w, perf["sensitivity_event_week"].values, w,
                 color=COLORS["secondary"], label="Sensitivity")
        ax_c.bar(x,     perf["precision_ppv_lead_allowed"].values, w,
                 color=COLORS["primary"], label="PPV (4-week lead allowed)")
        ax_c.bar(x + w, perf["false_alarm_rate_lead_allowed"].values, w,
                 color=COLORS["warn"], label="False-alarm rate")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(["Mean+1.5SD", "Mean+2SD", "P97.5"], rotation=20, ha="right")
        ax_c.set_ylim(0, 1.05)
        ax_c.set_ylabel("Metric value")
        ax_c.legend(fontsize=7, framealpha=0.9)
        ax_c.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    _panel_label(ax_c, "C")

    # ── Panel D: COVID cardiac mechanism check ───────────────────────────────
    ax_d = fig.add_subplot(grid[1, 2])
    _plot_covid_cardiac_mechanism(ax_d)
    _panel_label(ax_d, "D")

    return fig


def main() -> int:
    FIG_CANONICAL.mkdir(parents=True, exist_ok=True)
    FIG_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load pre-computed results
    rolling = pd.read_csv(RES_DIR / "weekly_rdi_42k_rolling4_weekly.csv", encoding="utf-8-sig")
    performance = pd.read_csv(RES_DIR / "weekly_rdi_42k_performance.csv", encoding="utf-8-sig")

    summary = _load_json(RES_DIR / "weekly_rdi_42k_summary.json")
    threshold = summary["baseline"]["thresholds"]["rolling4_weekly"]["rdi_mean_plus_1_5sd"]

    fig = build_figure(rolling, performance, threshold)

    stem = "FigureS10_weekly_rdi_42k"
    for out_dir in (FIG_CANONICAL, FIG_OUTPUT):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=300,
                        bbox_inches="tight", facecolor="white")
        fig.savefig(out_dir / f"{stem}.tif", dpi=300,
                    bbox_inches="tight", facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        print(f"Saved {stem}.* to {out_dir}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
