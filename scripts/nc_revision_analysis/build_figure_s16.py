#!/usr/bin/env python3
"""Build Figure S16: Influenza-anchored validation extended analyses (4 panels).

Panel A: SCH quarterly Pearson profile correlation (6 groups)
Panel B: WHU consensus k-sweep Pareto (sensitivity, PPV, FAR vs k)
Panel C: WHU resp_score lag-Spearman with bootstrap 95% CI
Panel D: WHU FluNet subtype lag-correlation (A(H3), A(H1N1)pdm09, B)

Outputs: FigureS16_influenza_extended.{png,pdf,tif,svg} written to:
  - Submit/   (root)
  - NC_revision/lgdi_results/
  - NC_revision/resubmission_package_20260512/figures/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _panel_label(ax, label):
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left", color="black",
            zorder=1000,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      boxstyle="square,pad=0.15"))


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "lgdi_results"
ROOT = BASE.parent
PKG_FIGS = BASE / "resubmission_package_20260512" / "figures"

SCH_CSV    = RESULTS / "sch_quarterly_profile_correlation.csv"
KSWEEP_CSV = RESULTS / "lgdi_whu_consensus_k_sweep.csv"
BOOT_CSV   = RESULTS / "lgdi_whu_lag_spearman_bootstrap.csv"
SUB_CSV    = RESULTS / "lgdi_whu_flunet_subtype_lag.csv"

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes",
          "Cerebrovascular", "Renal", "Respiratory"]
GROUP_COLORS = {
    "Cardiovascular":  "#5DADE2",
    "Hypertension":    "#F4A261",
    "Diabetes":        "#66BB6A",
    "Cerebrovascular": "#B39DDB",
    "Renal":           "#4DD0C4",
    "Respiratory":     "#F4845F",
}
SUBTYPE_COLORS = {"AH3": "#F4845F", "AH1N12009": "#5DADE2", "INF_B": "#66BB6A"}


def panel_a(ax):
    df = pd.read_csv(SCH_CSV)
    quarters = sorted(df["quarter"].unique())
    x = np.arange(len(quarters))
    for g in GROUPS:
        sub = df[df["group"] == g].set_index("quarter").reindex(quarters)
        ax.plot(x, sub["pearson_r"].values, marker="o", linewidth=1.6,
                color=GROUP_COLORS[g], label=g, markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r vs reference profile")
    _panel_label(ax, "A")
    ax.set_title("SCH quarterly profile correlation (2016–2025)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


def panel_b(ax):
    df = pd.read_csv(KSWEEP_CSV).sort_values("k_groups_required")
    k = df["k_groups_required"].values
    ax.plot(k, df["sensitivity"].values, marker="o", color="#F4845F",
            label="Sensitivity", linewidth=2)
    ax.plot(k, df["ppv"].values, marker="s", color="#5DADE2",
            label="PPV", linewidth=2)
    ax.plot(k, df["false_alarm_rate"].values, marker="^", color="#90A4AE",
            label="False-alarm rate", linewidth=2)
    # Highlight Pareto-frontier ks
    for kk in (2, 3):
        ax.axvline(kk, color="gold", alpha=0.18, linewidth=8, zorder=0)
    ax.set_xlabel("k (number of groups required)")
    ax.set_ylabel("Metric")
    _panel_label(ax, "B")
    ax.set_title("WHU consensus k-sweep: minimum concurrent groups vs performance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.75)
    ax.set_xticks(range(1, 7))


def panel_c(ax):
    df = pd.read_csv(BOOT_CSV).sort_values("lag_weeks")
    lags = df["lag_weeks"].values
    rho = df["rho_point"].values
    lo = df["rho_low_2_5"].values
    hi = df["rho_high_97_5"].values
    err_lo = rho - lo
    err_hi = hi - rho
    ax.errorbar(lags, rho, yerr=[err_lo, err_hi], fmt="o-",
                color="#F4845F", ecolor="#888", capsize=4, linewidth=2,
                markersize=7, label="Spearman ρ (resp_score → FluNet)")
    ax.axhline(0, color="black", linewidth=0.8)
    # Highlight statistically significant lags
    for lg, rh, l in zip(lags, rho, lo):
        if l > 0:
            ax.annotate("*", xy=(lg, rh + 0.02), ha="center", fontsize=14, color="#F4845F")
    ax.set_xlabel("Lag (weeks, resp_score leads FluNet)")
    ax.set_ylabel("Spearman ρ (95% bootstrap CI)")
    _panel_label(ax, "C")
    ax.set_title("Lag-Spearman bootstrap CI: EHR z-score vs FluNet positivity")
    ax.set_xticks(range(0, 5))
    ax.grid(True, alpha=0.3)


def panel_d(ax):
    df = pd.read_csv(SUB_CSV)
    for sub in ["AH3", "AH1N12009", "INF_B"]:
        s = df[df["subtype"] == sub].sort_values("lag_weeks")
        ax.plot(s["lag_weeks"], s["spearman_rho"], marker="o",
                color=SUBTYPE_COLORS[sub], linewidth=2, markersize=6,
                label={"AH3": "A(H3)", "AH1N12009": "A(H1N1)pdm09", "INF_B": "B"}[sub])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (weeks, resp_score leads subtype)")
    ax.set_ylabel("Spearman ρ")
    _panel_label(ax, "D")
    ax.set_title("FluNet subtype lag-correlation: H3N2 vs H1N1 vs INF_B")
    ax.set_xticks(range(0, 5))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout()

    DEEP_FIGS = BASE / "DeepseekRevision" / "figures"
    out_dirs = [ROOT, RESULTS, PKG_FIGS, DEEP_FIGS]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    stem = "FigureS16_influenza_extended"
    for d in out_dirs:
        for ext in ("png", "pdf", "svg", "tif"):
            path = d / f"{stem}.{ext}"
            kwargs = {"dpi": 300} if ext in ("png", "tif") else {}
            if ext == "tif":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, bbox_inches="tight", **kwargs)
            print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
