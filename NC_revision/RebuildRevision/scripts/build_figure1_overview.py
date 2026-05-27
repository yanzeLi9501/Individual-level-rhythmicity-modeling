from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebuild_common import (
    COLORS,
    FIGURES_DIR,
    LGDI_DIR,
    OUTPUTS_DIR,
    SHADING,
    add_note,
    apply_style,
    ensure_dirs,
    panel_label,
    read_csv,
    save_figure,
)


def box(ax, xy, width, height, text, color, fontsize=8.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["dark"],
        wrap=True,
    )


def arrow(ax, start, end, color=COLORS["gray"]):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def panel_data_sources(ax):
    ax.set_title("Data sources\nand evidence roles", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.05, 0.62), 0.40, 0.22, "WHU EHR\nDiscovery set\nretrospective proof-of-concept", COLORS["blue"])
    box(ax, (0.55, 0.62), 0.40, 0.22, "WHO FluNet China\nPrimary endpoint\n213 monitor weeks", COLORS["green"])
    box(ax, (0.05, 0.18), 0.40, 0.22, "WHU cardiac expansion\nSame-system audit\nnot independent validation", COLORS["purple"])
    box(ax, (0.55, 0.18), 0.40, 0.22, "Public ICU datasets\nCross-setting positive controls\nnative features only", COLORS["orange"])
    arrow(ax, (0.45, 0.72), (0.55, 0.72), COLORS["green"])
    arrow(ax, (0.45, 0.29), (0.55, 0.29), COLORS["orange"])


def panel_profile(ax):
    ax.set_title("Profile construction\nfrom chronic-disease groups", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Weekly admissions\nby diagnosis group", COLORS["blue"]),
        ("Baseline calibration\n2016-2018", COLORS["teal"]),
        ("Residual profiles\nstandardized vectors", COLORS["purple"]),
        ("Pearson profile\ncorrelation", COLORS["green"]),
    ]
    x = 0.04
    for idx, (text, color) in enumerate(steps):
        box(ax, (x, 0.45), 0.20, 0.25, text, color, fontsize=8)
        if idx < len(steps) - 1:
            arrow(ax, (x + 0.20, 0.57), (x + 0.25, 0.57), color)
        x += 0.25
    ax.text(0.06, 0.18, "Multi-system co-elevation,\nnot respiratory sentinel status", fontsize=8.0)


def panel_rules(ax):
    ax.set_title("S1\u20134 alert-rule hierarchy", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        ("S1", "single respiratory threshold", COLORS["blue"]),
        ("S2", "multi-group consensus >=2", COLORS["green"]),
        ("S3", "season-constrained consensus", COLORS["orange"]),
        ("S4", "Nov-Mar sustained consensus", COLORS["purple"]),
    ]
    for idx, (name, text, color) in enumerate(rows):
        y = 0.72 - idx * 0.17
        box(ax, (0.08, y), 0.15, 0.10, name, color, fontsize=9.5)
        box(ax, (0.28, y), 0.62, 0.10, text, color, fontsize=8.5)
        arrow(ax, (0.23, y + 0.05), (0.28, y + 0.05), color)


def panel_xgb(ax):
    ax.set_title("XGBoost bridge:\nthree-layer architecture", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layers = [
        ("Layer 1\nweekly group residuals", COLORS["blue"]),
        ("Layer 2\nconsensus alert weeks", COLORS["green"]),
        ("Layer 3\nXGBoost FluNet fit\nrepresentative n=2478", COLORS["orange"]),
    ]
    for idx, (text, color) in enumerate(layers):
        x = 0.08 + idx * 0.30
        box(ax, (x, 0.43), 0.23, 0.28, text, color)
        if idx < 2:
            arrow(ax, (x + 0.23, 0.57), (x + 0.30, 0.57), color)
    ax.text(0.08, 0.20, "Frequent-visitor high R\u00b2 is\nselective supplementary evidence.", fontsize=8.0)


def panel_external(ax):
    ax.set_title("External public-data strategy", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    datasets = ["MIMIC-IV", "eICU", "NWICU"]
    for idx, name in enumerate(datasets):
        y = 0.72 - idx * 0.22
        box(ax, (0.06, y), 0.22, 0.13, name, COLORS["purple"])
        box(ax, (0.38, y), 0.24, 0.13, "native feature set", COLORS["teal"])
        box(ax, (0.72, y), 0.22, 0.13, "independent model", COLORS["orange"])
        arrow(ax, (0.28, y + 0.065), (0.38, y + 0.065), COLORS["gray"])
        arrow(ax, (0.62, y + 0.065), (0.72, y + 0.065), COLORS["gray"])
    ax.text(0.06, 0.08, "Cross-setting positive controls,\nnot external validation.", fontsize=8.0)


_FLU_SEASONS = [
    ("'15-'16", pd.Timestamp("2015-11-01"), pd.Timestamp("2016-04-30")),
    ("'16-'17", pd.Timestamp("2016-11-01"), pd.Timestamp("2017-04-30")),
    ("'17-'18", pd.Timestamp("2017-11-01"), pd.Timestamp("2018-04-30")),
    ("'18-'19", pd.Timestamp("2018-11-01"), pd.Timestamp("2019-04-30")),
    ("'19-'20", pd.Timestamp("2019-11-01"), pd.Timestamp("2020-04-30")),
]


def panel_pregap(ax):
    """Pre-Gap WHU monthly admissions with influenza-season shading."""
    ax.set_title("Pre-Gap WHU admissions with influenza seasons\n(monitoring period 2016\u20132020)")
    roll_path = LGDI_DIR / "lgdi_whu_rolling4_weekly.csv"
    if not roll_path.exists():
        add_note(ax, "lgdi_whu_rolling4_weekly.csv not found.")
        return
    roll = read_csv(roll_path)
    roll["window_anchor"] = pd.to_datetime(roll["window_anchor"])
    roll["month"] = roll["window_anchor"].dt.to_period("M").dt.to_timestamp()
    monthly = roll.groupby("month")["n_admissions"].sum().reset_index()
    ax.bar(monthly["month"], monthly["n_admissions"],
           width=25, color=COLORS["blue"], alpha=0.78, label="Monthly admissions")
    for season_id, t_start, t_end in _FLU_SEASONS:
        ax.axvspan(t_start, t_end,
                   color=SHADING["cream"][0], alpha=SHADING["cream"][1] + 0.12, zorder=0)
        mid = t_start + (t_end - t_start) / 2
        y_top = float(monthly["n_admissions"].max()) * 0.93 if not monthly.empty else 1000
        ax.text(mid, y_top, season_id, ha="center", va="top",
                fontsize=6.5, color=COLORS["orange"], rotation=0)
    gap_start = pd.Timestamp("2020-02-01")
    ax.axvline(gap_start, color=COLORS["red"], lw=1.2, ls="--")
    ax.text(gap_start, float(monthly["n_admissions"].max()) * 0.65,
            "  data gap\u2192", fontsize=7, color=COLORS["red"], va="bottom")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Monthly admissions")
    ax.legend(fontsize=7.5, loc="upper left")


def panel_postgap(ax):
    """Post-Gap cardiac admissions 2020\u20132024 with EHR expansion marker."""
    ax.set_title("Post-Gap cardiac admissions 2020\u20132024\n(EHR system expansion marker)")
    pg_path = OUTPUTS_DIR / "xgb_symmetric" / "cardiac_extended_covid_lgdi_timeline.csv"
    if not pg_path.exists():
        add_note(ax, "cardiac_extended_covid_lgdi_timeline.csv not found.")
        return
    pg = pd.read_csv(pg_path, parse_dates=["week_start"])
    pg["month"] = pg["week_start"].dt.to_period("M").dt.to_timestamp()
    pg_monthly = pg.groupby("month")["n"].sum().reset_index()
    # Pre-gap mean reference
    roll_path = LGDI_DIR / "lgdi_whu_rolling4_weekly.csv"
    pre_mean = None
    if roll_path.exists():
        roll_ref = read_csv(roll_path)
        roll_ref["window_anchor"] = pd.to_datetime(roll_ref["window_anchor"])
        roll_ref["month"] = roll_ref["window_anchor"].dt.to_period("M").dt.to_timestamp()
        pre_mean = float(roll_ref.groupby("month")["n_admissions"].sum().mean())
    ax.bar(pg_monthly["month"], pg_monthly["n"],
           width=25, color=COLORS["amber"], alpha=0.78, label="Monthly admissions")
    ehr_exp = pd.Timestamp("2021-09-01")
    ax.axvline(ehr_exp, color=COLORS["red"], lw=1.5, ls="--", zorder=3)
    y_top = float(pg_monthly["n"].max()) if not pg_monthly.empty else 1
    ax.text(ehr_exp, y_top * 0.93,
            " EHR expansion\n \u00d76 monthly admissions",
            fontsize=7, color=COLORS["red"], va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8,
                      ec=COLORS["red"], lw=0.8))
    if pre_mean is not None:
        ax.axhline(pre_mean, color=COLORS["blue"], lw=1.1, ls=":",
                   label=f"Pre-Gap mean ({pre_mean:.0f}/mo)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Monthly admissions")
    ax.legend(fontsize=7.5, loc="upper left")


def main() -> None:
    ensure_dirs()
    apply_style()

    fig = plt.figure(figsize=(20.0, 5.0), constrained_layout=True)
    # Single merged Panel A: five schematic overview sub-panels
    fig.text(0.005, 0.97, "A", fontsize=14, fontweight="bold",
             va="top", transform=fig.transFigure, color=COLORS["dark"])
    axes = fig.subplots(1, 5, gridspec_kw={"wspace": 0.08})
    panel_data_sources(axes[0])
    panel_profile(axes[1])
    panel_rules(axes[2])
    panel_xgb(axes[3])
    panel_external(axes[4])

    save_figure(fig, FIGURES_DIR / "Figure1")


if __name__ == "__main__":
    main()