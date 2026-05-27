from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis_outputs"
SUPP = ROOT / "supp_figures"

POLICY_DATE = pd.Timestamp("2018-09-01")


COLORS = {
    "ink": "#202124",
    "muted": "#5f6368",
    "grid": "#d9d9d9",
    "blue": "#1976d2",
    "orange": "#ef7d22",
    "purple": "#7b1fa2",
    "red": "#c62828",
    "green": "#2e7d32",
    "gray": "#8d8d8d",
}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg", "tif"]:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(ANALYSIS / "whu_policy_xgb_daily_volume_detection.csv", parse_dates=["date"])
    weekly = pd.read_csv(ANALYSIS / "whu_policy_xgb_volume_detection_weekly.csv", parse_dates=["week"])
    chronic = pd.read_csv(ANALYSIS / "whu_policy_chronic_group_sensitivity_rolling4.csv")
    return daily, weekly, chronic


def panel_a_daily_xgb(ax: plt.Axes, daily: pd.DataFrame) -> None:
    panel_label(ax, "A")
    ax.set_title("Daily volume forecast")
    focus = daily[(daily["date"] >= "2018-07-01") & (daily["date"] <= "2019-02-28")].copy()
    ax.plot(focus["date"], focus["n_admissions"], color=COLORS["ink"], lw=0.9, label="Observed daily admissions")
    ax.plot(focus["date"], focus["n_admissions_pred"], color=COLORS["blue"], lw=1.2, label="XGBoost expected")
    point_alert = focus[focus["n_admissions_point_alert"]]
    shift_alert = focus[focus["n_admissions_roll14_shift_alert"]]
    ax.scatter(
        point_alert["date"],
        point_alert["n_admissions"],
        s=28,
        color=COLORS["red"],
        zorder=3,
        label="Point residual alert",
    )
    ax.scatter(
        shift_alert["date"],
        shift_alert["n_admissions"],
        s=36,
        facecolors="none",
        edgecolors=COLORS["purple"],
        linewidths=1.0,
        zorder=3,
        label="14-day shift alert",
    )
    ax.axvline(POLICY_DATE, color=COLORS["purple"], ls="--", lw=1.1, label="Policy date")
    ax.set_ylabel("Daily admissions")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="upper left", ncol=2, fontsize=7, frameon=False)


def panel_b_delayed_response(ax: plt.Axes, daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    panel_label(ax, "B")
    ax.set_title("Windowed volume response")
    windows = [
        ("Pre 28 d", pd.Timestamp("2018-08-04"), pd.Timestamp("2018-08-31")),
        ("Post 28 d", pd.Timestamp("2018-09-01"), pd.Timestamp("2018-09-28")),
        ("Nov 2018", pd.Timestamp("2018-11-01"), pd.Timestamp("2018-11-30")),
        ("Post through 2019", pd.Timestamp("2018-09-01"), pd.Timestamp("2019-12-31")),
    ]
    means = []
    patients = []
    for _, start, end in windows:
        sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        means.append(sub["n_admissions"].mean())
        patients.append(sub["n_patients"].mean())
    x = np.arange(len(windows))
    width = 0.34
    ax.bar(x - width / 2, means, width=width, color=COLORS["blue"], label="Admissions/day")
    ax.bar(x + width / 2, patients, width=width, color=COLORS["orange"], label="Patients/day")
    ax.axhline(means[0], color=COLORS["blue"], lw=0.8, ls=":", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([w[0] for w in windows], rotation=0)
    ax.set_ylabel("Daily mean")
    ax.set_ylim(0, max(means + patients) * 1.28)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def panel_c_sensitive_groups(ax: plt.Axes, chronic: pd.DataFrame) -> None:
    panel_label(ax, "C")
    ax.set_title("Chronic-group sensitivity")
    plot = chronic.sort_values("postrise8_ratio_vs_pre", ascending=True)
    y = np.arange(len(plot))
    vals = (plot["postrise8_ratio_vs_pre"] - 1.0) * 100
    colors = [
        COLORS["green"] if group in {"Cerebrovascular", "Diabetes"} else COLORS["gray"]
        for group in plot["group"]
    ]
    ax.barh(y, vals, color=colors, height=0.62, label="Higher sensitivity")
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            vals.loc[row.name] + 0.8,
            i,
            f"{vals.loc[row.name]:.0f}%",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["group"])
    ax.set_xlabel("4-week rolling count increase, late Oct-Dec vs Jul-Aug (%)")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    daily, weekly, chronic = load_data()
    fig = plt.figure(figsize=(12.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.18, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_a_daily_xgb(ax_a, daily)
    panel_b_delayed_response(ax_b, daily, weekly)
    panel_c_sensitive_groups(ax_c, chronic)
    save_figure(fig, SUPP / "FigureS11")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
