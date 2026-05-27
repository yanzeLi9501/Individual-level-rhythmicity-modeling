#!/usr/bin/env python3
"""Regenerate Figure 3 (Plan B): replace Cosine-Similarity / RDI with LGDI.

Inputs:
  - NC_revision/lgdi_results/lgdi_whu_rolling4_weekly.csv  (Task 3 output)
  - NC_revision/lgdi_results/lgdi_whu_2019_focus.json
  - WHU primary all_admissions.csv (for respiratory admission-rate bars)

Layout (two panels, matching the original Figure 3 footprint):

  Panel A (left): Weekly LGDI respiratory residual score (from XGBoost LOS+gap
                  models trained on 2016-2018 WHU primary baseline) for
                  2019-01 through 2020-05. Overlay:
                    - mean+1.5 SD and mean+2 SD baseline thresholds
                    - monthly respiratory admission-rate bars (right axis)
                    - vertical lines for top alert weeks and pandemic onset
                    - annotations for first sustained alert and 2020-01 peak

  Panel B (right): Weekly LGDI per-group residual score (z relative to
                   2016-2018 baseline) heatmap for the same monitoring
                   window, replacing the original behavioural-deviation
                   heatmap with LGDI evidence.

Outputs (PNG/PDF/TIF/SVG): Figure3_early_warning.* written to the Submit/
root, NC_revision/lgdi_results/, and resubmission_package_20260512/figures/.
The original cosine version is preserved as
Figure3_early_warning_cosine_legacy.*.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent
SUBMIT = BASE.parent
LGDI_DIR = BASE / "lgdi_results"
PKG_FIG = BASE / "resubmission_package_20260512" / "figures"
WHU_RAW = Path(r"data\readmission_output\all_admissions.csv")

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
MONITOR_START = pd.Timestamp("2019-01-01")
MONITOR_END = pd.Timestamp("2020-05-31")
PANDEMIC_ONSET = pd.Timestamp("2020-01-23")  # Wuhan lockdown

RESPIRATORY_KEYWORDS = [
    "肺炎", "肺", "支气管", "哮喘", "COPD", "慢阻肺", "呼吸", "上呼吸道",
    "下呼吸道", "肺部感染", "肺结核", "肺癌", "肺栓塞", "气管", "急性支气管",
    "毛细支气管", "肺水肿", "胸腔", "胸膜", "Influenza", "Pneumonia",
]


def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv")
    df["window_anchor_dt"] = pd.to_datetime(df["window_anchor"])
    return df


def baseline_stats(df: pd.DataFrame) -> dict:
    base = df[(df["valid"].astype(bool)) & df["window_anchor_dt"].between("2016-01-01", "2018-12-31")]
    stats: dict = {}
    stats["resp_mean"] = float(base["resp_score"].mean())
    stats["resp_std"] = float(base["resp_score"].std())
    stats["lgdi_mean"] = float(base["lgdi"].mean())
    stats["lgdi_std"] = float(base["lgdi"].std())
    for g in GROUPS:
        col = f"score_{g}"
        if col in base.columns:
            stats[f"{g}_mean"] = float(base[col].mean())
            stats[f"{g}_std"] = float(base[col].std())
    return stats


def respiratory_monthly_rate() -> pd.DataFrame:
    """Monthly respiratory admission rate (% of all admissions) at WHU."""
    if not WHU_RAW.exists():
        return pd.DataFrame(columns=["month", "resp_pct", "n_total", "n_resp"])
    cols = ["入院日期", "诊断文本", "EMR_初步诊断"]
    avail = pd.read_csv(WHU_RAW, nrows=1).columns.tolist()
    use = [c for c in cols if c in avail]
    df = pd.read_csv(WHU_RAW, usecols=use, low_memory=False)
    df["入院日期"] = pd.to_datetime(df["入院日期"], errors="coerce")
    df = df.dropna(subset=["入院日期"])
    text = df.get("诊断文本", "").fillna("").astype(str)
    if "EMR_初步诊断" in df.columns:
        text = text + " " + df["EMR_初步诊断"].fillna("").astype(str)
    pat = "|".join(RESPIRATORY_KEYWORDS)
    df["is_resp"] = text.str.contains(pat, na=False, regex=True)
    df["month"] = df["入院日期"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month").agg(n_total=("is_resp", "size"), n_resp=("is_resp", "sum")).reset_index()
    monthly["resp_pct"] = 100.0 * monthly["n_resp"] / monthly["n_total"].clip(lower=1)
    return monthly[(monthly["month"] >= MONITOR_START) & (monthly["month"] <= MONITOR_END)].copy()


def fmt_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_ha("right")


def panel_a(ax: plt.Axes, weekly: pd.DataFrame, stats: dict, monthly_rate: pd.DataFrame) -> None:
    win = weekly[
        weekly["valid"].astype(bool)
        & weekly["window_anchor_dt"].between(MONITOR_START, MONITOR_END)
    ].copy().sort_values("window_anchor_dt")

    # Right axis: respiratory admission-rate bars
    if not monthly_rate.empty:
        ax2 = ax.twinx()
        ax2.bar(monthly_rate["month"], monthly_rate["resp_pct"], width=22,
                color="#E9C46A", alpha=0.45, label="Respiratory admission rate (%)", zorder=1)
        ax2.set_ylabel("Respiratory admission rate (%)", color="#B07A1E")
        ax2.tick_params(axis="y", labelcolor="#B07A1E")
        ax2.set_ylim(0, max(60.0, monthly_rate["resp_pct"].max() * 1.2))

    # LGDI threshold bands
    thr_15 = stats["resp_mean"] + 1.5 * stats["resp_std"]
    thr_20 = stats["resp_mean"] + 2.0 * stats["resp_std"]
    ax.axhspan(stats["resp_mean"] - stats["resp_std"], stats["resp_mean"] + stats["resp_std"],
               color="#B0BEC5", alpha=0.18, label="Baseline ±1 SD (2016-2018)")
    ax.axhline(thr_15, ls="--", color="#E76F51", lw=1.4,
               label=f"Mean + 1.5 SD = {thr_15:.2f}")
    ax.axhline(thr_20, ls=":", color="#9B2226", lw=1.4,
               label=f"Mean + 2 SD = {thr_20:.2f}")

    # LGDI respiratory line
    ax.plot(win["window_anchor_dt"], win["resp_score"],
            color="#264653", marker="o", ms=4.0, lw=1.6, zorder=4,
            label="Respiratory LGDI residual score (4-week rolling)")

    # Pandemic onset
    ax.axvline(PANDEMIC_ONSET, color="#9B2226", ls="--", lw=1.4, zorder=3)
    ax.annotate("Pandemic onset\n(2020-01-23)", xy=(PANDEMIC_ONSET, ax.get_ylim()[1] * 0.92),
                xytext=(PANDEMIC_ONSET - pd.Timedelta(days=160), ax.get_ylim()[1] * 0.92),
                fontsize=8.5, color="#9B2226", ha="right",
                arrowprops=dict(arrowstyle="-", color="#9B2226", lw=0.8))

    # Top alert annotations: first crossing of mean+1.5SD in late 2019, and 2020-01 peak
    over = win[win["resp_score"] >= thr_15].sort_values("window_anchor_dt")
    if not over.empty:
        first = over.iloc[0]
        ax.annotate(f"First sustained alert\n{first['window_anchor']} ({first['resp_score']:.2f})",
                    xy=(first["window_anchor_dt"], first["resp_score"]),
                    xytext=(first["window_anchor_dt"] - pd.Timedelta(days=80),
                            first["resp_score"] + 0.35),
                    fontsize=8.5, color="#264653",
                    arrowprops=dict(arrowstyle="->", color="#264653", lw=0.9))
    peak = win.sort_values("resp_score", ascending=False).iloc[0]
    ax.annotate(f"Peak: {peak['window_anchor']}\nresp_score={peak['resp_score']:.2f}",
                xy=(peak["window_anchor_dt"], peak["resp_score"]),
                xytext=(peak["window_anchor_dt"] + pd.Timedelta(days=15),
                        peak["resp_score"] - 0.35),
                fontsize=8.5, color="#9B2226",
                arrowprops=dict(arrowstyle="->", color="#9B2226", lw=0.9))

    ax.set_ylabel("Respiratory LGDI residual score", color="#264653")
    ax.tick_params(axis="y", labelcolor="#264653")
    ax.set_title("LGDI surveillance timeline (WHU primary, retrospective first-prediction 2019-2020)")
    fmt_axis(ax)
    ax.set_xlim(MONITOR_START - pd.Timedelta(days=10), MONITOR_END + pd.Timedelta(days=10))
    ax.set_ylim(min(-0.4, win["resp_score"].min() - 0.1), max(1.7, win["resp_score"].max() * 1.15))
    ax.grid(axis="y", color="#CFD8DC", lw=0.4, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)


def panel_b(ax: plt.Axes, weekly: pd.DataFrame, stats: dict) -> None:
    win = weekly[
        weekly["valid"].astype(bool)
        & weekly["window_anchor_dt"].between(MONITOR_START, MONITOR_END)
    ].copy().sort_values("window_anchor_dt")
    matrix = []
    for g in GROUPS:
        col = f"score_{g}"
        mu = stats.get(f"{g}_mean", 0.0)
        sd = stats.get(f"{g}_std", 1.0) or 1.0
        z = (win[col] - mu) / sd
        matrix.append(z.to_numpy())
    matrix = np.array(matrix)

    cmap = LinearSegmentedColormap.from_list(
        "lgdi_div", ["#1B7837", "#A6DBA0", "#F7F7F7", "#F4A582", "#B2182B"])
    vmax = float(np.nanmax(np.abs(matrix))) or 3.0
    vmax = min(max(vmax, 3.0), 8.0)
    extent = [
        mdates.date2num(win["window_anchor_dt"].iloc[0]),
        mdates.date2num(win["window_anchor_dt"].iloc[-1]),
        len(GROUPS) - 0.5, -0.5,
    ]
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", extent=extent)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels(GROUPS)
    ax.set_title("Per-group LGDI residual score (z relative to 2016-2018 baseline)")
    ax.xaxis_date()
    fmt_axis(ax)
    ax.axvline(PANDEMIC_ONSET, color="#9B2226", ls="--", lw=1.2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score (group residual vs. baseline)")


def main() -> int:
    weekly = load_weekly()
    stats = baseline_stats(weekly)
    monthly_rate = respiratory_monthly_rate()
    print("Baseline (2016-2018) resp_score mean/std:",
          f"{stats['resp_mean']:.4f} / {stats['resp_std']:.4f}")
    if monthly_rate.empty:
        print("WARNING: WHU raw not available; respiratory admission-rate bars omitted")
    else:
        print(f"Monthly respiratory rate rows: {len(monthly_rate)}; "
              f"max%={monthly_rate['resp_pct'].max():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    panel_a(axes[0], weekly, stats, monthly_rate)
    panel_b(axes[1], weekly, stats)
    fig.text(0.005, 0.96, "A", fontsize=15, fontweight="bold")
    fig.text(0.605, 0.96, "B", fontsize=15, fontweight="bold")
    fig.tight_layout()

    # Preserve legacy cosine version, then write new outputs to all 3 destinations
    targets = [
        SUBMIT,
        LGDI_DIR,
        PKG_FIG,
    ]
    PKG_FIG.mkdir(parents=True, exist_ok=True)
    LGDI_DIR.mkdir(parents=True, exist_ok=True)
    for tdir in [SUBMIT, PKG_FIG]:
        for ext in (".png", ".pdf", ".tif"):
            old = tdir / f"Figure3_early_warning{ext}"
            legacy = tdir / f"Figure3_early_warning_cosine_legacy{ext}"
            if old.exists() and not legacy.exists():
                shutil.copy2(old, legacy)

    for tdir in targets:
        for ext in (".png", ".pdf", ".tif", ".svg"):
            out = tdir / f"Figure3_early_warning{ext}"
            if ext == ".tif":
                fig.savefig(out, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
            else:
                fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  wrote {out}")
    plt.close(fig)

    # Side-car JSON describing what was plotted
    side = {
        "panels": {
            "A": "Weekly LGDI respiratory residual score (4-week rolling) "
                 "with baseline thresholds and respiratory admission-rate bars",
            "B": "Per-group weekly LGDI residual score z-scored vs. 2016-2018 baseline",
        },
        "baseline_window": "2016-01-01 to 2018-12-31",
        "monitor_window": f"{MONITOR_START.date()} to {MONITOR_END.date()}",
        "thresholds": {
            "respiratory_mean_plus_1_5sd": stats["resp_mean"] + 1.5 * stats["resp_std"],
            "respiratory_mean_plus_2sd": stats["resp_mean"] + 2.0 * stats["resp_std"],
        },
        "baseline_stats": stats,
        "replaces": "Original cosine-similarity / RDI Figure 3",
        "legacy_cosine_files": "Figure3_early_warning_cosine_legacy.{png,pdf,tif}",
    }
    (LGDI_DIR / "Figure3_early_warning_lgdi_planB.json").write_text(
        json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
