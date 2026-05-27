#!/usr/bin/env python3
"""Figure 3 v3 — 4 panel layout combining LGDI surveillance + influenza
validation + WHU annual coverage + performance comparison.

Panels:
  A (top-left) : Weekly respiratory LGDI residual score timeline 2019-2020
                 with FluNet-defined influenza-season shading, baseline
                 thresholds, monthly admission-rate bars, pandemic onset.
  B (top-right): Per-group LGDI residual z-score heatmap 2019-2020.
  C (bot-left) : WHU annual admissions 2012-2020 stacked bar
                 (total vs. respiratory subset) -- documents data coverage.
  D (bot-right): Strategy performance comparison -- COVID-coded reference
                 endpoint vs. influenza-season endpoint, three metrics
                 (sensitivity / PPV / FAR), grouped horizontal bars.

Inputs:
  - NC_revision/lgdi_results/lgdi_whu_rolling4_weekly.csv
  - NC_revision/lgdi_results/lgdi_whu_influenza_validation.csv  (Phase 2)
  - NC_revision/lgdi_results/lgdi_whu_influenza_metrics.csv      (Phase 2)
  - NC_revision/lgdi_results/lgdi_optimized_performance.csv      (COVID side)
  - all_admissions.csv (annual histogram + monthly admission rate)
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

BASE = Path(__file__).resolve().parent
SUBMIT = BASE.parent
LGDI_DIR = BASE / "lgdi_results"
PKG_FIG = BASE / "resubmission_package_20260512" / "figures"
WHU_RAW = Path(r"data\readmission_output\all_admissions.csv")

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes",
          "Cerebrovascular", "Renal", "Respiratory"]
MONITOR_START = pd.Timestamp("2019-01-01")
MONITOR_END = pd.Timestamp("2020-05-31")
PANDEMIC_ONSET = pd.Timestamp("2020-01-23")

RESPIRATORY_KEYWORDS = [
    "肺炎", "肺", "支气管", "哮喘", "COPD", "慢阻肺", "呼吸", "上呼吸道",
    "下呼吸道", "肺部感染", "肺结核", "肺癌", "肺栓塞", "气管", "急性支气管",
    "毛细支气管", "肺水肿", "胸腔", "胸膜", "Influenza", "Pneumonia",
]


# ----------------------- data loading -----------------------

def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv")
    df["window_anchor_dt"] = pd.to_datetime(df["window_anchor"])
    return df


def baseline_stats(df: pd.DataFrame) -> dict:
    base = df[df["valid"].astype(bool)
              & df["window_anchor_dt"].between("2016-01-01", "2018-12-31")]
    stats: dict = {
        "resp_mean": float(base["resp_score"].mean()),
        "resp_std": float(base["resp_score"].std()),
        "lgdi_mean": float(base["lgdi"].mean()),
        "lgdi_std": float(base["lgdi"].std()),
    }
    for g in GROUPS:
        col = f"score_{g}"
        if col in base.columns:
            stats[f"{g}_mean"] = float(base[col].mean())
            stats[f"{g}_std"] = float(base[col].std())
    return stats


def load_admissions() -> pd.DataFrame:
    cols_avail = pd.read_csv(WHU_RAW, nrows=1).columns.tolist()
    use = [c for c in ["入院日期", "诊断文本", "EMR_初步诊断"] if c in cols_avail]
    df = pd.read_csv(WHU_RAW, usecols=use, low_memory=False)
    df["入院日期"] = pd.to_datetime(df["入院日期"], errors="coerce")
    df = df.dropna(subset=["入院日期"]).copy()
    text = df.get("诊断文本", "").fillna("").astype(str)
    if "EMR_初步诊断" in df.columns:
        text = text + " " + df["EMR_初步诊断"].fillna("").astype(str)
    pat = "|".join(RESPIRATORY_KEYWORDS)
    df["is_resp"] = text.str.contains(pat, na=False, regex=True)
    return df


def respiratory_monthly_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["入院日期"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month").agg(
        n_total=("is_resp", "size"),
        n_resp=("is_resp", "sum"),
    ).reset_index()
    monthly["resp_pct"] = 100.0 * monthly["n_resp"] / monthly["n_total"].clip(lower=1)
    return monthly[(monthly["month"] >= MONITOR_START)
                   & (monthly["month"] <= MONITOR_END)].copy()


def annual_admissions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["入院日期"].dt.year
    yearly = df.groupby("year").agg(
        n_total=("is_resp", "size"),
        n_resp=("is_resp", "sum"),
    ).reset_index()
    yearly["n_other"] = yearly["n_total"] - yearly["n_resp"]
    return yearly[yearly["year"].between(2012, 2020)].copy()


def load_flu_labels() -> pd.DataFrame:
    df = pd.read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    flu = pd.read_csv(LGDI_DIR / "lgdi_whu_influenza_metrics.csv")
    cov = pd.read_csv(LGDI_DIR / "lgdi_optimized_performance.csv")
    return flu, cov


def fmt_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_ha("right")


# ----------------------- panels -----------------------

def panel_a(ax: plt.Axes, weekly: pd.DataFrame, stats: dict,
            monthly_rate: pd.DataFrame, flu: pd.DataFrame) -> None:
    win = weekly[weekly["valid"].astype(bool)
                 & weekly["window_anchor_dt"].between(MONITOR_START, MONITOR_END)
                 ].copy().sort_values("window_anchor_dt")

    # Influenza-season shading (FluNet positivity above mean+1SD, NH season)
    flu_win = flu[flu["week_start"].between(MONITOR_START, MONITOR_END)].copy()
    flu_pos = flu_win[flu_win["flu_event_window"]].sort_values("week_start")
    if not flu_pos.empty:
        # Group consecutive weeks
        ws = flu_pos["week_start"].tolist()
        spans = []
        s = ws[0]
        prev = ws[0]
        for d in ws[1:]:
            if (d - prev).days <= 8:
                prev = d
            else:
                spans.append((s, prev + pd.Timedelta(days=7)))
                s = d
                prev = d
        spans.append((s, prev + pd.Timedelta(days=7)))
        first = True
        for a, b in spans:
            ax.axvspan(a, b, color="#F4A261", alpha=0.18,
                       label="Influenza season (FluNet, above baseline)" if first else None,
                       zorder=0)
            first = False

    # Right axis: respiratory admission-rate bars
    if not monthly_rate.empty:
        ax2 = ax.twinx()
        ax2.bar(monthly_rate["month"], monthly_rate["resp_pct"], width=22,
                color="#E9C46A", alpha=0.45,
                label="Respiratory admission rate (%)", zorder=1)
        ax2.set_ylabel("Respiratory admission rate (%)", color="#B07A1E")
        ax2.tick_params(axis="y", labelcolor="#B07A1E")
        ax2.set_ylim(0, max(60.0, monthly_rate["resp_pct"].max() * 1.2))

    # LGDI threshold bands
    thr_15 = stats["resp_mean"] + 1.5 * stats["resp_std"]
    thr_20 = stats["resp_mean"] + 2.0 * stats["resp_std"]
    ax.axhspan(stats["resp_mean"] - stats["resp_std"],
               stats["resp_mean"] + stats["resp_std"],
               color="#B0BEC5", alpha=0.18,
               label="Baseline ±1 SD (2016-2018)")
    ax.axhline(thr_15, ls="--", color="#E76F51", lw=1.4,
               label=f"Mean + 1.5 SD = {thr_15:.2f}")
    ax.axhline(thr_20, ls=":", color="#9B2226", lw=1.4,
               label=f"Mean + 2 SD = {thr_20:.2f}")

    # LGDI line
    ax.plot(win["window_anchor_dt"], win["resp_score"],
            color="#264653", marker="o", ms=4.0, lw=1.6, zorder=4,
            label="Respiratory LGDI residual score (4-week rolling)")

    # Pandemic onset
    ax.axvline(PANDEMIC_ONSET, color="#9B2226", ls="--", lw=1.4, zorder=3)

    # Annotations
    over = win[win["resp_score"] >= thr_15].sort_values("window_anchor_dt")
    if not over.empty:
        first_alert = over.iloc[0]
        ax.annotate(f"First sustained alert\n{first_alert['window_anchor']} "
                    f"({first_alert['resp_score']:.2f})",
                    xy=(first_alert["window_anchor_dt"], first_alert["resp_score"]),
                    xytext=(first_alert["window_anchor_dt"] - pd.Timedelta(days=80),
                            first_alert["resp_score"] + 0.35),
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
    ax.set_title("A. LGDI surveillance timeline with FluNet influenza-season "
                 "shading (WHU primary, 2019-2020)",
                 fontsize=10.5, loc="left")
    fmt_axis(ax)
    ax.set_xlim(MONITOR_START - pd.Timedelta(days=10),
                MONITOR_END + pd.Timedelta(days=10))
    ax.set_ylim(min(-0.4, win["resp_score"].min() - 0.1),
                max(1.7, win["resp_score"].max() * 1.15))
    ax.grid(axis="y", color="#CFD8DC", lw=0.4, alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.92, ncol=2)


def panel_b(ax: plt.Axes, weekly: pd.DataFrame, stats: dict) -> None:
    win = weekly[weekly["valid"].astype(bool)
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
    ax.set_yticklabels(GROUPS, fontsize=9)
    ax.set_title("B. Per-group LGDI residual z-score (vs. 2016-2018 baseline)",
                 fontsize=10.5, loc="left")
    ax.xaxis_date()
    fmt_axis(ax)
    ax.axvline(PANDEMIC_ONSET, color="#9B2226", ls="--", lw=1.2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score (group residual vs. baseline)", fontsize=8.5)


def panel_c(ax: plt.Axes, yearly: pd.DataFrame) -> None:
    years = yearly["year"].astype(int).tolist()
    n_other = yearly["n_other"].values
    n_resp = yearly["n_resp"].values
    width = 0.7
    ax.bar(years, n_other, width, color="#557B97", label="Non-respiratory")
    ax.bar(years, n_resp, width, bottom=n_other, color="#E76F51",
           label="Respiratory")
    # Annotate totals on top
    totals = yearly["n_total"].values
    for x, t in zip(years, totals):
        ax.text(x, t + max(totals) * 0.01, f"{int(t):,}",
                ha="center", va="bottom", fontsize=7.8, color="#333")
    # Highlight COVID-disrupted year
    if 2020 in years:
        ax.axvline(2020 - 0.4, color="#9B2226", ls=":", lw=0.8, alpha=0.6)
        ax.text(2020, max(totals) * 0.97,
                "2020 truncated\n(May 24)",
                ha="center", va="top", fontsize=7.5, color="#9B2226")
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("WHU primary admissions per year")
    ax.set_title("C. WHU primary annual admissions (2012-2020): total vs. "
                 "respiratory subset", fontsize=10.5, loc="left")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=0, fontsize=8.5)
    ax.grid(axis="y", color="#CFD8DC", lw=0.4, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, max(totals) * 1.10)


def panel_d(ax: plt.Axes, flu_metrics: pd.DataFrame,
            cov_metrics: pd.DataFrame) -> None:
    """Grouped horizontal bars: strategies × {sensitivity, PPV, FAR}.
    Two endpoint families side-by-side: Influenza vs. COVID-coded."""

    # Map influenza strategies (Phase 2 output) to display labels
    flu_map = {
        "REFERENCE_resp_mean_plus_1_5sd": "RespScore mean+1.5SD",
        "REFERENCE_resp_mean_plus_2sd": "RespScore mean+2SD",
        "LGDI_mean_plus_1_5sd": "LGDI mean+1.5SD",
        "consensus_2groups_mean_plus_1_5sd": "Consensus (\u22652 groups)",
        "consensus_3groups_mean_plus_1_5sd": "Consensus (\u22653 groups)",
        "REFERENCE_resp_mean_plus_1sd_relaxed": "RespScore mean+1SD",
    }
    # Map COVID strategies (lgdi_optimized_performance.csv)
    cov_map = {
        "REFERENCE_lgdi_mean_plus_1_5sd": "LGDI mean+1.5SD",
        "multi_scale_2w_or_4w": "Multi-scale 2w|4w",
        "cusum_k0_5sd_h4sd": "CUSUM (k=0.5SD, h=4SD)",
        "seasonal_month_mean_plus_1_5sd": "Seasonal mean+1.5SD",
    }

    flu_rows = []
    for raw, disp in flu_map.items():
        r = flu_metrics[flu_metrics["strategy"] == raw]
        if not r.empty:
            r = r.iloc[0]
            flu_rows.append((disp,
                             float(r["sensitivity"]),
                             float(r["ppv"]),
                             float(r["false_alarm_rate"])))
    cov_rows = []
    for raw, disp in cov_map.items():
        r = cov_metrics[cov_metrics["threshold_rule"] == raw]
        if not r.empty:
            r = r.iloc[0]
            cov_rows.append((disp,
                             float(r["sensitivity_event_week"]),
                             float(r["precision_ppv_event_strict"]),
                             float(r["false_alarm_rate_event_strict"])))

    # Compose long table for plotting
    labels = []
    sens, ppv, far, color, group = [], [], [], [], []
    for disp, s, p, f in flu_rows:
        labels.append(f"[Flu] {disp}")
        sens.append(s); ppv.append(p); far.append(f)
        color.append("#2A9D8F"); group.append("Influenza")
    for disp, s, p, f in cov_rows:
        labels.append(f"[COVID] {disp}")
        sens.append(s); ppv.append(p); far.append(f)
        color.append("#9B2226"); group.append("COVID-coded")

    y = np.arange(len(labels))
    bar_h = 0.22
    ax.barh(y - bar_h, sens, bar_h, color="#264653",
            label="Sensitivity", alpha=0.95)
    ax.barh(y,         ppv,  bar_h, color="#2A9D8F",
            label="PPV", alpha=0.95)
    ax.barh(y + bar_h, far,  bar_h, color="#E76F51",
            label="False-alarm rate", alpha=0.95)

    # Numeric annotations
    for i, (s, p, f) in enumerate(zip(sens, ppv, far)):
        for j, v in zip([-bar_h, 0, bar_h], [s, p, f]):
            ax.text(v + 0.012, i + j, f"{v:.2f}",
                    va="center", ha="left", fontsize=7.2, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Metric value")
    ax.set_title("D. Strategy performance: Influenza endpoint vs. COVID-coded "
                 "endpoint", fontsize=10.5, loc="left")
    ax.axhline(len(flu_rows) - 0.5, color="#999", ls="--", lw=0.8, alpha=0.7)
    ax.text(0.83, len(flu_rows) / 2 - 0.5, "Influenza\nvalidation",
            color="#2A9D8F", fontsize=8.5, ha="right", va="center", alpha=0.75)
    ax.text(0.83, len(flu_rows) + len(cov_rows) / 2 - 0.5,
            "COVID-coded\nendpoint", color="#9B2226", fontsize=8.5,
            ha="right", va="center", alpha=0.75)
    ax.grid(axis="x", color="#CFD8DC", lw=0.4, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)


# ----------------------- main -----------------------

def main() -> int:
    weekly = load_weekly()
    stats = baseline_stats(weekly)
    print("Baseline 2016-2018 resp_score mean/std:",
          f"{stats['resp_mean']:.4f} / {stats['resp_std']:.4f}")

    print("Loading WHU admissions...")
    adm = load_admissions()
    monthly_rate = respiratory_monthly_rate(adm)
    yearly = annual_admissions(adm)
    print(f"  monthly rows: {len(monthly_rate)}; annual rows: {len(yearly)}")

    flu = load_flu_labels()
    flu_metrics, cov_metrics = load_metrics()
    print(f"  flu validation rows={len(flu)}; metrics={len(flu_metrics)}; "
          f"covid metrics={len(cov_metrics)}")

    fig = plt.figure(figsize=(16.5, 12.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, weekly, stats, monthly_rate, flu)
    panel_b(ax_b, weekly, stats)
    panel_c(ax_c, yearly)
    panel_d(ax_d, flu_metrics, cov_metrics)

    targets = [SUBMIT, LGDI_DIR, PKG_FIG]
    PKG_FIG.mkdir(parents=True, exist_ok=True)
    LGDI_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve previous (Plan B 2-panel) version once
    for tdir in [SUBMIT, PKG_FIG]:
        for ext in (".png", ".pdf", ".tif", ".svg"):
            old = tdir / f"Figure3_early_warning{ext}"
            v2_legacy = tdir / f"Figure3_early_warning_planB_v2_legacy{ext}"
            if old.exists() and not v2_legacy.exists():
                shutil.copy2(old, v2_legacy)

    for tdir in targets:
        for ext in (".png", ".pdf", ".tif", ".svg"):
            out = tdir / f"Figure3_early_warning{ext}"
            if ext == ".tif":
                fig.savefig(out, dpi=300, bbox_inches="tight",
                            pil_kwargs={"compression": "tiff_lzw"})
            else:
                fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  wrote {out}")
    plt.close(fig)

    side = {
        "version": "v3_4panels",
        "panels": {
            "A": "Weekly LGDI respiratory residual score with FluNet influenza-season shading and admission-rate bars",
            "B": "Per-group LGDI residual z-score heatmap",
            "C": "WHU annual admissions 2012-2020 (total vs. respiratory)",
            "D": "Strategy performance: Influenza endpoint (FluNet labels) vs. COVID-coded endpoint",
        },
        "baseline_window": "2016-01-01 to 2018-12-31",
        "monitor_window": f"{MONITOR_START.date()} to {MONITOR_END.date()}",
        "thresholds": {
            "respiratory_mean_plus_1_5sd": stats["resp_mean"] + 1.5 * stats["resp_std"],
            "respiratory_mean_plus_2sd": stats["resp_mean"] + 2.0 * stats["resp_std"],
        },
        "annual_admissions": yearly.to_dict(orient="records"),
        "replaces": "Figure3_early_warning_planB_v2_legacy.{png,pdf,tif,svg} (2-panel Plan B)",
    }
    (LGDI_DIR / "Figure3_early_warning_v3.json").write_text(
        json.dumps(side, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
