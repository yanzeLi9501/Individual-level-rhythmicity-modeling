"""build_merged_figures.py — Steps 2-3 of merge_figures.md

Generates:
  FigureS3_merged  (P0)  Historical reference window + 3-segment diagram + FluNet inset
  FigureS6_merged  (P0)  Seasonal S1-S4 metrics + per-group dominance ranking
  FigureS2_merged  (P1)  Cardiac Pre-Gap monthly admissions
  FigureS9_merged  (P1)  XGBoost model selection + feature-expansion analysis

All figures use the §1.5 spring/summer palette from rebuild_common.py
and are saved to [RR]/outputs/figures/merged/ as png/pdf/svg/tif.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_common import (
    ANALYSIS_OUTPUTS_DIR,
    COLORS,
    LGDI_DIR,
    MERGED_FIGURES_DIR,
    OUTPUTS_DIR,
    SHADING,
    XGB_VISIT_DIR,
    add_note,
    apply_style,
    panel_label,
    parse_dates,
    read_csv,
    read_json,
    safe_auc,
    save_merged,
    short_label,
)

XGB_SYM_DIR = OUTPUTS_DIR / "xgb_symmetric"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_validation() -> pd.DataFrame:
    return parse_dates(read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv"))


def _load_rolling() -> pd.DataFrame:
    df = read_csv(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv")
    df["window_anchor"] = pd.to_datetime(df["window_anchor"])
    df["window_start"]  = pd.to_datetime(df["window_start"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FigureS3_merged  (P0)
# Panel A — Historical LGDI surveillance time series
# Panel B — Three-segment split schematic
# Panel C — FluNet subtype lag-Spearman inset
# ─────────────────────────────────────────────────────────────────────────────

def figure_s3_merged() -> None:
    apply_style()
    val   = _load_validation()
    roll  = _load_rolling()

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])

    ax_a  = fig.add_subplot(gs[0, :])     # Panel A — full width
    ax_b  = fig.add_subplot(gs[1, 0])     # Panel B — 3-segment diagram
    ax_c  = fig.add_subplot(gs[1, 1])     # Panel C — FluNet subtype inset

    # ── Panel A ───────────────────────────────────────────────────────────────
    panel_label(ax_a, "A")
    ax_a.set_title(
        "Historical influenza-season reference vector construction "
        "and pre-specified window validation",
        fontsize=10,
    )

    # Season-window background shading (cream)
    _cal_end = pd.Timestamp("2017-07-01")
    _val_end = pd.Timestamp("2019-01-01")
    _t0      = val["week_start"].min()
    _t1      = val["week_start"].max()
    ax_a.axvspan(_t0,      _cal_end, color=SHADING["blue"][0],   alpha=SHADING["blue"][1],   zorder=0, label="Calibration 2015-2017")
    ax_a.axvspan(_cal_end, _val_end, color=SHADING["orange"][0], alpha=SHADING["orange"][1], zorder=0, label="Validation 2017-2019")
    ax_a.axvspan(_val_end, _t1,      color=SHADING["red"][0],    alpha=SHADING["red"][1],    zorder=0, label="Quasi-test 2019-2020")

    # FluNet event weeks (darker cream overlay)
    ax_a.fill_between(
        val["week_start"], 0, 1, where=val["flu_event_window"].astype(bool),
        transform=ax_a.get_xaxis_transform(),
        color=SHADING["mint"][0], alpha=0.28, zorder=1,
    )

    # resp_score primary axis
    ax_a.plot(val["week_start"], val["resp_score"],
              color=COLORS["blue"], lw=1.6, label="Respiratory residual score")
    ax_a.set_ylabel("Respiratory residual score")

    # LGDI scalar twin axis
    ax_a2 = ax_a.twinx()
    ax_a2.plot(val["week_start"], val["lgdi"],
               color=COLORS["teal"], lw=1.1, ls="--", alpha=0.9, label="LGDI scalar")
    ax_a2.set_ylabel("LGDI scalar")

    # FluNet positivity (third axis via ax_a; rescale for readability)
    pos_max = val["positivity"].max()
    resp_max = val["resp_score"].max() or 1.0
    ax_a.plot(val["week_start"],
              val["positivity"] * (resp_max / max(pos_max, 1e-6)),
              color=COLORS["green"], lw=1.0, alpha=0.75, ls=":",
              label="FluNet positivity (rescaled)")

    # Stage labels
    _stage_kw = dict(transform=ax_a.get_xaxis_transform(), fontsize=8, va="top")
    ax_a.text(pd.Timestamp("2016-08-01"), 0.97, "Calibration",
              color=COLORS["blue"], ha="center", fontweight="bold", **_stage_kw)
    ax_a.text(pd.Timestamp("2018-04-01"), 0.97, "Validation",
              color=COLORS["orange"], ha="center", fontweight="bold", **_stage_kw)
    ax_a.text(pd.Timestamp("2019-07-01"), 0.97, "Quasi-test",
              color=COLORS["red"], ha="center", fontweight="bold", **_stage_kw)

    for _vline in [_cal_end, _val_end]:
        ax_a.axvline(_vline, color=COLORS["gray"], lw=0.8, ls=":", zorder=2)

    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_a.set_xlabel("Calendar year")

    # Combined legend
    lines_a, labels_a = ax_a.get_legend_handles_labels()
    lines_a2, labels_a2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(lines_a + lines_a2, labels_a + labels_a2,
                ncol=4, fontsize=7.5, loc="upper left")

    # ── Panel B — stacked bar: event vs non-event weeks per segment ──────────
    panel_label(ax_b, "B")
    ax_b.set_title(
        "FluNet event-week density and resp_score peak\nby pre-specified analysis segment",
        fontsize=9.5,
    )

    segs_def = [
        ("Calibration\n2016-2017", val["week_start"] < _cal_end,  COLORS["blue"]),
        ("Validation\n2017-2019",  (val["week_start"] >= _cal_end) & (val["week_start"] < _val_end), COLORS["orange"]),
        ("Quasi-test\n2019-2020", val["week_start"] >= _val_end,  COLORS["red"]),
    ]
    labels_b, n_total_b, n_event_b, resp_max_b = [], [], [], []
    for label, mask, _ in segs_def:
        seg = val[mask]
        labels_b.append(label)
        n_total_b.append(len(seg))
        n_event_b.append(int(seg["flu_event_window"].sum()))
        resp_max_b.append(float(seg["resp_score"].max()))

    x_b = np.arange(len(labels_b))
    n_non = [t - e for t, e in zip(n_total_b, n_event_b)]
    ax_b.bar(x_b, n_non,    width=0.55, color=COLORS["warm_gray"], alpha=0.55,
             label="Non-event weeks")
    ax_b.bar(x_b, n_event_b, width=0.55, color=COLORS["mint"], alpha=0.85,
             bottom=n_non, label="FluNet event weeks")
    for xi, (ne, tot) in enumerate(zip(n_event_b, n_total_b)):
        ax_b.text(xi, tot + 0.8, f"{ne}/{tot}\nwks", ha="center", va="bottom",
                  fontsize=8, color=COLORS["dark"])
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, fontsize=9)
    ax_b.set_ylabel("Number of weeks")
    ax_b.legend(fontsize=8, loc="upper left")

    ax_b2 = ax_b.twinx()
    ax_b2.plot(x_b, resp_max_b, marker="D", color=COLORS["coral"], lw=1.5, ms=7,
               label="Max resp_score")
    for xi, rv in enumerate(resp_max_b):
        ax_b2.text(xi + 0.06, rv + 0.005, f"{rv:.3f}", fontsize=7.5,
                   color=COLORS["coral"], va="bottom")
    ax_b2.set_ylabel("Max respiratory residual score", color=COLORS["coral"])
    ax_b2.tick_params(axis="y", colors=COLORS["coral"])
    ax_b2.legend(loc="upper right", fontsize=8)

    # ── Panel C ───────────────────────────────────────────────────────────────
    panel_label(ax_c, "C")
    ax_c.set_title("FluNet China vs global — subtype lag-Spearman (2009-2024)")

    subtype_path = LGDI_DIR / "lgdi_whu_flunet_subtype_lag.csv"
    if subtype_path.exists():
        sub = read_csv(subtype_path)
        palette_sub = {"AH3": COLORS["blue"], "AH1": COLORS["green"],
                       "BVIC": COLORS["purple"], "BYAM": COLORS["orange"]}
        for subtype, grp in sub.groupby("subtype"):
            color = palette_sub.get(subtype, COLORS["gray"])
            ax_c.plot(grp["lag_weeks"], grp["spearman_rho"],
                      marker="o", ms=4, color=color, lw=1.2,
                      label=subtype.replace("AH3", "A/H3N2")
                                   .replace("AH1", "A/H1N1pdm09")
                                   .replace("BVIC", "B/Vic")
                                   .replace("BYAM", "B/Yam"))
        ax_c.axhline(0, color=COLORS["gray"], lw=0.7, ls="--")
        ax_c.set_xlabel("Lead weeks (LGDI leads FluNet)")
        ax_c.set_ylabel("Spearman ρ")
        ax_c.set_title("FluNet subtype lag-Spearman inset (2016-2020)", fontsize=9)
        ax_c.legend(fontsize=7.5, ncol=2)
    else:
        add_note(ax_c, "lgdi_whu_flunet_subtype_lag.csv not found.")

    save_merged(fig, "FigureS3_merged")
    print("  [ok] FigureS3_merged saved")


# ─────────────────────────────────────────────────────────────────────────────
# FigureS6_merged  (P0)
# Panel A — S1-S4 strategy metrics (global + per-season)
# Panel B — Per-group dominance ranking during FluNet event weeks
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_SHORT = {
    "REFERENCE_resp_mean_plus_1_5sd":         "S1 Resp +1.5SD",
    "REFERENCE_resp_mean_plus_2sd":            "S1 Resp +2SD",
    "LGDI_mean_plus_1_5sd":                   "LGDI scalar",
    "consensus_2groups_mean_plus_1_5sd":       "S2 Consensus ≥2",
    "consensus_2groups_season_oct_apr":        "S3 Season Oct-Apr",
    "season_sustained_consensus2grp_nov_mar":  "S4 Sustained Nov-Mar",
}
_S_COLORS = {
    "S1": COLORS["blue"],
    "S2": COLORS["orange"],
    "S3": COLORS["green"],
    "S4": COLORS["purple"],
}


def _group_dominance(val: pd.DataFrame) -> pd.DataFrame:
    """Return % rank-1 per diagnostic group during FluNet event weeks.

    Uses the validation CSV (lgdi_whu_influenza_validation.csv) which has
    flu_event_window properly populated, NOT the rolling4 CSV where
    event_window is always False.
    """
    score_cols = [c for c in val.columns if c.startswith("score_")]
    events = val[val["flu_event_window"].astype(bool)].copy()
    if events.empty or not score_cols:
        return pd.DataFrame(columns=["group", "pct_rank1"])
    ranks = events[score_cols].rank(axis=1, ascending=False, method="min")
    n_top  = (ranks == 1).sum()
    n_total = len(events)
    result = pd.DataFrame({
        "group":     [c.replace("score_", "") for c in score_cols],
        "pct_rank1": (n_top / n_total * 100).values,
    })
    return result.sort_values("pct_rank1", ascending=True).reset_index(drop=True)


def figure_s6_merged() -> None:
    apply_style()

    metrics_path  = LGDI_DIR / "lgdi_whu_influenza_metrics.csv"
    season_path   = LGDI_DIR / "lgdi_whu_per_season_performance.csv"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    ax_a, ax_b = axes

    # ── Panel A — global S1-S4 metrics ────────────────────────────────────────
    panel_label(ax_a, "A")
    ax_a.set_title(
        "S1-S4 operating characteristics — sensitivity, PPV, FAR",
        fontsize=9.5,
    )

    if metrics_path.exists():
        mdf = read_csv(metrics_path)
        selected = [
            "REFERENCE_resp_mean_plus_1_5sd",
            "consensus_2groups_mean_plus_1_5sd",
            "consensus_2groups_season_oct_apr",
            "season_sustained_consensus2grp_nov_mar",
        ]
        mdf = mdf[mdf["strategy"].isin(selected)].copy()
        mdf["label"] = mdf["strategy"].map(_STRATEGY_SHORT).fillna(mdf["strategy"])

        x = np.arange(len(mdf))
        w = 0.25
        ax_a.bar(x - w,   mdf["sensitivity"],    width=w, color=COLORS["blue"],
                 label="Sensitivity",      zorder=2)
        ax_a.bar(x,       mdf["ppv"],            width=w, color=COLORS["orange"],
                 label="PPV",              zorder=2)
        ax_a.bar(x + w,   mdf["false_alarm_rate"], width=w, color=COLORS["red"],
                 label="False-alarm rate", zorder=2, alpha=0.75)

        for idx, row in mdf.reset_index(drop=True).iterrows():
            for val, offset in [(row["sensitivity"], -w), (row["ppv"], 0),
                                 (row["false_alarm_rate"], w)]:
                ax_a.text(idx + offset, val + 0.012, f"{val:.2f}",
                          ha="center", va="bottom", fontsize=7)

        ax_a.set_xticks(x)
        ax_a.set_xticklabels(mdf["label"], rotation=20, ha="right", fontsize=8)
        ax_a.set_ylim(0, 0.72)
        ax_a.set_ylabel("Proportion")
        ax_a.legend(fontsize=8)
        ax_a.axhline(0.05, color=COLORS["gray"], lw=0.8, ls=":", label="5% reference")

    # Per-season line overlay on twin axis
    if season_path.exists():
        sdf = read_csv(season_path)
        ax_a2 = ax_a.twinx()
        ax_a2.plot(sdf["season_id"], sdf["n_event_weeks"],
                   marker="s", color=COLORS["teal"], lw=1.0, ms=5, ls="--",
                   label="Event weeks / season")
        ax_a2.set_ylabel("Event weeks", color=COLORS["teal"])
        ax_a2.tick_params(axis="y", colors=COLORS["teal"])
        ax_a2.set_ylim(0, sdf["n_event_weeks"].max() * 1.6)
        ax_a2.legend(loc="upper right", fontsize=7.5)

    # ── Panel B — group dominance ──────────────────────────────────────────────
    panel_label(ax_b, "B")
    ax_b.set_title(
        "Multi-system co-elevation: leading chronic-disease groups\n"
        "during FluNet-confirmed event weeks",
        fontsize=9.5,
    )

    val_b = _load_validation()  # use validation CSV with proper flu_event_window
    dom   = _group_dominance(val_b)

    if dom.empty:
        add_note(ax_b, "No FluNet event weeks found in validation data.")
    else:
        palette_b = {
            "Cerebrovascular": COLORS["coral"],
            "Respiratory":     COLORS["green"],
        }
        colors_b = [palette_b.get(g, COLORS["blue"]) for g in dom["group"]]
        # Use numeric y-axis to keep annotation coordinates reliable
        y_pos = np.arange(len(dom))
        bars  = ax_b.barh(y_pos, dom["pct_rank1"].values, color=colors_b, alpha=0.85, height=0.7)
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(dom["group"].values, fontsize=9)
        ax_b.set_xlabel("% of FluNet event weeks ranked #1")

        for bar, pct_val in zip(bars, dom["pct_rank1"].values):
            ax_b.text(
                pct_val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{pct_val:.1f}%",
                va="center", ha="left", fontsize=8,
            )

        # Arrow annotation for the top-ranked group (Cerebrovascular)
        top_idx = int(dom["pct_rank1"].idxmax())
        top_grp = dom.loc[top_idx, "group"]
        top_v   = dom.loc[top_idx, "pct_rank1"]
        ax_b.annotate(
            f"{top_grp} leads:\n{top_v:.0f}% of event weeks",
            xy=(top_v, float(top_idx)),
            xytext=(top_v - 22, float(top_idx) + 0.8),
            fontsize=8, color=COLORS["coral"],
            arrowprops=dict(arrowstyle="->", color=COLORS["coral"], lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85,
                      ec=COLORS["coral"], lw=0.8),
        )

        # Respiratory note — direct label on bar
        resp_rows = dom[dom["group"] == "Respiratory"]
        if not resp_rows.empty:
            ri  = int(resp_rows.index[0])
            rv  = resp_rows["pct_rank1"].values[0]
            ax_b.text(
                rv + 0.4, float(ri) - 0.42,
                f"(Respiratory {rv:.1f}%: not primary driver)",
                va="top", ha="left", fontsize=7.5, color=COLORS["gray"],
            )

    save_merged(fig, "FigureS6_merged")
    print("  [ok] FigureS6_merged saved")


# ─────────────────────────────────────────────────────────────────────────────
# FigureS2_merged  (P1)
# Panel A — Pre-Gap monthly WHU admissions 2016-2019 + flu season shading
# Panel B — Post-Gap data note + EHR expansion context
# ─────────────────────────────────────────────────────────────────────────────

_FLU_SEASONS = [
    ("2015-2016", pd.Timestamp("2015-11-01"), pd.Timestamp("2016-04-30")),
    ("2016-2017", pd.Timestamp("2016-11-01"), pd.Timestamp("2017-04-30")),
    ("2017-2018", pd.Timestamp("2017-11-01"), pd.Timestamp("2018-04-30")),
    ("2018-2019", pd.Timestamp("2018-11-01"), pd.Timestamp("2019-04-30")),
    ("2019-2020", pd.Timestamp("2019-11-01"), pd.Timestamp("2020-04-30")),
]


def figure_s2_merged() -> None:
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    ax_a, ax_b = axes

    # ── Panel A — Pre-Gap monthly admissions ─────────────────────────────────
    panel_label(ax_a, "A")
    ax_a.set_title(
        "Pre-Gap cardiac admissions with influenza-season annotations\n"
        "(WHU monitor period 2016-2020)",
        fontsize=9.5,
    )

    roll_path = LGDI_DIR / "lgdi_whu_rolling4_weekly.csv"
    if roll_path.exists():
        roll = _load_rolling()
        # Aggregate weekly → monthly
        roll["month"] = roll["window_anchor"].dt.to_period("M").dt.to_timestamp()
        monthly = roll.groupby("month")["n_admissions"].sum().reset_index()

        ax_a.bar(monthly["month"], monthly["n_admissions"],
                 width=25, color=COLORS["blue"], alpha=0.80, label="Monthly admissions")

        # Flu season shading (cream)
        for season_id, t_start, t_end in _FLU_SEASONS:
            ax_a.axvspan(t_start, t_end,
                         color=SHADING["cream"][0], alpha=SHADING["cream"][1] + 0.12,
                         zorder=0)
            mid = t_start + (t_end - t_start) / 2
            y_top = monthly["n_admissions"].max() * 0.96 if not monthly.empty else 1000
            ax_a.text(mid, y_top, season_id.replace("20", "'", 1),
                      ha="center", va="top", fontsize=7, color=COLORS["orange"],
                      rotation=0)

        ax_a.xaxis.set_major_locator(mdates.YearLocator())
        ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_a.set_xlabel("Calendar year")
        ax_a.set_ylabel("Monthly admissions (aggregated from weekly 4-wk windows)")
        ax_a.legend(fontsize=8)

        # Gap indicator (data ends 2020-01)
        gap_start = pd.Timestamp("2020-02-01")
        ax_a.axvline(gap_start, color=COLORS["red"], lw=1.2, ls="--")
        ax_a.text(gap_start, monthly["n_admissions"].max() * 0.70,
                  "  WHU data gap →", fontsize=8, color=COLORS["red"], va="bottom")
    else:
        add_note(ax_a, "lgdi_whu_rolling4_weekly.csv not found in lgdi_results/.")

    # ── Panel B — Post-Gap 2020-2024 actual timeline ──────────────────────────
    panel_label(ax_b, "B")
    ax_b.set_title(
        "Post-Gap cardiac admissions 2020-2024\n(EHR system expansion marker)",
        fontsize=9.5,
    )

    pg_path = XGB_SYM_DIR / "cardiac_extended_covid_lgdi_timeline.csv"
    if pg_path.exists():
        pg = pd.read_csv(pg_path, parse_dates=["week_start"])
        # Monthly aggregate
        pg["month"] = pg["week_start"].dt.to_period("M").dt.to_timestamp()
        pg_monthly = pg.groupby("month")["n"].sum().reset_index()

        # Pre-gap mean baseline (from Panel A data for reference line)
        if roll_path.exists():
            _roll_ref = _load_rolling()
            _roll_ref["month"] = _roll_ref["window_anchor"].dt.to_period("M").dt.to_timestamp()
            _pre_monthly = _roll_ref.groupby("month")["n_admissions"].sum()
            pre_mean = float(_pre_monthly.mean())
        else:
            pre_mean = None

        ax_b.bar(pg_monthly["month"], pg_monthly["n"],
                 width=25, color=COLORS["amber"], alpha=0.80, label="Monthly admissions (Post-Gap)")

        # EHR expansion marker (2021-09)
        ehr_exp = pd.Timestamp("2021-09-01")
        ax_b.axvline(ehr_exp, color=COLORS["red"], lw=1.5, ls="--", zorder=3)
        y_top = pg_monthly["n"].max() if not pg_monthly.empty else 1
        ax_b.text(ehr_exp, y_top * 0.96,
                  " EHR expansion\n ×6 monthly admissions",
                  fontsize=8, color=COLORS["red"], va="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8,
                            ec=COLORS["red"], lw=0.8))

        # Pre-gap mean reference line
        if pre_mean is not None:
            ax_b.axhline(pre_mean, color=COLORS["blue"], lw=1.2, ls=":",
                         label=f"Pre-Gap monthly mean ({pre_mean:.0f})")

        ax_b.xaxis.set_major_locator(mdates.YearLocator())
        ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_b.set_xlabel("Calendar year")
        ax_b.set_ylabel("Monthly cardiac admissions (weekly data aggregated)")
        ax_b.legend(fontsize=8)
    else:
        add_note(ax_b, "cardiac_extended_covid_lgdi_timeline.csv not found.\n"
                       "Expected in outputs/xgb_symmetric/.")
        ax_b.set_xlabel("Calendar year")
        ax_b.set_ylabel("Monthly admissions")

    save_merged(fig, "FigureS2_merged")
    print("  [ok] FigureS2_merged saved")


# ─────────────────────────────────────────────────────────────────────────────
# FigureS9_merged  (P1)
# Panel A — XGBoost model selection: CV R² by configuration
# Panel B — R² vs number of features scatter
# Panel C — MAE distribution by split mode
# ─────────────────────────────────────────────────────────────────────────────

def figure_s9_merged() -> None:
    apply_style()

    audit_path = XGB_VISIT_DIR / "visit_order_first_version_audit.csv"
    fe_path    = OUTPUTS_DIR / "xgb_symmetric" / "feature_expansion" / \
                 "mimic_feature_expansion_results.json"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    if not audit_path.exists():
        for ax in axes:
            add_note(ax, f"visit_order_first_version_audit.csv not found at\n{audit_path}")
        save_merged(fig, "FigureS9_merged")
        print("  [ok] FigureS9_merged saved (no audit data)")
        return

    xgb = read_csv(audit_path)
    xgb = xgb.dropna(subset=["r2_mean"]).copy()

    # ── Panel A — R² by experiment (top configs) ────────────────────────────
    panel_label(ax_a, "A")
    ax_a.set_title(
        "Model selection and hyperparameter tuning\nfor WHU XGBoost residual backbone",
        fontsize=9.5,
    )

    # Group by min_visit_order
    grouped = xgb.groupby("min_visit_order")[["r2_mean", "r2_std"]].mean().reset_index()
    color_map = {
        vo: c for vo, c in zip(
            sorted(grouped["min_visit_order"].unique()),
            [COLORS["blue"], COLORS["orange"], COLORS["green"],
             COLORS["purple"], COLORS["teal"]],
        )
    }
    for _, row in grouped.iterrows():
        label = f"min_VO ≥ {int(row['min_visit_order'])}"
        ax_a.bar(str(int(row["min_visit_order"])), row["r2_mean"],
                 yerr=row["r2_std"],
                 color=color_map.get(row["min_visit_order"], COLORS["gray"]),
                 alpha=0.80, label=label, capsize=4, zorder=2)
        ax_a.text(str(int(row["min_visit_order"])), row["r2_mean"] + row["r2_std"] + 0.008,
                  f"{row['r2_mean']:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax_a.set_xlabel("Minimum visit-order threshold")
    ax_a.set_ylabel("Mean cross-validated R²")
    ax_a.set_ylim(0, min(1.0, xgb["r2_mean"].max() * 1.25))
    ax_a.legend(fontsize=7.5)

    # ── Panel B — R² vs n_features ──────────────────────────────────────────
    panel_label(ax_b, "B")
    ax_b.set_title("Feature count vs cross-validated R²\n(all configurations)", fontsize=9.5)

    split_colors = {"patient": COLORS["blue"], "temporal": COLORS["orange"],
                    "site": COLORS["green"]}
    for split, grp in xgb.groupby("split_mode"):
        color = split_colors.get(split, COLORS["gray"])
        ax_b.scatter(grp["n_features"], grp["r2_mean"],
                     c=color, s=35, alpha=0.75, edgecolors="none",
                     label=f"split={split}")
    ax_b.set_xlabel("Number of features")
    ax_b.set_ylabel("CV R²")
    ax_b.legend(fontsize=8)

    # Feature-expansion data (MIMIC)
    if fe_path.exists():
        try:
            fe = read_json(fe_path)
            fe_results = fe.get("results", fe) if isinstance(fe, dict) else []
            if isinstance(fe_results, list) and fe_results:
                fe_df = pd.DataFrame(fe_results)
                if "n_features" in fe_df.columns and "r2_mean" in fe_df.columns:
                    ax_b.scatter(fe_df["n_features"], fe_df["r2_mean"],
                                 c=COLORS["coral"], s=50, marker="D",
                                 alpha=0.85, edgecolors="none",
                                 label="MIMIC feature-expansion")
                    ax_b.legend(fontsize=7.5)
        except Exception:
            pass

    # ── Panel C — MAE distribution by split mode ─────────────────────────────
    panel_label(ax_c, "C")
    ax_c.set_title(
        "WHU XGBoost: CV MAE distribution by split strategy\n"
        "(lower = better prediction of LOS residuals)",
        fontsize=9.5,
    )

    split_modes = xgb["split_mode"].unique()
    positions   = np.arange(len(split_modes))
    w = 0.6

    for idx, split in enumerate(split_modes):
        sub = xgb[xgb["split_mode"] == split]["mae_mean"].dropna()
        color = split_colors.get(split, COLORS["gray"])
        bp = ax_c.boxplot(
            sub, positions=[idx], widths=w,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.5),
            medianprops=dict(color=COLORS["dark"], lw=1.5),
            whiskerprops=dict(color=COLORS["gray"]),
            capprops=dict(color=COLORS["gray"]),
            flierprops=dict(marker="o", color=color, alpha=0.5, ms=4),
        )

    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(
        [f"split=\n{s}" for s in split_modes], fontsize=8
    )
    ax_c.set_ylabel("Mean CV MAE (days)")

    save_merged(fig, "FigureS9_merged")
    print("  [ok] FigureS9_merged saved")


# ─────────────────────────────────────────────────────────────────────────────
# Also regenerate main Figure1-6 via existing scripts (Step 0 completion)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    MERGED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Building merged figures → ", MERGED_FIGURES_DIR)

    print("  Building FigureS3_merged (P0) …")
    figure_s3_merged()

    print("  Building FigureS6_merged (P0) …")
    figure_s6_merged()

    print("  Building FigureS2_merged (P1) …")
    figure_s2_merged()

    print("  Building FigureS9_merged (P1) …")
    figure_s9_merged()

    # Report
    outputs = sorted(MERGED_FIGURES_DIR.glob("*.png"))
    print(f"\nDone. {len(outputs)} PNG files in merged/:")
    for p in outputs:
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
