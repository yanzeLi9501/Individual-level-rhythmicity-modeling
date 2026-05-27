from __future__ import annotations

import ast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rebuild_common import (
    ANALYSIS_OUTPUTS_DIR,
    COLORS,
    FIGURES_DIR,
    GPU_XGB_DIR,
    LGDI_DIR,
    OUTPUTS_DIR,
    SHADING,
    XGB_VISIT_DIR,
    add_note,
    apply_style,
    ensure_dirs,
    num,
    panel_label,
    parse_dates,
    pct,
    read_csv,
    read_json,
    safe_auc,
    save_figure,
    short_label,
)


SELECTED_STRATEGIES = [
    "REFERENCE_resp_mean_plus_1_5sd",
    "consensus_2groups_mean_plus_1_5sd",
    "consensus_2groups_season_oct_apr",
    "season_sustained_consensus2grp_nov_mar",
]

_FLU_SEASONS = [
    ("'15-'16", pd.Timestamp("2015-11-01"), pd.Timestamp("2016-04-30")),
    ("'16-'17", pd.Timestamp("2016-11-01"), pd.Timestamp("2017-04-30")),
    ("'17-'18", pd.Timestamp("2017-11-01"), pd.Timestamp("2018-04-30")),
    ("'18-'19", pd.Timestamp("2018-11-01"), pd.Timestamp("2019-04-30")),
    ("'19-'20", pd.Timestamp("2019-11-01"), pd.Timestamp("2020-04-30")),
]


def _panel_pregap(ax):
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
    ax.legend(fontsize=7.5, loc="lower right")


def _panel_postgap(ax):
    """Post-Gap cardiac admissions 2020\u20132024 with EHR expansion marker."""
    ax.set_title("Post-Gap cardiac admissions 2020\u20132024\n(EHR system expansion marker)")
    pg_path = OUTPUTS_DIR / "xgb_symmetric" / "cardiac_extended_covid_lgdi_timeline.csv"
    if not pg_path.exists():
        add_note(ax, "cardiac_extended_covid_lgdi_timeline.csv not found.")
        return
    pg = pd.read_csv(pg_path, parse_dates=["week_start"])
    pg["month"] = pg["week_start"].dt.to_period("M").dt.to_timestamp()
    pg_monthly = pg.groupby("month")["n"].sum().reset_index()
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
    ax.legend(fontsize=7.5, loc="lower left")


def load_lgdi_inputs() -> dict[str, pd.DataFrame | dict]:
    return {
        "validation": parse_dates(read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv")),
        "metrics": read_csv(LGDI_DIR / "lgdi_whu_influenza_metrics.csv"),
        "roc": read_csv(LGDI_DIR / "lgdi_whu_influenza_roc.csv"),
        "per_season": read_csv(LGDI_DIR / "lgdi_whu_per_season_performance.csv"),
        "summary": read_json(LGDI_DIR / "lgdi_whu_influenza_summary.json"),
        "lag": read_csv(ANALYSIS_OUTPUTS_DIR / "lgdi_whu_influenza_lag_spearman_corrected.csv"),
    }


def figure2(inputs: dict[str, pd.DataFrame | dict]) -> None:
    validation = inputs["validation"]
    assert isinstance(validation, pd.DataFrame)
    score_cols = [col for col in validation.columns if col.startswith("score_")]
    fig = plt.figure(figsize=(17.5, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.7],
                          width_ratios=[0.75, 1.6],
                          left=0.04, right=0.97, top=0.94, bottom=0.05,
                          hspace=0.28, wspace=0.22)

    # Panel A: top-left
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    ax_a.set_title("WHU monitor weeks and FluNet China positivity")
    ax_a.plot(validation["week_start"], validation["n_admissions"], color=COLORS["gray"], lw=1.0, label="WHU admissions")
    ax_a.set_ylabel("Admissions per week")
    ax_a2 = ax_a.twinx()
    ax_a2.plot(validation["week_start"], validation["positivity"], color=COLORS["green"], lw=1.5, label="FluNet positivity")
    ax_a2.axhline(validation["positivity_threshold"].iloc[0], color=COLORS["green"], lw=0.9, ls="--")
    ax_a2.fill_between(
        validation["week_start"],
        0,
        validation["positivity"],
        where=validation["flu_event_window"].astype(bool),
        color=COLORS["green"],
        alpha=0.18,
        label="Flu event weeks",
    )
    ax_a2.set_ylabel("FluNet positivity")
    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Three-stage split annotation
    _cal_end = pd.Timestamp("2017-07-01")
    _val_end = pd.Timestamp("2019-01-01")
    _data_end = validation["week_start"].max()
    ax_a.axvspan(validation["week_start"].min(), _cal_end, alpha=0.07, color=COLORS["blue"], zorder=0)
    ax_a.axvspan(_cal_end, _val_end, alpha=0.07, color=COLORS["orange"], zorder=0)
    ax_a.axvspan(_val_end, _data_end, alpha=0.07, color=COLORS["red"], zorder=0)
    for _vline in [_cal_end, _val_end]:
        ax_a.axvline(_vline, color=COLORS["gray"], lw=0.7, ls=":", zorder=1)
    ax_a.text(pd.Timestamp("2016-10-01"), 0.97, "Calibration", transform=ax_a.get_xaxis_transform(),
            fontsize=7.5, color=COLORS["blue"], ha="center", va="top", fontweight="bold")
    ax_a.text(pd.Timestamp("2018-04-01"), 0.97, "Validation", transform=ax_a.get_xaxis_transform(),
            fontsize=7.5, color="darkorange", ha="center", va="top", fontweight="bold")
    ax_a.text(pd.Timestamp("2019-07-01"), 0.97, "Test", transform=ax_a.get_xaxis_transform(),
            fontsize=7.5, color=COLORS["red"], ha="center", va="top", fontweight="bold")

    # Panel B: right column, spans rows 0-1 (ENLARGED 3D surface)
    ax_b = fig.add_subplot(gs[0:2, 1], projection="3d", facecolor="white")
    ax_b.text2D(-0.10, 1.04, "B", transform=ax_b.transAxes,
                 fontsize=12, fontweight="bold", color=COLORS["dark"])
    ax_b.set_title("Multi-group residual activation terrain", fontsize=10, pad=14)
    _group_names_2b = [col.replace("score_", "") for col in score_cols]
    Z_2b = validation[score_cols].T.to_numpy(dtype=float)
    _n_groups_2b, _n_weeks_2b = Z_2b.shape
    _X_time_2b = np.arange(_n_weeks_2b)
    _Y_groups_2b = np.arange(_n_groups_2b)
    X_2b, Y_2b = np.meshgrid(_X_time_2b, _Y_groups_2b)
    _z_max_2b = np.nanmax(np.abs(Z_2b))
    surf_2b = ax_b.plot_surface(
        X_2b, Y_2b, Z_2b,
        cmap="RdBu_r",
        vmin=-_z_max_2b, vmax=_z_max_2b,
        edgecolor="none", alpha=0.92,
        antialiased=True, rstride=1, cstride=1,
    )
    _event_mask_2b = validation["flu_event_window"].astype(bool).values
    _event_idx_2b = np.where(_event_mask_2b)[0]
    for _ei in _event_idx_2b[::3]:
        _verts_2b = [
            [(_ei, 0, -_z_max_2b), (_ei, _n_groups_2b - 1, -_z_max_2b),
             (_ei, _n_groups_2b - 1, _z_max_2b), (_ei, 0, _z_max_2b)],
        ]
        ax_b.add_collection3d(
            Poly3DCollection(_verts_2b, facecolor=COLORS["green"],
                             alpha=0.06, edgecolor="none", zorder=1),
        )
    ax_b.set_xlabel("Monitoring week", labelpad=8, fontsize=8)
    ax_b.set_ylabel("Disease group", labelpad=8, fontsize=8)
    ax_b.set_zlabel("Residual score", labelpad=6, fontsize=8)
    ax_b.set_yticks(_Y_groups_2b)
    ax_b.set_yticklabels([g[:8] for g in _group_names_2b], fontsize=7)
    _wk_starts_2b = validation["week_start"]
    _yr_ticks_2b, _yr_labels_2b = [], []
    for _yi in range(2016, 2021):
        _mask_y = _wk_starts_2b.dt.year == _yi
        if _mask_y.any():
            _yr_ticks_2b.append(int(np.where(_mask_y)[0][0]))
            _yr_labels_2b.append(str(_yi))
    ax_b.set_xticks(_yr_ticks_2b)
    ax_b.set_xticklabels(_yr_labels_2b, fontsize=7)
    ax_b.view_init(elev=22, azim=-55)
    fig.colorbar(surf_2b, ax=ax_b, shrink=0.40, aspect=14, pad=0.08, label="Residual score")
    ax_b.text2D(0.02, 0.96,
                f"Green planes = FluNet event weeks\n"
                f"Cerebrovascular peak = {Z_2b[0].max():.3f}  |  "
                f"Respiratory peak = {Z_2b[5].max():.3f}",
                transform=ax_b.transAxes, fontsize=7,
                color=COLORS["dark"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82,
                          ec=COLORS["gray"], lw=0.7))

    # Panel C: middle-left
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    ax_c.set_title("Respiratory residual score and LGDI scalar")
    ax_c.plot(validation["week_start"], validation["resp_score"], color=COLORS["blue"], lw=1.4, label="resp_score")
    ax_c.plot(validation["week_start"], validation["lgdi"], color=COLORS["purple"], lw=1.1, label="LGDI scalar")
    ax_c.axhline(validation["resp_score"].mean(), color=COLORS["gray"], lw=0.8, ls=":")
    ax_c.fill_between(
        validation["week_start"],
        validation["resp_score"].min(),
        validation["resp_score"].max(),
        where=validation["flu_event_window"].astype(bool),
        color=COLORS["green"],
        alpha=0.12,
    )
    ax_c.set_ylabel("Score")
    ax_c.legend(loc="upper right")
    ax_c.xaxis.set_major_locator(mdates.YearLocator())
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel D+E: bottom row — D narrower, spacer between D and E
    sub_gs_bottom = gs[2, :].subgridspec(1, 3, width_ratios=[0.45, 0.08, 1.15])
    ax_d = fig.add_subplot(sub_gs_bottom[0, 0])
    panel_label(ax_d, "D")
    ax_d.set_title("FluNet event-week density by analysis segment", pad=0, fontsize=9.5)
    _segs = [
        ("Calibration\n2016\u20132017", validation["week_start"] < _cal_end),
        ("Validation\n2017\u20132019",
         (validation["week_start"] >= _cal_end) & (validation["week_start"] < _val_end)),
        ("Quasi-test\n2019\u20132020", validation["week_start"] >= _val_end),
    ]
    _labels_d, _n_total_d, _n_event_d, _resp_max_d = [], [], [], []
    for _lbl, _mask in _segs:
        _seg = validation[_mask]
        _labels_d.append(_lbl)
        _n_total_d.append(len(_seg))
        _n_event_d.append(int(_seg["flu_event_window"].sum()))
        _resp_max_d.append(float(_seg["resp_score"].max()))
    _x_d = np.arange(len(_labels_d))
    _n_non_d = [t - e for t, e in zip(_n_total_d, _n_event_d)]
    ax_d.bar(_x_d, _n_non_d, width=0.68, color=COLORS["warm_gray"], alpha=0.55, label="Non-event weeks")
    ax_d.bar(_x_d, _n_event_d, width=0.68, color=COLORS["mint"], alpha=0.85,
             bottom=_n_non_d, label="FluNet event weeks")
    for xi, (ne, n_non, tot) in enumerate(zip(_n_event_d, _n_non_d, _n_total_d)):
        ax_d.text(xi, n_non + ne / 2, f"{ne}/{tot}\nwks", ha="center", va="center",
                  fontsize=8, color="white", fontweight="bold")
    ax_d.set_xticks(_x_d)
    ax_d.set_xticklabels(_labels_d, fontsize=9)
    ax_d.set_ylabel("Number of weeks")
    ax_d.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.15, 0.72))
    _ax_d2 = ax_d.twinx()
    _ax_d2.plot(_x_d, _resp_max_d, marker="D", color=COLORS["coral"], lw=1.5, ms=7,
                label="Max resp_score")
    for xi, rv in enumerate(_resp_max_d):
        _ax_d2.text(xi + 0.06, rv + 0.005, f"{rv:.3f}", fontsize=7.5,
                    color=COLORS["coral"], va="bottom")
    _ax_d2.set_ylabel("Max respiratory residual score", color=COLORS["coral"])
    _ax_d2.tick_params(axis="y", colors=COLORS["coral"])
    _ax_d2.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.15, 0.30))

    # Panel E: bottom-right (same row as D)
    ax_e = fig.add_subplot(sub_gs_bottom[0, 2])
    panel_label(ax_e, "E")
    _panel_pregap(ax_e)

    save_figure(fig, FIGURES_DIR / "Figure2")


def figure3(inputs: dict[str, pd.DataFrame | dict]) -> None:
    metrics = inputs["metrics"]
    roc = inputs["roc"]
    per_season = inputs["per_season"]
    summary = inputs["summary"]
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(roc, pd.DataFrame)
    assert isinstance(per_season, pd.DataFrame)
    assert isinstance(summary, dict)
    selected = metrics[metrics["strategy"].isin(SELECTED_STRATEGIES)].copy()
    selected["label"] = selected["strategy"].map(short_label)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    panel_label(ax, "A")
    ax.set_title("S1-S4 operating characteristics")
    x = np.arange(len(selected))
    width = 0.25
    for offset, metric, color in [(-width, "sensitivity", COLORS["blue"]), (0, "ppv", COLORS["orange"]), (width, "false_alarm_rate", COLORS["gray"] )]:
        ax.bar(x + offset, selected[metric], width=width, label=metric.replace("_", " "), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(selected["label"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("Season-level event burden and respiratory-score peaks")
    ax.bar(per_season["season_id"], per_season["n_event_weeks"], color=COLORS["green"], alpha=0.75, label="Event weeks")
    ax.set_ylabel("Event weeks")
    ax.tick_params(axis="x", rotation=25)
    # Annotate null season (2016-2017: 0 FluNet event weeks, embedded specificity control)
    _null = per_season[per_season["n_event_weeks"] == 0]
    for _, _row in _null.iterrows():
        ax.text(_row["season_id"], 0.04, "null season\n(specificity\ncontrol)",
                ha="center", va="bottom", fontsize=6, color=COLORS["dark"],
                transform=ax.get_xaxis_transform())
    ax2 = ax.twinx()
    ax2.plot(per_season["season_id"], per_season["resp_score_max"], marker="o", color=COLORS["blue"], label="Max resp_score")
    ax2.set_ylabel("Max resp_score")
    # Pandemic onset vertical marker (between 2018-2019 and 2019-2020, categorical pos 3.5)
    _n_seasons = len(per_season)  # 5 seasons → positions 0–4
    _pandemic_x = _n_seasons - 1.5  # 3.5, between last two bars
    ax2.axvline(x=_pandemic_x, color="#cccccc", ls="--", lw=1.1, zorder=4)
    ax2.text(_pandemic_x - 0.05, 0.72,
             "Pandemic onset\n(not used as reference standard)",
             fontsize=6, color=COLORS["dark"], va="top", ha="right",
             transform=ax2.get_xaxis_transform())

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("ROC curves for FluNet event-week classification")
    for score_type, curve in roc.groupby("score_type"):
        auc = safe_auc(curve)
        color = COLORS["blue"] if score_type == "resp_score" else COLORS["purple"] if score_type == "lgdi" else COLORS["gray"]
        ax.plot(curve["fpr"], curve["tpr"], lw=1.5, color=color, label=f"{short_label(score_type)} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], lw=0.8, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_title("Leading chronic-disease group during FluNet event weeks")
    _val_3d = inputs["validation"]
    _score_cols_3d = [c for c in _val_3d.columns if c.startswith("score_")]
    _events_3d = _val_3d[_val_3d["flu_event_window"].astype(bool)].copy()
    if _events_3d.empty or not _score_cols_3d:
        add_note(ax, "No FluNet event weeks found in validation data.")
    else:
        _ranks_3d = _events_3d[_score_cols_3d].rank(axis=1, ascending=False, method="min")
        _n_top_3d = (_ranks_3d == 1).sum()
        _n_total_3d = len(_events_3d)
        _dom = pd.DataFrame({
            "group": [c.replace("score_", "") for c in _score_cols_3d],
            "pct_rank1": (_n_top_3d / _n_total_3d * 100).values,
        })
        _dom = _dom.sort_values("pct_rank1", ascending=True).reset_index(drop=True)
        _palette_3d = {"Cerebrovascular": COLORS["coral"], "Respiratory": COLORS["green"]}
        _colors_3d = [_palette_3d.get(g, COLORS["blue"]) for g in _dom["group"]]
        _y_pos_3d = np.arange(len(_dom))
        _bars_3d = ax.barh(_y_pos_3d, _dom["pct_rank1"].values,
                           color=_colors_3d, alpha=0.85, height=0.7)
        ax.set_yticks(_y_pos_3d)
        ax.set_yticklabels(_dom["group"].values, fontsize=9)
        ax.set_xlabel("% of FluNet event weeks ranked #1")
        for _bar_3d, _pct_v in zip(_bars_3d, _dom["pct_rank1"].values):
            ax.text(_pct_v + 0.4, _bar_3d.get_y() + _bar_3d.get_height() / 2,
                    f"{_pct_v:.1f}%", va="center", ha="left", fontsize=8)
        _top_idx_3d = int(_dom["pct_rank1"].idxmax())
        _top_grp_3d = _dom.loc[_top_idx_3d, "group"]
        _top_v_3d = _dom.loc[_top_idx_3d, "pct_rank1"]
        # Place annotation inside the wide Cerebrovascular bar to avoid overlapping others
        ax.annotate(
            f"{_top_grp_3d} leads:\n{_top_v_3d:.0f}% of event weeks",
            xy=(_top_v_3d * 0.92, float(_top_idx_3d)),
            xytext=(_top_v_3d * 0.38, float(_top_idx_3d)),
            fontsize=8, color=COLORS["coral"],
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=COLORS["coral"], lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85,
                      ec=COLORS["coral"], lw=0.8),
        )
        _resp_rows_3d = _dom[_dom["group"] == "Respiratory"]
        if not _resp_rows_3d.empty:
            _ri_3d = int(_resp_rows_3d.index[0])
            _rv_3d = float(_resp_rows_3d["pct_rank1"].values[0])
            ax.text(_rv_3d + 0.4, float(_ri_3d) - 0.42,
                    f"(Respiratory {_rv_3d:.1f}%: not primary driver)",
                    va="top", ha="left", fontsize=7.5, color=COLORS["gray"])
    save_figure(fig, FIGURES_DIR / "Figure3")


def figure4(inputs: dict[str, pd.DataFrame | dict]) -> None:
    lag = inputs["lag"]
    assert isinstance(lag, pd.DataFrame)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    x = lag["lgdi_leads_flunet_weeks"].astype(int)

    ax = axes[0, 0]
    panel_label(ax, "A")
    ax.set_title("Lag-Spearman correlation after multiplicity correction")
    colors = [COLORS["orange"] if sig else COLORS["gray"] for sig in lag["significant_bonferroni_alpha_0_05"].astype(bool)]
    ax.bar(x, lag["spearman_rho"], color=colors)
    ax.set_xlabel("EHR score leads FluNet by weeks")
    ax.set_ylabel("Spearman rho")
    for xpos, rho, sig in zip(x, lag["spearman_rho"], lag["significant_bonferroni_alpha_0_05"]):
        ax.text(xpos, rho + 0.01, "*" if sig else "", ha="center", va="bottom", fontsize=12)

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("Raw p-values, Bonferroni p-values, and BH-FDR q-values")
    ax.plot(x, lag["p_value"], marker="o", color=COLORS["blue"], label="raw p")
    ax.plot(x, lag["p_bonferroni"], marker="o", color=COLORS["red"], label="Bonferroni p")
    ax.plot(x, lag["q_bh_fdr"], marker="o", color=COLORS["purple"], label="BH-FDR q")
    ax.axhline(0.05, color=COLORS["gray"], lw=0.8, ls=":", label="0.05")
    ax.axhline(float(lag["bonferroni_alpha"].iloc[0]), color=COLORS["red"], lw=0.8, ls="--", label="Bonferroni alpha")
    ax.set_yscale("log")
    ax.set_xlabel("Lead weeks")
    ax.set_ylabel("p or q value, log scale")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("Lag-4: LGDI (t−4 weeks) vs FluNet positivity (t)")
    _val = inputs["validation"].copy()
    _lgdi_lagged = _val["lgdi"].shift(4)
    _pos = _val["positivity"]
    _mask = _lgdi_lagged.notna() & _pos.notna()
    _x = _lgdi_lagged[_mask].values.astype(float)
    _y = _pos[_mask].values.astype(float)
    ax.scatter(_x, _y, color=COLORS["blue"], alpha=0.50, s=22, linewidths=0)
    if len(_x) > 2:
        _coef = np.polyfit(_x, _y, 1)
        _xline = np.linspace(_x.min(), _x.max(), 80)
        ax.plot(_xline, np.polyval(_coef, _xline), color=COLORS["red"], lw=1.4, ls="--")
    ax.set_xlabel("LGDI score (t−4 weeks)")
    ax.set_ylabel("FluNet positivity (t)")
    _lag4_row = lag[lag["lgdi_leads_flunet_weeks"] == 4].iloc[0]
    ax.annotate(
        f"ρ = {_lag4_row['spearman_rho']:.3f}\nBonferroni p = {_lag4_row['p_bonferroni']:.4f} ★",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85, edgecolor=COLORS["gray"]),
    )

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_title("FluNet subtype lag-Spearman (LGDI leads FluNet)")
    _subtype_path = LGDI_DIR / "lgdi_whu_flunet_subtype_lag.csv"
    if _subtype_path.exists():
        _sub = read_csv(_subtype_path)
        _palette_sub = {
            "AH3": COLORS["blue"],
            "AH1N12009": COLORS["green"],
            "INF_B": COLORS["orange"],
            "BVIC": COLORS["purple"],
            "BYAM": COLORS["amber"],
        }
        _label_map_sub = {
            "AH3": "A/H3N2", "AH1N12009": "A/H1N1pdm09",
            "INF_B": "B (combined)", "BVIC": "B/Vic", "BYAM": "B/Yam",
        }
        for _subtype, _grp in _sub.groupby("subtype"):
            _clr_sub = _palette_sub.get(str(_subtype), COLORS["gray"])
            _lbl_sub = _label_map_sub.get(str(_subtype), str(_subtype))
            ax.plot(_grp["lag_weeks"], _grp["spearman_rho"],
                    marker="o", ms=4, color=_clr_sub, lw=1.2, label=_lbl_sub)
        ax.axhline(0, color=COLORS["gray"], lw=0.7, ls="--")
        ax.set_xlabel("Lead weeks (LGDI leads FluNet)")
        ax.set_ylabel("Spearman \u03c1")
        ax.set_xticks(sorted(_sub["lag_weeks"].unique()))
        ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    else:
        add_note(ax, "lgdi_whu_flunet_subtype_lag.csv not found.\n"
                     "FluNet subtype lag analysis not available.")
    save_figure(fig, FIGURES_DIR / "Figure4")


def figure5() -> None:
    xgb = read_csv(XGB_VISIT_DIR / "visit_order_first_version_audit.csv")
    lgdi_audit = read_csv(LGDI_DIR / "lgdi_whu_model_audit.csv")
    representative = xgb[xgb["experiment"].str.contains("vo5_cap30")].copy()
    selective = xgb[xgb["experiment"].str.contains("vo20_cap10")].copy()
    selected = pd.concat([representative.head(2), selective.head(3)], ignore_index=True)
    selected["label"] = selected.apply(
        lambda row: f"VO>={int(row['min_visit_order'])}, cap {int(row['gap_cap'])}d\n{row['split_mode']} split, n={int(row['n'])}",
        axis=1,
    )

    # Load feature expansion data for Panel B
    _feat_exp_path = (
        OUTPUTS_DIR / "xgb_symmetric" / "feature_expansion" / "mimic_feature_expansion_results.json"
    )
    _feat_exp = read_json(_feat_exp_path) if _feat_exp_path.exists() else {}

    fig = plt.figure(figsize=(14.5, 10), constrained_layout=True)
    gs5 = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.55],
                           width_ratios=[1.15, 0.85],
                           left=0.10, right=0.96, top=0.94, bottom=0.06,
                           hspace=0.10, wspace=0.12)
    axes = np.empty((2, 2), dtype=object)
    for _i in range(2):
        for _j in range(2):
            axes[_i, _j] = fig.add_subplot(gs5[_i, _j])
    ax = axes[0, 0]
    panel_label(ax, "A")
    ax.set_title("WHU individual XGBoost: representative vs selective settings")
    ax.barh(selected["label"], selected["r2_mean"], xerr=selected["r2_std"],
            color=[COLORS["blue"]] * 2 + [COLORS["orange"]] * 3, height=0.65)
    ax.set_xlabel("Cross-validated R2")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.axvline(0.541, color=COLORS["blue"], lw=0.9, ls="--", label="Representative target")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("MIMIC-IV feature expansion: 12 → 121 features (ΔR²≈0.010)")
    if _feat_exp:
        _sparse_r2 = _feat_exp.get("sparse_los_r2", 0.0)
        _dense_r2 = _feat_exp.get("dense_los_r2", 0.0)
        _delta = _feat_exp.get("delta_los_r2", 0.0)
        _n_sparse = _feat_exp.get("sparse_n_features", 12)
        _n_dense = _feat_exp.get("dense_n_features", 121)
        _bar_labels = [f"Sparse\n({_n_sparse} features)", f"Dense\n({_n_dense} features)"]
        _bar_vals = [_sparse_r2, _dense_r2]
        bars = ax.bar(_bar_labels, _bar_vals, color=[COLORS["blue"], COLORS["orange"]], width=0.45)
        # Annotate values on bars
        for _bar, _val in zip(bars, _bar_vals):
            ax.text(_bar.get_x() + _bar.get_width() / 2, _val + 0.001,
                    f"R²={_val:.3f}", ha="center", va="bottom", fontsize=8.5)
        # Ceiling bracket annotation showing ΔR²
        _bracket_y = _dense_r2 + 0.006
        ax.annotate(
            "", xy=(1.0, _bracket_y), xytext=(0.0, _bracket_y),
            arrowprops=dict(arrowstyle="<->", color=COLORS["gray"], lw=1.0),
            xycoords=("data", "data"),
        )
        ax.text(0.5, _bracket_y + 0.002, f"ΔR²={_delta:.3f}\n(ceiling gain, near-zero improvement)",
                ha="center", va="bottom", fontsize=8, color=COLORS["gray"])
        ax.set_ylim(0, _dense_r2 + 0.025)
        ax.axhline(_dense_r2, color=COLORS["orange"], lw=0.8, ls="--", label=f"Performance ceiling R²={_dense_r2:.3f}")
        ax.legend(loc="lower right", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "Feature expansion data not found", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Cross-validated R² (next LOS)")
    ax.set_xlabel("Feature set")

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("Prediction error metrics across settings")
    x = np.arange(len(selected))
    ax.bar(x - 0.18, selected["mae_mean"], width=0.36, color=COLORS["green"], label="MAE")
    ax.bar(x + 0.18, selected["rmse_mean"], width=0.36, color=COLORS["purple"], label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels([label.split("\n")[0] for label in selected["label"]],
                        rotation=22, ha="right", fontsize=7)
    ax.set_ylabel("Days")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_title("LGDI residual-module audit is exploratory")
    ax.bar(lgdi_audit["target"], lgdi_audit["cv_r2"], color=[COLORS["blue"], COLORS["orange"]])
    ax.set_ylabel("GroupKFold CV R2")
    for idx, row in lgdi_audit.iterrows():
        ax.text(idx, row["cv_r2"] + 0.01, f"MAE={row['cv_mae']:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(0.22, lgdi_audit["cv_r2"].max() + 0.06))

    # Panel E: Post-Gap cardiac admissions (full width)
    ax_post = fig.add_subplot(gs5[2, :])
    panel_label(ax_post, "E")
    _panel_postgap(ax_post)

    save_figure(fig, FIGURES_DIR / "Figure5")


def top_features(dataset: str, top_n: int = 5) -> pd.DataFrame:
    path = GPU_XGB_DIR / dataset / f"{dataset}_feature_importance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "gain"])
    frame = read_csv(path)
    if "gain" not in frame.columns:
        value_cols = [col for col in frame.columns if col != "feature"]
        frame = frame.rename(columns={value_cols[0]: "gain"}) if value_cols else frame.assign(gain=0.0)
    return frame.sort_values("gain", ascending=False).head(top_n)


def figure6() -> None:
    summary = read_csv(GPU_XGB_DIR / "native_xgboost_public_summary.csv")
    summary["label"] = summary["dataset"].map(short_label)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    panel_label(ax, "A", x=-0.06)
    ax.set_title("Native-feature XGBoost discrimination in public ICU datasets")
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["roc_auc"], width=0.36, color=COLORS["blue"], label="ROC-AUC")
    ax.bar(x + 0.18, summary["average_precision"], width=0.36, color=COLORS["orange"], label="Average precision")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[0, 1]
    panel_label(ax, "B", x=-0.06)
    ax.set_title("Thresholded positive-control performance")
    ax.bar(x - 0.18, summary["ppv_precision"], width=0.36, color=COLORS["green"], label="PPV")
    ax.bar(x + 0.18, summary["sensitivity_recall"], width=0.36, color=COLORS["purple"], label="Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric")
    ax.legend(loc="upper right", fontsize=7.5)
    # Cohen's d annotation for NWICU (zero-effect: COVID+ vs COVID- period, d = -0.03)
    _nwicu_idx = summary.index[summary["dataset"] == "nwicu_native_features"]
    if len(_nwicu_idx) > 0:
        _ni = int(_nwicu_idx[0])
        _nwicu_x = x[_ni]
        _nwicu_y = float(summary.loc[_ni, "sensitivity_recall"])
        ax.annotate(
            "Cohen's d = −0.03\n(negligible effect, p≈0.50)",
            xy=(_nwicu_x + 0.18, _nwicu_y + 0.02),
            xytext=(_nwicu_x + 0.60, 0.88),
            fontsize=7, color=COLORS["gray"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=0.7),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                      alpha=0.80, edgecolor=COLORS["gray"], lw=0.6),
        )

    ax = axes[1, 0]
    panel_label(ax, "C", x=-0.06)
    ax.set_title("Dataset size and endpoint prevalence")
    ax.bar(x - 0.18, summary["n_rows"], width=0.36, color=COLORS["gray"], label="Rows")
    ax.bar(x + 0.18, summary["positive_count"], width=0.36, color=COLORS["red"], label="Positive labels")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"])
    ax.set_ylabel("Count, log scale")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    panel_label(ax, "D", x=-0.06)
    ax.set_title("Top 5 native features by gain (per dataset)")
    _ds_list = list(summary["dataset"])
    _ds_colors = {
        "mimic_iv_native_features": COLORS["blue"],
        "eicu_native_features": COLORS["orange"],
        "nwicu_native_features": COLORS["purple"],
    }
    _feat_rows: list[dict] = []
    for _ds in _ds_list:
        _feats = top_features(_ds, top_n=5)
        for _, _frow in _feats.iterrows():
            _feat_rows.append({"feat": _frow["feature"], "gain": float(_frow["gain"]), "ds": _ds})
    _feat_df = pd.DataFrame(_feat_rows)
    if not _feat_df.empty:
        # Reverse so first dataset appears at top of barh
        _feat_df = _feat_df.iloc[::-1].reset_index(drop=True)
        _yticks = np.arange(len(_feat_df))
        _bar_colors = [_ds_colors.get(_ds, COLORS["gray"]) for _ds in _feat_df["ds"]]
        _y_labels = _feat_df["feat"].str.replace("_", " ").str[:22]
        _max_gain = float(_feat_df["gain"].max())
        ax.barh(_yticks, _feat_df["gain"].values, color=_bar_colors, height=0.70)
        ax.set_yticks(_yticks)
        ax.set_yticklabels(_y_labels, fontsize=7.0)
        ax.set_xlabel("Feature gain (within-dataset, not cross-comparable)")
        ax.set_xlim(0, _max_gain * 1.55)
        for _yi, _val in zip(_yticks, _feat_df["gain"].values):
            ax.text(_val + _max_gain * 0.015, _yi, f"{_val:.3f}", va="center", fontsize=6.2)
        # Dividers between dataset groups
        _group_sizes = [len(top_features(_ds, top_n=5)) for _ds in _ds_list]
        _sep = 0
        for _gs in _group_sizes[1:]:
            _sep += _gs
            ax.axhline(_sep - 0.5, color=COLORS["gray"], lw=0.7, ls="--", alpha=0.6)
        # Legend
        from matplotlib.patches import Patch as _Patch
        _legend_elems = [
            _Patch(facecolor=_ds_colors.get(_ds, COLORS["gray"]), label=short_label(_ds))
            for _ds in _ds_list
        ]
        ax.legend(handles=_legend_elems, loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Feature importance data not found", ha="center", va="center", transform=ax.transAxes)
    save_figure(fig, FIGURES_DIR / "Figure6")


def main() -> None:
    ensure_dirs()
    apply_style()
    inputs = load_lgdi_inputs()
    figure2(inputs)
    figure3(inputs)
    figure4(inputs)
    figure5()
    figure6()


if __name__ == "__main__":
    main()