from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow importing rebuild_common from the scripts directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_common import (
    ANALYSIS_OUTPUTS_DIR,
    COLORS,
    GPU_XGB_DIR,
    LGDI_DIR,
    NC_DIR,
    SUPP_FIGURES_DIR,
    XGB_VISIT_DIR,
    add_note,
    apply_style,
    ensure_dirs,
    panel_label,
    parse_dates,
    read_csv,
    read_json,
    safe_auc,
    save_figure,
    short_label,
)


def load_validation() -> pd.DataFrame:
    return parse_dates(read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv"))


def save_single(fig: plt.Figure, name: str) -> None:
    save_figure(fig, SUPP_FIGURES_DIR / name)


def s1_flow() -> None:
    whu = read_json(LGDI_DIR / "lgdi_whu_summary.json")
    flu = read_json(LGDI_DIR / "lgdi_whu_influenza_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)

    # Panel A: cohort-size bar chart
    ax = axes[0]
    panel_label(ax, "A")
    ax.set_title("WHU cohort size at each pipeline stage")
    stage_labels = ["All admissions\nwith dates", "Unique patient\nrecord numbers", "Baseline\nadmissions\n2016\u20132018"]
    stage_values = [
        int(whu.get("cohort", {}).get("admissions_with_dates", 0)),
        int(whu.get("cohort", {}).get("unique_patient_record_numbers", 0)),
        int(whu.get("baseline", {}).get("n_admissions", 0)),
    ]
    bars = ax.barh(stage_labels, stage_values,
                   color=[COLORS["blue"], COLORS["teal"], COLORS["green"]], alpha=0.85)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    for bar, val in zip(bars, stage_values):
        ax.text(bar.get_width() + max(stage_values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)

    # Panel B: FluNet monitoring-week composition
    ax = axes[1]
    panel_label(ax, "B")
    ax.set_title("FluNet monitoring-week composition")
    n_total = int(flu.get("n_weeks_total", 0))
    n_event = int(flu.get("n_weeks_flu_event", 0))
    n_nonevent = n_total - n_event
    wk_labels = ["FluNet\nevent weeks", "Non-event\nweeks"]
    wk_values = [n_event, n_nonevent]
    bars2 = ax.bar(wk_labels, wk_values,
                   color=[COLORS["orange"], COLORS["gray"]], alpha=0.85)
    ax.set_ylabel("Weeks")
    for bar, val in zip(bars2, wk_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=8)
    ax.text(0.5, 0.02, f"Total: {n_total} monitor weeks",
            transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["warm_gray"])
    save_single(fig, "FigureS1")


def s2_group_scores() -> None:
    frame = load_validation()
    score_cols = [col for col in frame.columns if col.startswith("score_")]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("Weekly residual scores by chronic-disease group")
    for col in score_cols:
        ax.plot(frame["week_start"], frame[col], lw=0.9, label=col.replace("score_", ""))
    ax.axhline(0, color=COLORS["gray"], lw=0.8)
    ax.set_ylabel("Residual score")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    save_single(fig, "FigureS2")


def s4_per_season() -> None:
    frame = read_csv(LGDI_DIR / "lgdi_whu_per_season_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    panel_label(axes[0], "A")
    axes[0].set_title("Per-season event weeks and alert performance")
    x = np.arange(len(frame))
    axes[0].bar(x - 0.18, frame["n_event_weeks"], width=0.36, color=COLORS["green"], label="Event weeks")
    axes[0].bar(x + 0.18, frame["tp"], width=0.36, color=COLORS["orange"], label="True-positive alerts")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(frame["season_id"], rotation=25, ha="right")
    axes[0].legend()
    panel_label(axes[1], "B")
    axes[1].set_title("Sensitivity, PPV, and false-alarm rate by season")
    for metric, color in [("sensitivity", COLORS["blue"]), ("ppv", COLORS["orange"]), ("false_alarm_rate", COLORS["gray"] )]:
        axes[1].plot(frame["season_id"], frame[metric], marker="o", color=color, label=metric.replace("_", " "))
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    save_single(fig, "FigureS3")


def s7_whu_xgb_folds() -> None:
    xgb = read_csv(XGB_VISIT_DIR / "visit_order_first_version_audit.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("WHU XGBoost fold-level R2 by configuration")
    for _, row in xgb.iterrows():
        folds = ast.literal_eval(row["r2_folds"])
        label = f"VO>={int(row['min_visit_order'])}, cap {int(row['gap_cap'])}d, {row['split_mode']}"
        ax.plot(range(1, len(folds) + 1), folds, marker="o", lw=1.0, label=label)
    ax.set_xlabel("Fold")
    ax.set_ylabel("R2")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    save_single(fig, "FigureS4")


def s9_frequent_subset() -> None:
    xgb = read_csv(XGB_VISIT_DIR / "visit_order_first_version_audit.csv")
    freq = xgb[xgb["min_visit_order"] >= 20].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("Frequent-visitor subset is selective supplementary evidence")
    labels = [f"{row.split_mode}\n{int(row.n_features)} feat" for row in freq.itertuples()]
    ax.bar(labels, freq["r2_mean"], yerr=freq["r2_std"], color=COLORS["orange"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cross-validated R2")
    ax.text(0.02, 0.08, "n=299; higher R2 is not used as the main performance claim.", transform=ax.transAxes, fontsize=8)
    save_single(fig, "FigureS5")


def external_detail(dataset: str, output_name: str, panel: str) -> None:
    summary = read_csv(GPU_XGB_DIR / "native_xgboost_public_summary.csv")
    row = summary[summary["dataset"] == dataset]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    panel_label(axes[0], panel)
    axes[0].set_title(f"{short_label(dataset)} native-feature metrics")
    if row.empty:
        # Structured placeholder — training output not available for this dataset
        _ph_metrics = ["ROC-AUC", "AP", "PPV", "Sensitivity"]
        _ph_bars = axes[0].bar(_ph_metrics, [0, 0, 0, 0],
                               color=COLORS["gray"], alpha=0.35, hatch="//")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Metric")
        axes[0].text(0.5, 0.5, "Training output\nnot available",
                     ha="center", va="center", transform=axes[0].transAxes,
                     fontsize=9, color=COLORS["warm_gray"], style="italic")
        _ph_feats = [f"feature_{i + 1}" for i in range(6)]
        axes[1].barh(_ph_feats, [0] * 6, color=COLORS["gray"], alpha=0.35, hatch="//")
        axes[1].set_xlabel("Importance")
        axes[1].text(0.5, 0.5, "Feature importance\nnot available",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=9, color=COLORS["warm_gray"], style="italic")
    else:
        row = row.iloc[0]
        metrics = pd.Series(
            {
                "ROC-AUC": row["roc_auc"],
                "AP": row["average_precision"],
                "PPV": row["ppv_precision"],
                "Sensitivity": row["sensitivity_recall"],
            }
        )
        _bar_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
        _m_bars = axes[0].bar(metrics.index, metrics.values, color=_bar_colors)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Metric")
        for _b, _v in zip(_m_bars, metrics.values):
            axes[0].text(_b.get_x() + _b.get_width() / 2,
                         _b.get_height() + 0.015, f"{_v:.3f}",
                         ha="center", va="bottom", fontsize=7)
        # Explain zero PPV/Sensitivity when threshold is too restrictive
        _ppv_val = float(row["ppv_precision"])
        _sens_val = float(row["sensitivity_recall"])
        if _ppv_val == 0.0 and _sens_val == 0.0:
            _thr = row.get("threshold", float("nan"))
            axes[0].annotate(
                f"Threshold={_thr:.4f}\n(too restrictive\n\u2192 zero positive\npredictions)",
                xy=(2.5, 0.05), xytext=(2.5, 0.35),
                fontsize=7, color=COLORS["red"], ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="mistyrose", alpha=0.8,
                          ec=COLORS["red"], lw=0.7),
            )
        feats_path = GPU_XGB_DIR / dataset / f"{dataset}_feature_importance.csv"
        panel_label(axes[1], chr(ord(panel) + 1))
        axes[1].set_title("Top native features")
        if feats_path.exists():
            feats = read_csv(feats_path)
            if "gain" not in feats.columns:
                value_cols = [col for col in feats.columns if col != "feature"]
                feats = feats.rename(columns={value_cols[0]: "gain"}) if value_cols else feats.assign(gain=0.0)
            feats = feats.sort_values("gain", ascending=False).head(8)
            axes[1].barh(feats["feature"][::-1], feats["gain"][::-1], color=COLORS["teal"])
            axes[1].set_xlabel("Importance")
        else:
            add_note(axes[1], "No feature-importance output found.")
    save_single(fig, output_name)


def s15_group_drivers() -> None:
    frame = load_validation()
    score_cols = [col for col in frame.columns if col.startswith("score_")]
    event = frame[frame["flu_event_window"].astype(bool)].copy()
    event["top_group"] = event[score_cols].idxmax(axis=1).str.replace("score_", "", regex=False)
    counts = event["top_group"].value_counts().reindex([col.replace("score_", "") for col in score_cols]).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("Leading chronic-disease group during FluNet event weeks")
    ax.barh(counts.index, counts.values, color=COLORS["blue"])
    ax.set_xlabel("Weeks ranked highest")
    total = counts.sum()
    for idx, value in enumerate(counts.values):
        ax.text(value + 0.2, idx, f"{value:.0f} ({100 * value / total:.1f}%)" if total else "0", va="center")
    save_single(fig, "FigureS9")


def s16_q4_audit() -> None:
    focus = parse_dates(read_csv(LGDI_DIR / "lgdi_whu_2019_focus.csv"), column="window_anchor_dt")
    qcorr = read_csv(LGDI_DIR / "sch_quarterly_profile_correlation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    # Panel A: 2019 weekly resp_score and LGDI time series
    ax = axes[0]
    panel_label(ax, "A")
    ax.set_title("2019 WHU monitoring: resp. score and LGDI")
    ax.plot(focus["window_anchor_dt"], focus["resp_score"],
            color=COLORS["blue"], lw=1.1, label="Resp. score")
    ax.set_ylabel("Resp. score", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax2 = ax.twinx()
    ax2.plot(focus["window_anchor_dt"], focus["lgdi"],
             color=COLORS["orange"], lw=1.1, ls="--", label="LGDI")
    ax2.set_ylabel("LGDI", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax.axhline(0, color=COLORS["gray"], lw=0.7, ls=":")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Week anchor")
    ax.text(0.01, 0.97,
            "Q4-2019 vs Q4-2018 permutation p\u200a=\u200a0.53\n"
            "(failed to detect distinguishable elevation)",
            transform=ax.transAxes, fontsize=6.5, va="top", color=COLORS["warm_gray"])

    # Panel B: post-gap quarterly Pearson r by disease group
    ax = axes[1]
    panel_label(ax, "B")
    ax.set_title("Post-gap quarterly profile correlation by group")
    groups = sorted(qcorr["group"].unique())
    quarters = sorted(qcorr["quarter"].unique())
    x = np.arange(len(quarters))
    _grp_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"],
                   COLORS["teal"], COLORS["purple"], COLORS["red"]]
    for i, grp in enumerate(groups):
        grp_data = (qcorr[qcorr["group"] == grp]
                    .set_index("quarter")
                    .reindex(quarters)["pearson_r"])
        ax.plot(x, grp_data.values, marker=".", lw=1.0,
                color=_grp_colors[i % len(_grp_colors)], label=grp)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([quarters[i] for i in range(0, len(quarters), 2)],
                       rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    save_single(fig, "FigureS10")


_CANONICAL_GROUPS = [
    "Cardiovascular", "Hypertension", "Diabetes",
    "Cerebrovascular", "Renal", "Respiratory",
]

_EXTERNAL_CTRL_DIR = NC_DIR / "external_positive_control_results"

_EXTERNAL_GROUP_MAP = {
    "cardiovascular": "Cardiovascular",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "cerebrovascular": "Cerebrovascular",
    "kidney": "Renal",
    "chronic_respiratory": "Respiratory",
}


def s11_pathogen_activation_signature() -> None:
    """FigureS11: heatmap of group-level activation across pathogens/datasets."""
    apply_style()

    # ── Column 1: WHU FluNet (% of event weeks where group ranked #1) ──────
    whu_path = LGDI_DIR / "lgdi_whu_influenza_validation.csv"
    whu_col: list[float | None] = [None] * len(_CANONICAL_GROUPS)
    if whu_path.exists():
        val = read_csv(whu_path)
        score_cols_map = {
            "Cardiovascular": "score_Cardiovascular",
            "Hypertension": "score_Hypertension",
            "Diabetes": "score_Diabetes",
            "Cerebrovascular": "score_Cerebrovascular",
            "Renal": "score_Renal",
            "Respiratory": "score_Respiratory",
        }
        flu_rows = val[val.get("flu_event_window", val.columns[0]).astype(bool) == True] if "flu_event_window" in val.columns else val
        if len(flu_rows) == 0:
            flu_rows = val
        ranks: dict[str, int] = {g: 0 for g in _CANONICAL_GROUPS}
        valid_cols = [sc for sc in score_cols_map.values() if sc in flu_rows.columns]
        if valid_cols:
            sub = flu_rows[[c for c in valid_cols]]
            for _, row in sub.iterrows():
                top = row.idxmax()
                for g, sc in score_cols_map.items():
                    if sc == top:
                        ranks[g] += 1
                        break
            total = len(flu_rows)
            if total > 0:
                whu_col = [100.0 * ranks.get(g, 0) / total for g in _CANONICAL_GROUPS]

    # ── Columns 2-4: external profile correlations ────────────────────────
    ext_datasets = [
        ("MIMIC-IV\nInfluenza", "mimic_influenza_profile_correlations.csv"),
        ("eICU\nViral Pneumonia", "eicu_viral_pneumonia_profile_correlations.csv"),
        ("NWICU\nCOVID-19", "nwicu_covid_profile_correlations.csv"),
    ]
    ext_cols: list[list[float | None]] = []
    for _, fname in ext_datasets:
        fpath = _EXTERNAL_CTRL_DIR / fname
        col_vals: list[float | None] = [None] * len(_CANONICAL_GROUPS)
        if fpath.exists():
            df = read_csv(fpath)
            # expects columns: group (or disease_group), pearson_r (or correlation)
            grp_col = next((c for c in df.columns if "group" in c.lower()), None)
            corr_col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in (
                        "pearson_profile_correlation",
                        "pearson_r",
                        "correlation",
                        "r",
                        "rho",
                    )
                ),
                None,
            )
            if grp_col and corr_col:
                grp_map = dict(zip(df[grp_col].astype(str).str.lower(), df[corr_col].astype(float)))
                for i, g in enumerate(_CANONICAL_GROUPS):
                    for raw_key, canon in _EXTERNAL_GROUP_MAP.items():
                        if canon == g and raw_key in grp_map:
                            col_vals[i] = grp_map[raw_key]
                            break
        ext_cols.append(col_vals)

    # ── Assemble matrix ────────────────────────────────────────────────────
    col_labels = ["WHU\nInfluenza"] + [lbl for lbl, _ in ext_datasets]
    n_groups = len(_CANONICAL_GROUPS)
    n_cols = len(col_labels)
    raw = np.full((n_groups, n_cols), np.nan)
    for i, v in enumerate(whu_col):
        if v is not None:
            raw[i, 0] = v
    for j, col_vals in enumerate(ext_cols):
        for i, v in enumerate(col_vals):
            if v is not None:
                raw[i, j + 1] = v

    # Normalize each column independently (z-score, clip to [-2, 2])
    norm = np.full_like(raw, np.nan)
    for j in range(n_cols):
        col = raw[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            mu, sd = valid.mean(), valid.std()
            if sd > 0:
                norm[:, j] = np.clip((col - mu) / sd, -2, 2)
            else:
                norm[:, j] = 0.0
        elif len(valid) == 1:
            norm[:, j] = 0.0

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("FigureS11: Pathogen Activation Signature by Chronic-Disease Group", fontsize=10)

    masked = np.ma.masked_invalid(norm)
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(masked, cmap=cmap, vmin=-2, vmax=2, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(_CANONICAL_GROUPS, fontsize=9)
    ax.set_xlabel("Dataset / Pathogen")
    ax.set_ylabel("Chronic-disease group")

    # Annotate cells with raw values; grey-out missing
    for i in range(n_groups):
        for j in range(n_cols):
            v_raw = raw[i, j]
            v_norm = norm[i, j]
            if np.isnan(v_raw):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#888888")
            else:
                label_str = f"{v_raw:.1f}" if j == 0 else f"{v_raw:.2f}"
                txt_clr = "white" if abs(v_norm) > 1.3 else COLORS["dark"]
                ax.text(j, i, label_str, ha="center", va="center", fontsize=7.5, color=txt_clr)

    # Mark NWICU respiratory contradiction
    nwicu_col_idx = n_cols - 1
    resp_row_idx = _CANONICAL_GROUPS.index("Respiratory")
    if not np.isnan(raw[resp_row_idx, nwicu_col_idx]):
        ax.text(nwicu_col_idx, resp_row_idx + 0.42,
                "\u2020", ha="center", va="center", fontsize=11, color=COLORS["orange"])

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalised activation (z-score, clipped \u00b12)", fontsize=8)

    # Legend / footnote
    note_lines = [
        "WHU Influenza: % of flu event-weeks group ranked #1 (z-scored per column).",
        "External datasets: Pearson profile correlation (z-scored per column).",
        "\u2020 NWICU COVID-19 Respiratory: r = \u22120.198 (negative control, hypothesis-generating).",
        "Colour scale is normalised within each dataset column; raw values are printed in each cell.",
    ]
    fig.text(0.01, -0.04, "\n".join(note_lines), fontsize=6.8,
             va="top", ha="left", color=COLORS["gray"], wrap=True)

    save_figure(fig, SUPP_FIGURES_DIR / "FigureS11")


# ─── FigureS12: WHO FluNet aggregate surveillance ────────────────────────────
_FLUNET_DIR = NC_DIR.parent / "external_data" / "flunet"


def s12_flunet_aggregate() -> None:
    """FigureS12: WHO FluNet aggregate influenza surveillance (China + world ex-China)."""
    china = pd.read_csv(_FLUNET_DIR / "flunet_china_2009_2024.csv", low_memory=False)
    china["date"] = pd.to_datetime(china["ISO_WEEKSTARTDATE"], errors="coerce")
    china["year"] = china["date"].dt.year
    china["spec"] = pd.to_numeric(china["SPEC_PROCESSED_NB"], errors="coerce").fillna(0)
    china["inf"] = pd.to_numeric(china["INF_ALL"], errors="coerce").fillna(0)

    ann_chn = (
        china.groupby("year")[["inf", "spec"]].sum()
        .assign(pos=lambda d: d["inf"] / d["spec"].replace(0, np.nan) * 100)
    )

    glb = pd.read_csv(_FLUNET_DIR / "flunet_global_2009_2024.csv", low_memory=False)
    glb = glb[glb["COUNTRY_AREA_TERRITORY"].str.strip() != "China"].copy()
    glb["year"] = pd.to_datetime(glb["ISO_WEEKSTARTDATE"], errors="coerce").dt.year
    glb["spec"] = pd.to_numeric(glb["SPEC_PROCESSED_NB"], errors="coerce").fillna(0)
    glb["inf"] = pd.to_numeric(glb["INF_ALL"], errors="coerce").fillna(0)
    ann_glb = (
        glb.groupby("year")[["inf", "spec"]].sum()
        .assign(pos=lambda d: d["inf"] / d["spec"].replace(0, np.nan) * 100)
    )

    # Panel B: 4-week rolling weekly China 2018-2024
    recent = china[china["year"] >= 2018].sort_values("date").copy()
    recent["pos"] = recent["inf"] / recent["spec"].replace(0, np.nan) * 100
    recent["rolling"] = recent["pos"].rolling(4, min_periods=2).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    # Panel A: grouped bar chart
    years = sorted(set(ann_chn.index) & set(ann_glb.index))
    xs = np.arange(len(years))
    w = 0.38
    axes[0].bar(xs - w / 2, ann_chn.loc[years, "pos"], width=w,
                color=COLORS["blue"], label="China", alpha=0.85)
    axes[0].bar(xs + w / 2, ann_glb.loc[years, "pos"], width=w,
                color=COLORS["orange"], label="World ex-China", alpha=0.85)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([str(y) for y in years], rotation=55, ha="right", fontsize=7)
    axes[0].set_ylabel("Influenza positivity (%)", fontsize=8)
    axes[0].set_title("Annual influenza positivity\n(WHO FluNet, 2009–2024)", fontsize=8.5)
    axes[0].legend(fontsize=7.5, framealpha=0.85)
    axes[0].set_ylim(0, None)

    # Panel B: rolling weekly line chart
    axes[1].plot(recent["date"], recent["rolling"],
                 color=COLORS["blue"], lw=1.4, label="4-wk rolling positivity")
    axes[1].fill_between(recent["date"], recent["rolling"].fillna(0),
                         alpha=0.18, color=COLORS["blue"])
    axes[1].axvline(pd.Timestamp("2020-01-20"), color=COLORS["red"],
                    lw=0.9, ls="--", alpha=0.75)
    y_max_b = float(np.nanmax(recent["rolling"].values)) if recent["rolling"].notna().any() else 5.0
    axes[1].text(pd.Timestamp("2020-02-01"), y_max_b * 0.88,
                 "COVID-19\nonset", fontsize=7, color=COLORS["red"], va="top")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].tick_params(axis="x", labelsize=7, rotation=30)
    axes[1].set_ylabel("Influenza positivity (%)", fontsize=8)
    axes[1].set_title("Rolling weekly influenza positivity\n(China, 2018–2024)", fontsize=8.5)
    axes[1].set_ylim(0, None)

    save_figure(fig, SUPP_FIGURES_DIR / "FigureS12")


# ─── FigureS13: WHU strategy comparison ─────────────────────────────────────
def s13_strategy_comparison() -> None:
    """FigureS13: WHU alert-strategy operating space (single panel).
    Panel B (lag-Spearman bars) removed — duplicates Figure 4A."""
    metrics = read_csv(LGDI_DIR / "lgdi_whu_influenza_metrics.csv")

    _labels = {
        "REFERENCE_resp_mean_plus_1_5sd":        "Ref-resp\n(1.5 SD)",
        "REFERENCE_resp_mean_plus_2sd":           "Ref-resp\n(2 SD)",
        "LGDI_mean_plus_1_5sd":                  "LGDI\n(1.5 SD)",
        "consensus_2groups_mean_plus_1_5sd":      "Consensus\n2-grp (1.5 SD)",
        "consensus_3groups_mean_plus_1_5sd":      "Consensus\n3-grp (1.5 SD)",
        "consensus_2groups_season_oct_apr":        "Seasonal 2-grp\n(Oct–Apr)",
        "REFERENCE_resp_mean_plus_1sd_relaxed":   "Ref-resp\n(relaxed)",
        "season_sustained_consensus2grp_nov_mar": "Season+\nsustained (S4)",
    }

    ppv  = metrics["ppv"].values
    sens = metrics["sensitivity"].values
    far  = metrics["false_alarm_rate"].values
    strats = metrics["strategy"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    panel_label(ax, "A")

    # PPV vs Sensitivity scatter coloured by FAR
    sc = ax.scatter(sens, ppv, c=far, cmap="RdYlGn_r", vmin=0, vmax=0.35,
                    s=90, zorder=3, edgecolors="gray", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="False alarm rate", shrink=0.82, pad=0.02)
    for i, s in enumerate(strats):
        lbl = _labels.get(s, s[:12])
        ax.annotate(lbl, (sens[i], ppv[i]), fontsize=6, ha="center",
                     xytext=(0, 9), textcoords="offset points", color=COLORS["dark"])
    ax.set_xlabel("Sensitivity", fontsize=8)
    ax.set_ylabel("PPV (precision)", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", lw=0.6, ls="--", alpha=0.45)
    ax.axvline(0.5, color="gray", lw=0.6, ls="--", alpha=0.45)
    ax.set_title("WHU influenza-alert strategy\noperating space (n=8)", fontsize=8.5)

    save_figure(fig, SUPP_FIGURES_DIR / "FigureS13")


# ─── FigureS14: External dataset forest plots ────────────────────────────────
def s14_external_forest_plots() -> None:
    """FigureS14: Profile-correlation forest plots for public external datasets."""
    datasets = [
        ("MIMIC-IV\nInfluenza",
         _EXTERNAL_CTRL_DIR / "mimic_influenza_profile_correlations.csv"),
        ("eICU\nViral Pneumonia",
         _EXTERNAL_CTRL_DIR / "eicu_viral_pneumonia_profile_correlations.csv"),
        ("NWICU\nCOVID-19",
         _EXTERNAL_CTRL_DIR / "nwicu_covid_profile_correlations.csv"),
    ]
    group_order  = ["chronic_respiratory", "cardiovascular", "hypertension",
                    "diabetes", "cerebrovascular", "kidney"]
    group_labels = ["Respiratory", "Cardiovascular", "Hypertension",
                    "Diabetes", "Cerebrovascular", "Kidney"]
    n_g  = len(group_order)
    ys   = np.arange(n_g)
    ds_colors = [COLORS["teal"], COLORS["orange"], COLORS["purple"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    panel_label(axes[2], "C")

    for ax, (title, fpath), col in zip(axes, datasets, ds_colors):
        df = read_csv(fpath).set_index("group").reindex(group_order)
        r  = df["pearson_profile_correlation"].values
        lo = df["bootstrap_ci_low"].values
        hi = df["bootstrap_ci_high"].values

        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.barh(ys, r,
                xerr=[r - lo, hi - r],
                error_kw=dict(lw=1.2, capsize=3, capthick=1.0, ecolor=COLORS["dark"]),
                color=col, alpha=0.75, height=0.55)
        ax.set_yticks(ys)
        ax.set_yticklabels(group_labels, fontsize=8)
        ax.set_xlabel("Profile correlation (Pearson r)", fontsize=8)
        ax.set_xlim(-0.55, 1.1)
        ax.set_title(f"{title}\nprofile correlations (95% CI)", fontsize=8.5)

        # Mark the respiratory group
        resp_idx = group_order.index("chronic_respiratory")
        ax.axhline(resp_idx, color=COLORS["teal"], lw=0.5, ls=":", alpha=0.6)

    save_figure(fig, SUPP_FIGURES_DIR / "FigureS14")


def main() -> None:
    ensure_dirs()
    apply_style()
    s1_flow()                                              # S1: cohort flow
    s2_group_scores()                                      # S2: all-group residual time series
    s4_per_season()                                        # S3: per-season performance
    s7_whu_xgb_folds()                                    # S4: WHU XGBoost fold R2
    s9_frequent_subset()                                   # S5: frequent-visitor subset
    external_detail("mimic_iv_native_features", "FigureS6", "A")   # S6: MIMIC-IV detail
    external_detail("eicu_native_features", "FigureS7", "A")       # S7: eICU detail
    external_detail("nwicu_native_features", "FigureS8", "A")      # S8: NWICU detail
    s15_group_drivers()                                    # S9: leading disease group
    s16_q4_audit()                                         # S10: 2019 Q4 audit
    s11_pathogen_activation_signature()                    # S11: pathogen activation heatmap
    s12_flunet_aggregate()                                 # S12: WHO FluNet aggregate
    s13_strategy_comparison()                              # S13: strategy scatter + lag-Spearman
    s14_external_forest_plots()                            # S14: external forest plots


if __name__ == "__main__":
    main()