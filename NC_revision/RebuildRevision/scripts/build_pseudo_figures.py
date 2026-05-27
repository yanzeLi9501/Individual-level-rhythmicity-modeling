"""
build_pseudo_figures.py — Enhanced 3D / advanced-visualisation versions of key figures.

Generates (to RebuildRevision/pseudo_figures/):
  Figure2B_pseudo    – 3D surface: 6 disease groups × time × residual score
  Figure4D_pseudo    – 3D ribbon/waterfall: flu subtypes × lag × Spearman ρ
  Figure6_pseudo     – Parallel coordinates: cross-dataset performance signatures
  FigureS11_pseudo   – 3D grouped bar: pathogen-specific activation signatures
  Figure2_trajectory_pseudo – 3D parametric trajectory: time × admissions × resp_score

All figures use the spring/summer palette from rebuild_common and are saved
as png/pdf/svg/tif WITHOUT overwriting the originals in figures/ or supp_figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_common import (
    COLORS,
    FIGURES_DIR,
    GPU_XGB_DIR,
    LGDI_DIR,
    REBUILD_DIR,
    SHADING,
    apply_style,
    ensure_dirs,
    panel_label,
    parse_dates,
    read_csv,
    read_json,
    save_figure,
    short_label,
)

PSEUDO_FIGURES_DIR = REBUILD_DIR / "pseudo_figures"
FORMATS = ("png", "pdf", "svg", "tif")

# Disease-group display names (consistent ordering)
GROUP_NAMES = [
    "Cerebrovascular",
    "Hypertension",
    "Diabetes",
    "Cardiovascular",
    "Renal",
    "Respiratory",
]
GROUP_COLORS = {
    "Cerebrovascular": COLORS["coral"],
    "Hypertension":    COLORS["amber"],
    "Diabetes":        COLORS["teal"],
    "Cardiovascular":  COLORS["blue"],
    "Renal":           COLORS["purple"],
    "Respiratory":     COLORS["green"],
}


def save_pseudo(fig: plt.Figure, name: str) -> None:
    """Save a pseudo-figure to the pseudo_figures directory."""
    PSEUDO_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = PSEUDO_FIGURES_DIR / name
    fig.suptitle("")
    for fmt in FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  ✓ saved {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure2B_pseudo — 3D Surface Plot
#   6 disease groups × 213 monitoring weeks × residual score
#   Replaces the flat imshow() heatmap with a literal "activation terrain"
# ═══════════════════════════════════════════════════════════════════════════════

def figure2B_pseudo() -> None:
    """3D surface: multi-system co-activation as a topographic terrain."""
    apply_style()
    val = parse_dates(read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv"))

    score_cols = [
        "score_Cerebrovascular", "score_Hypertension", "score_Diabetes",
        "score_Cardiovascular", "score_Renal", "score_Respiratory",
    ]
    # Build Z matrix: rows = groups, cols = weeks
    Z = val[score_cols].T.to_numpy(dtype=float)  # (6, 213)

    # Create meshgrid
    n_groups, n_weeks = Z.shape
    X_time = np.arange(n_weeks)          # week index
    Y_groups = np.arange(n_groups)        # group index
    X, Y = np.meshgrid(X_time, Y_groups)

    # ── 3D figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Surface
    z_max = np.nanmax(np.abs(Z))
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="RdBu_r",
        vmin=-z_max, vmax=z_max,
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
        rstride=1, cstride=1,
    )

    # FluNet event-week vertical planes (semi-transparent green)
    event_mask = val["flu_event_window"].astype(bool).values
    event_indices = np.where(event_mask)[0]
    for ei in event_indices[::3]:  # decimate for performance / visual clarity
        verts = [
            [(ei, 0, -z_max), (ei, n_groups - 1, -z_max),
             (ei, n_groups - 1, z_max), (ei, 0, z_max)],
        ]
        ax.add_collection3d(
            Poly3DCollection(verts, facecolor=COLORS["green"],
                             alpha=0.06, edgecolor="none", zorder=1),
        )

    # Labels & ticks
    ax.set_xlabel("Monitoring week", labelpad=10)
    ax.set_ylabel("Chronic-disease group", labelpad=10)
    ax.set_zlabel("Residual score", labelpad=8)
    ax.set_yticks(Y_groups)
    ax.set_yticklabels([g[:6] for g in GROUP_NAMES], fontsize=8)

    # Time-axis tick labels (year markers)
    week_starts = val["week_start"]
    year_ticks, year_labels = [], []
    for yi in range(2016, 2021):
        mask = week_starts.dt.year == yi
        if mask.any():
            idx = np.where(mask)[0][0]
            year_ticks.append(idx)
            year_labels.append(str(yi))
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, fontsize=8)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=16, pad=0.08)
    cbar.set_label("Residual score", fontsize=9)

    # View angle — optimized to show the "cerebrovascular ridge" prominently
    ax.view_init(elev=22, azim=-55)
    ax.set_title(
        "Multi-system residual activation terrain\n"
        "6 chronic-disease groups × 213 monitoring weeks (WHU 2016–2020)",
        fontsize=11, pad=18,
    )

    # Annotation box
    ax.text2D(0.02, 0.96,
              "Green planes = FluNet event weeks\n"
              f"Cerebrovascular peak = {Z[0].max():.3f}  |  "
              f"Respiratory peak = {Z[5].max():.3f}",
              transform=ax.transAxes, fontsize=8,
              color=COLORS["dark"],
              bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                        ec=COLORS["gray"], lw=0.8))

    save_pseudo(fig, "Figure2B_pseudo")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure4D_pseudo — 3D Ribbon / Waterfall Plot
#   Flu subtypes × lag weeks × Spearman ρ
#   Each subtype = a coloured ribbon showing correlation decay with lag
# ═══════════════════════════════════════════════════════════════════════════════

def figure4D_pseudo() -> None:
    """3D ribbon: flu-subtype lag-Spearman correlations as a waterfall."""
    apply_style()
    sub = read_csv(LGDI_DIR / "lgdi_whu_flunet_subtype_lag.csv")

    subtype_palette = {
        "AH3":        COLORS["blue"],
        "AH1N12009":  COLORS["coral"],
        "INF_B":      COLORS["amber"],
        "BVIC":       COLORS["purple"],
        "BYAM":       COLORS["teal"],
    }
    subtype_label = {
        "AH3": "A/H3N2", "AH1N12009": "A/H1N1pdm09",
        "INF_B": "B (combined)", "BVIC": "B/Vic", "BYAM": "B/Yam",
    }

    subtypes = sorted(sub["subtype"].unique(), key=lambda s: str(s))
    lags_all = sorted(sub["lag_weeks"].unique())

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Draw a ribbon for each subtype
    for si, st in enumerate(subtypes):
        grp = sub[sub["subtype"] == st].sort_values("lag_weeks")
        xs = grp["lag_weeks"].values.astype(float)
        ys = np.full_like(xs, si)
        zs = grp["spearman_rho"].values.astype(float)
        color = subtype_palette.get(str(st), COLORS["gray"])
        label = subtype_label.get(str(st), str(st))

        # 3D line
        ax.plot(xs, ys, zs, color=color, lw=2.5, label=label, zorder=5)

        # Ribbon fill (build polygons between consecutive lag points)
        for i in range(len(xs) - 1):
            poly_verts = [
                (xs[i],     si, zs[i]),
                (xs[i + 1], si, zs[i + 1]),
                (xs[i + 1], si, 0),
                (xs[i],     si, 0),
            ]
            ribbon = Poly3DCollection([poly_verts], facecolor=color, alpha=0.18,
                                       edgecolor="none", zorder=2)
            ax.add_collection3d(ribbon)

        # Scatter markers at each lag point
        ax.scatter(xs, ys, zs, c=color, s=30, edgecolors="white",
                   linewidths=0.5, zorder=6)

    # Zero-reference plane
    xx_plane, yy_plane = np.meshgrid(
        np.linspace(min(lags_all), max(lags_all), 20),
        np.linspace(-0.5, len(subtypes) - 0.5, 20),
    )
    zz_plane = np.zeros_like(xx_plane)
    ax.plot_surface(xx_plane, yy_plane, zz_plane,
                    facecolor=COLORS["gray"], alpha=0.10, edgecolor="none", zorder=0)

    # Axes
    ax.set_xlabel("LGDI leads FluNet by (weeks)", labelpad=10)
    ax.set_ylabel("Flu subtype", labelpad=10)
    ax.set_zlabel("Spearman ρ", labelpad=8)
    ax.set_yticks(range(len(subtypes)))
    ax.set_yticklabels([subtype_label.get(s, s) for s in subtypes], fontsize=8)
    ax.set_xticks(lags_all)
    ax.set_zlim(-0.30, 0.30)
    ax.view_init(elev=18, azim=-50)

    ax.set_title(
        "FluNet subtype-specific lag-Spearman signatures\n"
        "LGDI leads FluNet — different subtypes show distinct decay profiles",
        fontsize=11, pad=18,
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # Annotation
    ax.text2D(0.02, 0.96,
              "A/H3N2 & B: positive peak at lag-4 (ρ≈0.20)\n"
              "A/H1N1pdm09: negative at lag-0 (ρ≈−0.18)\n"
              "→ subtype-specific lag structure supports biological signal",
              transform=ax.transAxes, fontsize=8,
              color=COLORS["dark"],
              bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                        ec=COLORS["gray"], lw=0.8))

    save_pseudo(fig, "Figure4D_pseudo")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure6_pseudo — Parallel Coordinates Plot
#   Collapses Fig 6A–C into one multi-dimensional view
#   Each dataset = one polyline traversing 6 standardised metrics
# ═══════════════════════════════════════════════════════════════════════════════

def figure6_pseudo() -> None:
    """Parallel coordinates: cross-dataset performance signatures."""
    apply_style()
    summary = read_csv(GPU_XGB_DIR / "native_xgboost_public_summary.csv")

    # Build the parallel-coordinates DataFrame
    pc_data = pd.DataFrame({
        "dataset":       summary["dataset"].map(short_label),
        "ROC-AUC":       summary["roc_auc"],
        "Avg Precision": summary["average_precision"],
        "PPV":           summary["ppv_precision"],
        "Sensitivity":   summary["sensitivity_recall"],
        "log10(Rows)":   np.log10(summary["n_rows"]),
        "Prevalence %":  summary["positive_count"] / summary["n_rows"] * 100,
    })

    ds_colors = {
        "MIMIC-IV": COLORS["blue"],
        "eICU":     COLORS["orange"],
        "NWICU":    COLORS["purple"],
    }

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")

    metrics = ["ROC-AUC", "Avg Precision", "PPV", "Sensitivity",
               "log10(Rows)", "Prevalence %"]
    x = np.arange(len(metrics))

    for _, row in pc_data.iterrows():
        ds = row["dataset"]
        color = ds_colors.get(ds, COLORS["gray"])
        vals = [row[m] for m in metrics]

        # Normalise to [0, 1] per metric for visual comparability
        norm_vals = []
        for mi, m in enumerate(metrics):
            col_min = pc_data[m].min()
            col_max = pc_data[m].max()
            rng = col_max - col_min
            if rng > 0:
                norm_vals.append((vals[mi] - col_min) / rng)
            else:
                norm_vals.append(0.5)

        ax.plot(x, norm_vals, marker="o", color=color, lw=2.2, ms=8,
                markeredgecolor="white", markeredgewidth=0.8, label=ds,
                alpha=0.88, zorder=3)

        # Annotate actual values at each axis
        for xi, (nv, raw) in enumerate(zip(norm_vals, vals)):
            y_offset = 10 if ds == "MIMIC-IV" else (-10 if ds == "NWICU" else 12)
            ax.annotate(
                f"{raw:.3f}" if raw < 10 else f"{raw:.1f}",
                (xi, nv),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center", fontsize=7, color=color, alpha=0.85,
            )

    # Axis setup
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlim(-0.3, len(metrics) - 0.7)
    ax.set_ylabel("Normalised metric (0–1 per axis)", fontsize=8.5, color=COLORS["gray"])

    # Vertical gridlines
    for xi in x:
        ax.axvline(xi, color=COLORS["light_gray"], lw=0.8, ls="--", alpha=0.5, zorder=0)

    ax.set_title(
        "Cross-dataset performance signatures (parallel coordinates)\n"
        "Each polyline = one public ICU dataset traversing 6 standardised metrics",
        fontsize=11, pad=14,
    )
    ax.legend(fontsize=9, loc="upper right",
              bbox_to_anchor=(1.12, 1.02))

    # Annotation
    ax.text(0.02, 0.06,
            "MIMIC-IV: low PPV/Sens despite large N → heterogeneous general ICU\n"
            "NWICU: high AUC but moderate PPV → COVID-specific ICU with stronger signal\n"
            "eICU: near-zero PPV/Sens at default threshold → viral pneumonia rare in mixed ICU",
            transform=ax.transAxes, fontsize=7.5,
            color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                      ec=COLORS["gray"], lw=0.8))

    save_pseudo(fig, "Figure6_pseudo")


# ═══════════════════════════════════════════════════════════════════════════════
# FigureS11_pseudo — 3D Grouped Bar Chart
#   Pathogen-specific chronic-disease group activation signatures
#   6 disease groups × N datasets × mean residual z-score
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_whu_event_zscores() -> dict[str, float]:
    """Compute mean residual z-score per group during WHU FluNet event weeks."""
    val = parse_dates(read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv"))
    events = val[val["flu_event_window"].astype(bool)]
    score_cols = [f"score_{g}" for g in GROUP_NAMES]
    means = events[score_cols].mean()
    # Z-score across groups within event weeks
    grand_mean = means.mean()
    grand_std = means.std()
    if grand_std == 0:
        return {g: 0.0 for g in GROUP_NAMES}
    return {g: float((means[f"score_{g}"] - grand_mean) / grand_std) for g in GROUP_NAMES}


def _compute_nwicu_zscores() -> dict[str, float]:
    """Compute mean group_score per group from NWICU group metrics (COVID reference)."""
    path = LGDI_DIR / "lgdi_nwicu_group_metrics.csv"
    if not path.exists():
        return {}
    df = read_csv(path)
    name_map = {
        "Cardiovascular": "Cardiovascular", "Hypertension": "Hypertension",
        "Diabetes": "Diabetes", "Cerebrovascular": "Cerebrovascular",
        "Renal": "Renal", "Respiratory": "Respiratory",
    }
    means = {}
    for g in GROUP_NAMES:
        nw_name = name_map.get(g, g)
        rows = df[df["group"] == nw_name]
        if rows.empty:
            continue
        means[g] = float(rows["group_score"].mean())
    if not means:
        return {}
    gm = float(np.mean(list(means.values())))
    gs = float(np.std(list(means.values())))
    if gs == 0:
        return {g: 0.0 for g in GROUP_NAMES}
    return {g: float((means.get(g, gm) - gm) / gs) for g in GROUP_NAMES}


def figureS11_pseudo() -> None:
    """3D grouped bar: pathogen-specific activation signatures."""
    apply_style()

    whu_z = _compute_whu_event_zscores()
    nwicu_z = _compute_nwicu_zscores()

    # Build the data matrix: rows=groups, cols=datasets
    datasets = ["WHU\n(influenza)", "NWICU\n(COVID)"]
    data_matrix = np.zeros((len(GROUP_NAMES), len(datasets)))
    for gi, g in enumerate(GROUP_NAMES):
        data_matrix[gi, 0] = whu_z.get(g, 0.0)
        data_matrix[gi, 1] = nwicu_z.get(g, 0.0)

    # ── 3D grouped bar chart ──────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    n_groups = len(GROUP_NAMES)
    n_datasets = len(datasets)
    bar_width = 0.55
    bar_depth = 0.55
    x_spacing = 0.85  # spacing between groups
    y_spacing = 1.2   # spacing between datasets

    x_positions = np.arange(n_groups) * x_spacing

    for di in range(n_datasets):
        for gi in range(n_groups):
            z_val = data_matrix[gi, di]
            # Positive = coral, negative = blue
            if z_val >= 0:
                bar_color = COLORS["coral"]
            else:
                bar_color = COLORS["blue"]

            ax.bar3d(
                x_positions[gi] - bar_width / 2,
                di * y_spacing - bar_depth / 2,
                0,
                bar_width, bar_depth, z_val,
                color=bar_color, alpha=0.82, edgecolor="white",
                linewidth=0.5, zsort="average",
            )

            # Value label on top of each bar
            label_z = z_val + 0.04 if z_val >= 0 else z_val - 0.08
            ax.text(
                x_positions[gi],
                di * y_spacing,
                label_z,
                f"{z_val:+.2f}",
                ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color=COLORS["dark"],
            )

    # Zero plane
    xx_plane, yy_plane = np.meshgrid(
        np.linspace(x_positions[0] - 1, x_positions[-1] + 1, 10),
        np.linspace(-0.8, (n_datasets - 1) * y_spacing + 0.8, 10),
    )
    ax.plot_surface(xx_plane, yy_plane, np.zeros_like(xx_plane),
                    facecolor=COLORS["gray"], alpha=0.08, edgecolor="none", zorder=0)

    # Axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([g[:8] for g in GROUP_NAMES], fontsize=8, rotation=15)
    ax.set_yticks([di * y_spacing for di in range(n_datasets)])
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_zlabel("Mean residual z-score (within-dataset)", labelpad=8)
    ax.set_xlabel("Chronic-disease group", labelpad=10)

    ax.view_init(elev=22, azim=-50)
    ax.set_title(
        "Pathogen-specific chronic-disease group activation signatures\n"
        "Influenza (WHU) → cerebrovascular-dominant  |  COVID (NWICU) → respiratory-active",
        fontsize=11, pad=18,
    )

    # Legend for group colours
    legend_elements = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="white", label=g, alpha=0.82)
        for g in GROUP_NAMES
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="upper left",
              ncol=2, bbox_to_anchor=(0.02, 0.98))

    # Hypothesis annotation
    ax.text2D(0.02, 0.04,
              "Hypothesis-generating\n"
              "Influenza → multi-system co-activation (cerebrovascular peak)\n"
              "COVID → respiratory relative elevation\n"
              "NWICU r = −0.198: variant-period difference, exploratory only",
              transform=ax.transAxes, fontsize=7.5,
              color=COLORS["dark"],
              bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", alpha=0.90,
                        ec=COLORS["amber"], lw=1.0))

    save_pseudo(fig, "FigureS11_pseudo")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure2_trajectory_pseudo — 3D Parametric Trajectory
#   Bonus: time × admissions × resp_score with FluNet colour-coding
#   Shows the lead-lag relationship as a 3D helix / loop
# ═══════════════════════════════════════════════════════════════════════════════

def figure2_trajectory_pseudo() -> None:
    """3D trajectory: admissions vs resp_score vs time, coloured by FluNet positivity."""
    apply_style()
    val = parse_dates(read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv"))

    time_idx = np.arange(len(val))
    admissions = val["n_admissions"].values.astype(float)
    resp_score = val["resp_score"].values.astype(float)
    positivity = val["positivity"].values.astype(float)

    # Normalize positivity for colour mapping
    pos_norm = positivity / max(positivity.max(), 1e-6)
    flu_colors = plt.cm.YlOrRd(0.2 + 0.7 * pos_norm)  # pale yellow → deep red

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Plot as a continuous 3D line, segment-coloured by FluNet positivity
    for i in range(len(time_idx) - 1):
        ax.plot(
            time_idx[i:i + 2],
            admissions[i:i + 2],
            resp_score[i:i + 2],
            color=flu_colors[i], lw=1.2, alpha=0.75,
        )

    # Scatter FluNet event weeks as larger markers
    event_idx = np.where(val["flu_event_window"].astype(bool))[0]
    ax.scatter(
        time_idx[event_idx], admissions[event_idx], resp_score[event_idx],
        c=positivity[event_idx], cmap="YlOrRd",
        s=40, edgecolors="white", linewidths=0.5,
        vmin=0, vmax=positivity.max(),
        zorder=5, label="FluNet event weeks",
    )

    # Three-segment shading (calibration / validation / test)
    cal_end = pd.Timestamp("2017-07-01")
    val_end = pd.Timestamp("2019-01-01")
    cal_mask = val["week_start"] < cal_end
    val_mask = (val["week_start"] >= cal_end) & (val["week_start"] < val_end)
    cal_idx = int(cal_mask.sum())
    val_idx = int(val_mask.sum())

    for seg_start, seg_end, color, label in [
        (0, cal_idx, COLORS["blue"], "Calibration"),
        (cal_idx, cal_idx + val_idx, COLORS["amber"], "Validation"),
        (cal_idx + val_idx, len(time_idx), COLORS["coral"], "Quasi-test"),
    ]:
        seg_x = np.array([seg_start, seg_end, seg_end, seg_start])
        seg_y = np.array([admissions.min()] * 2 + [admissions.max()] * 2)
        seg_z = np.array([resp_score.min()] * 4)
        verts = [list(zip(seg_x, seg_y, seg_z))]
        ax.add_collection3d(
            Poly3DCollection(verts, facecolor=color, alpha=0.07,
                             edgecolor="none", zorder=0),
        )

    ax.set_xlabel("Monitoring week index", labelpad=10)
    ax.set_ylabel("Weekly admissions", labelpad=10)
    ax.set_zlabel("Respiratory residual score", labelpad=8)

    # Year annotations on the time axis
    for yi in range(2016, 2021):
        mask = val["week_start"].dt.year == yi
        if mask.any():
            idx = np.where(mask)[0][0]
            ax.text(idx, admissions.min(), resp_score.min() - 0.02,
                    str(yi), fontsize=8, ha="center", color=COLORS["gray"])

    ax.view_init(elev=20, azim=-55)
    ax.set_title(
        "3D parametric trajectory: admissions × respiratory score × time\n"
        "Colour = FluNet positivity (yellow → red); loops = influenza seasons",
        fontsize=11, pad=18,
    )

    # Colour bar for FluNet positivity
    sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                                norm=plt.Normalize(0, positivity.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, aspect=16, pad=0.08)
    cbar.set_label("FluNet positivity", fontsize=9)

    ax.text2D(0.02, 0.96,
              "Flu seasons appear as 3D loops where both admissions\n"
              "and residual scores rise simultaneously (co-activation).\n"
              "Lead-lag: residual peaks ~4 weeks before FluNet peaks.",
              transform=ax.transAxes, fontsize=8,
              color=COLORS["dark"],
              bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                        ec=COLORS["gray"], lw=0.8))

    save_pseudo(fig, "Figure2_trajectory_pseudo")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ensure_dirs()
    PSEUDO_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pseudo-figures will be saved to: {PSEUDO_FIGURES_DIR}")
    print()

    print("Generating Figure2B_pseudo (3D surface — activation terrain) ...")
    figure2B_pseudo()

    print("Generating Figure4D_pseudo (3D ribbon — subtype lag waterfall) ...")
    figure4D_pseudo()

    print("Generating Figure6_pseudo (parallel coordinates — dataset signatures) ...")
    figure6_pseudo()

    print("Generating FigureS11_pseudo (3D grouped bar — pathogen activation) ...")
    figureS11_pseudo()

    print("Generating Figure2_trajectory_pseudo (3D trajectory — bonus) ...")
    figure2_trajectory_pseudo()

    print(f"\nDone. All pseudo-figures saved to {PSEUDO_FIGURES_DIR}/")
    for f in sorted(PSEUDO_FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
