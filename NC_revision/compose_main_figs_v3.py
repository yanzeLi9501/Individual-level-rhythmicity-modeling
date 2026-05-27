"""Compose the v3 main figures and the extended S19 panel.

Outputs (NC_revision/revised_main_panels_v3/):
  - FigureS19_extended_32k_42k.{png,pdf,svg}  4-row monthly+weekly comparison
  - Figure3_v3.{png,pdf,svg}                   LGDI timeline + heatmap + 32k/42k + unified metric
  - Figure4_v3.{png,pdf,svg}                   Bars + external forest + cardiac timeline + heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).resolve().parent
LGDI_DIR = BASE / "lgdi_results"
W3_8W_WEEKLY = BASE / "DeepseekRevision" / "TestMethod" / "results_gpu" / "track2" / "W3_8w" / "W3_8w_rolling4_weekly.csv"
WEEKLY_DIR = BASE / "weekly_rdi_42k_results"
WEEKLY_EXPANDED_ONLY_DIR = BASE / "weekly_rdi_42k_expanded_only_results"
EXT_DIR = BASE / "external_positive_control_results"
UNIFIED_DIR = BASE / "unified_metric_results"
OUT_DIR = BASE / "revised_main_panels_v3"
OUT_DIR.mkdir(exist_ok=True)

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
GROUP_KEY_MAP = {"cardiovascular": "Cardiovascular", "hypertension": "Hypertension",
                 "diabetes": "Diabetes", "cerebrovascular": "Cerebrovascular",
                 "kidney": "Renal", "renal": "Renal",
                 "chronic_respiratory": "Respiratory", "respiratory": "Respiratory"}

RESP_COLOR = "#c0392b"
NEUTRAL_COLOR = "#7f8c8d"
RANK1_COLOR = "#d35400"
EVENT_COLOR = "#FFD580"
POLICY_GAP_START = pd.Timestamp("2019-07-01")
POLICY_GAP_END = pd.Timestamp("2020-04-26")
POLICY_GAP_LABEL = "Policy-induced data loss"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / f'{name}.png'}")


def _add_policy_gap(ax: plt.Axes, *, label: bool = True) -> None:
    ax.axvspan(
        POLICY_GAP_START,
        POLICY_GAP_END,
        color="#7f7f7f",
        alpha=0.18,
        lw=0,
        label=POLICY_GAP_LABEL if label else None,
    )
    midpoint = POLICY_GAP_START + (POLICY_GAP_END - POLICY_GAP_START) / 2
    ax.text(
        midpoint,
        0.5,
        POLICY_GAP_LABEL,
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=8,
        color="#4d4d4d",
        alpha=0.85,
    )


def _plot_policy_break(ax: plt.Axes, frame: pd.DataFrame, date_col: str, value_col: str, **kwargs) -> None:
    dates = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")
    for idx, segment in enumerate([dates < POLICY_GAP_START, dates > POLICY_GAP_END]):
        plot_kwargs = dict(kwargs)
        if idx > 0:
            plot_kwargs["label"] = None
        ax.plot(dates[segment], values[segment], **plot_kwargs)


def _read_expanded_only(name: str) -> pd.DataFrame | None:
    path = WEEKLY_EXPANDED_ONLY_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


# ============================================================
#  Extended S19  — 4 rows
# ============================================================

def build_extended_s19() -> None:
    # 32k WHU LGDI weekly
    whu_w = pd.read_csv(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv")
    whu_w["wd"] = pd.to_datetime(whu_w["window_anchor"])
    whu_base = whu_w[whu_w["valid"] & (whu_w["wd"].between("2016-01-01", "2018-12-31"))]
    whu_thr_w = float(whu_base["lgdi"].mean() + 1.5 * whu_base["lgdi"].std())

    # 32k WHU LGDI monthly = aggregate weekly to month
    whu_m = whu_w[whu_w["valid"]].copy()
    whu_m["month"] = whu_m["wd"].values.astype("datetime64[M]")
    whu_m = whu_m.groupby("month", as_index=False).agg(
        lgdi=("lgdi", "mean"), n_admissions=("n_admissions", "sum"))
    whu_m["wd"] = pd.to_datetime(whu_m["month"])
    whu_base_m = whu_m[whu_m["wd"].between("2016-01-01", "2018-12-31")]
    whu_thr_m = float(whu_base_m["lgdi"].mean() + 1.5 * whu_base_m["lgdi"].std())

    # 42k cardiac monthly / weekly
    c_m = pd.read_csv(WEEKLY_DIR / "weekly_rdi_42k_monthly.csv")
    c_w = pd.read_csv(WEEKLY_DIR / "weekly_rdi_42k_rolling4_weekly.csv")
    c_m["wd"] = pd.to_datetime(c_m["window_end"])
    c_w["wd"] = pd.to_datetime(c_w["window_end"])
    c_m_expanded_only = _read_expanded_only("weekly_rdi_42k_expanded_only_monthly.csv")
    c_w_expanded_only = _read_expanded_only("weekly_rdi_42k_expanded_only_rolling4_weekly.csv")
    if c_m_expanded_only is not None:
        c_m_expanded_only["wd"] = pd.to_datetime(c_m_expanded_only["window_end"])
        c_m_expanded_only = c_m_expanded_only[c_m_expanded_only["valid"]].copy()
    if c_w_expanded_only is not None:
        c_w_expanded_only["wd"] = pd.to_datetime(c_w_expanded_only["window_end"])
        c_w_expanded_only = c_w_expanded_only[c_w_expanded_only["valid"]].copy()
    summary = json.loads((WEEKLY_DIR / "weekly_rdi_42k_summary.json").read_text(encoding="utf-8"))
    c_thr_m = summary["baseline"]["thresholds"]["monthly"]["rdi_mean_plus_1_5sd"]
    c_thr_w = summary["baseline"]["thresholds"]["rolling4_weekly"]["rdi_mean_plus_1_5sd"]
    ev_s = pd.Timestamp(summary["event_definition"]["event_start"])
    ev_e = pd.Timestamp(summary["event_definition"]["event_end"])

    # WHU event = influenza event windows
    whu_val = pd.read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv")
    whu_val["wd"] = pd.to_datetime(whu_val["week_start"])
    whu_events = whu_val[whu_val["flu_event_window"].astype(bool)]["wd"].tolist()

    rows = [
        ("A. WHU primary cohort (32 056 records) - weekly LGDI",
         whu_w, "lgdi", whu_thr_w, whu_events, "week", None),
        ("B. WHU primary cohort (32 056 records) - monthly LGDI (averaged)",
         whu_m, "lgdi", whu_thr_m, whu_events, "week", None),
        ("C. Cardiac cohort (42 795 records) - weekly RDI (4-wk rolling)",
         c_w[c_w["valid"]], "rdi", c_thr_w, [(ev_s, ev_e)], "event_span", c_w_expanded_only),
        ("D. Cardiac cohort (42 795 records) - monthly RDI",
         c_m[c_m["valid"]], "rdi", c_thr_m, [(ev_s, ev_e)], "event_span", c_m_expanded_only),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.2), sharex=False)
    for ax, (title, df, col, thr, events, ev_mode, expanded_only_df) in zip(axes, rows):
        # event shading
        if ev_mode == "event_span":
            for s, e in events:
                ax.axvspan(s, e, color=EVENT_COLOR, alpha=0.5, lw=0,
                           label="Dec 2022 – Jan 2023 reopening window")
            _add_policy_gap(ax)
        else:
            shown = False
            for w in events:
                ax.axvspan(w, w + pd.Timedelta(days=7), color=EVENT_COLOR, alpha=0.45, lw=0,
                           label="FluNet event week" if not shown else None)
                shown = True
        if "wd" not in df.columns:
            df = df.copy(); df["wd"] = pd.to_datetime(df["window_end"])
        if ev_mode == "event_span":
            _plot_policy_break(ax, df, "wd", col, color="#2c3e50", lw=0.95, label="Full 42k cardiac validation")
            if expanded_only_df is not None and not expanded_only_df.empty:
                _plot_policy_break(
                    ax,
                    expanded_only_df,
                    "wd",
                    col,
                    color="#8e44ad",
                    lw=0.9,
                    ls="--",
                    alpha=0.9,
                    label="Expanded-only validation (model-unseen records)",
                )
        else:
            ax.plot(df["wd"], df[col], color="#2c3e50", lw=0.95)
        ax.axhline(thr, color=RESP_COLOR, lw=0.9, ls="--",
                   label=f"Alert threshold (μ+1.5σ = {thr:.3f})")
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel(col.upper())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDD", lw=0.5, alpha=0.7)
        ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Window end date")
    fig.suptitle("FigureS19. Monthly vs weekly respiratory-dominance signal in two cohorts",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    _save(fig, "FigureS19_extended_32k_42k")


# ============================================================
#  Figure 3 v3 — 6 panels (A: LGDI timeline · B: per-group heatmap ·
#  C: 32k+42k joint view · D: unified-metric ROC · E: alert bar comparison ·
#  F: lead time analysis)
# ============================================================

def _whu_baseline(df: pd.DataFrame) -> dict[str, float]:
    base = df[df["valid"] & pd.to_datetime(df["window_anchor"]).between("2016-01-01", "2018-12-31")]
    return {
        "lgdi_mean": float(base["lgdi"].mean()),
        "lgdi_std":  float(base["lgdi"].std()),
        "resp_mean": float(base["resp_score"].mean()),
        "resp_std":  float(base["resp_score"].std()),
    }


def build_figure3_v3() -> None:
    weekly = pd.read_csv(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv")
    weekly["wd"] = pd.to_datetime(weekly["window_anchor"])
    val = pd.read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv")
    val["wd"] = pd.to_datetime(val["week_start"])
    unified = pd.read_csv(UNIFIED_DIR / "metric_comparison.csv")
    per_week = pd.read_csv(UNIFIED_DIR / "per_week_scores.csv")
    per_week["wd"] = pd.to_datetime(per_week["week_start"])

    base = _whu_baseline(weekly)
    thr_lgdi = base["lgdi_mean"] + 1.5 * base["lgdi_std"]
    thr_resp  = base["resp_mean"] + 1.5 * base["resp_std"]

    fig = plt.figure(figsize=(16, 11.5))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.32,
                           left=0.06, right=0.98, top=0.95, bottom=0.06,
                           height_ratios=[1.0, 1.0, 1.0],
                           width_ratios=[1.0, 1.0, 1.0])

    # ── Panel A: LGDI + Pearson Profile Corr. timeline 2019-2020 with FluNet shading ──
    ax_a = fig.add_subplot(gs[0, :2])
    win = weekly[weekly["wd"].between("2019-01-01", "2020-05-31") & weekly["valid"]]
    ax_a.plot(win["wd"], win["lgdi"], color="#2c3e50", lw=1.2, label="4-week LGDI (baseline)")
    ax_a.plot(win["wd"], win["resp_score"], color="#1F77B4", lw=1.0, ls="--", alpha=0.85,
              label="Pearson Profile Corr.")
    ax_a.axhline(thr_lgdi, color=RESP_COLOR, lw=0.9, ls="--",
                 label=f"LGDI threshold \u03bc+1.5\u03c3 = {thr_lgdi:.3f}")
    ax_a.axhline(thr_resp, color="#1F77B4", lw=0.7, ls=":",
                 label=f"Pearson threshold \u03bc+1.5\u03c3 = {thr_resp:.3f}")
    # 8-week LGDI overlay
    if W3_8W_WEEKLY.exists():
        w8 = pd.read_csv(W3_8W_WEEKLY)
        w8["wd"] = pd.to_datetime(w8["window_anchor"])
        win8 = w8[w8["wd"].between("2019-01-01", "2020-05-31") & w8["valid"]]
        ax_a.plot(win8["wd"], win8["lgdi"], color="#9b59b6", lw=1.1, ls="--", alpha=0.65,
                  label="8-week LGDI (S2 sens=0.627)")
    # FluNet shading
    flu_first = True
    for _, r in val[val["flu_event_window"].astype(bool) & val["wd"].between("2019-01-01", "2020-05-31")].iterrows():
        ax_a.axvspan(r["wd"], r["wd"] + pd.Timedelta(days=7), color=EVENT_COLOR, alpha=0.4,
                     lw=0, label="FluNet event week" if flu_first else None); flu_first = False
    ax_a.axvline(pd.Timestamp("2020-01-23"), color="black", lw=0.6, ls=":")
    ax_a.text(pd.Timestamp("2020-01-23"), ax_a.get_ylim()[1] * 0.95, "Pandemic onset",
              rotation=90, fontsize=7, va="top", color="black")
    ax_a.set_title("A. WHU primary: weekly LGDI vs Pearson Profile Corr. (2019\u20132020)",
                   loc="left", fontweight="bold")
    ax_a.set_ylabel("Score (LGDI / Pearson Profile Corr.)")
    ax_a.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax_a.grid(axis="y", color="#DDD", lw=0.5, alpha=0.7)

    # ── Panel B: Per-group z-score heatmap 2019-2020 ──
    ax_b = fig.add_subplot(gs[0, 2])
    heat_rows = []
    for g in GROUPS:
        col = f"score_{g}"
        if col not in weekly.columns: continue
        base_vals = weekly.loc[
            weekly["valid"] & pd.to_datetime(weekly["window_anchor"]).between("2016-01-01", "2018-12-31"),
            col].dropna()
        mu = base_vals.mean(); sd = base_vals.std() or 1.0
        z = (win[col] - mu) / sd
        heat_rows.append(z.values)
    H = np.array(heat_rows)
    cmap = LinearSegmentedColormap.from_list("rwb", ["#2c5aa0", "white", "#c0392b"], N=256)
    im = ax_b.imshow(H, aspect="auto", cmap=cmap, vmin=-3, vmax=3,
                     extent=[mdates.date2num(win["wd"].min()), mdates.date2num(win["wd"].max()),
                             len(GROUPS) - 0.5, -0.5])
    ax_b.set_yticks(range(len(GROUPS))); ax_b.set_yticklabels(GROUPS, fontsize=8)
    ax_b.xaxis_date(); ax_b.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_b.get_xticklabels(), rotation=35, ha="right", fontsize=7)
    cb = plt.colorbar(im, ax=ax_b, shrink=0.85, pad=0.02)
    cb.set_label("z-score vs 2016–2018", fontsize=7)
    ax_b.set_title("Per-group residual z-score", loc="left", fontweight="bold")
    ax_b.text(-0.06, 1.04, "B", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)

    # ── Panel C: 32k vs 42k joint comparison — WHU Pearson + LGDI vs Cardiac Pearson RDI ──
    ax_c = fig.add_subplot(gs[1, :])
    whu_full = weekly[weekly["valid"]]
    ax_c.plot(whu_full["wd"], whu_full["resp_score"], color="#1F77B4", lw=0.9, label="WHU 32k weekly Pearson (resp_score)")
    ax_c.plot(whu_full["wd"], whu_full["lgdi"], color="#1F77B4", lw=0.7, ls="--", alpha=0.5, label="WHU 32k weekly LGDI")
    c_w = pd.read_csv(WEEKLY_DIR / "weekly_rdi_42k_rolling4_weekly.csv")
    c_w = c_w[c_w["valid"]].copy(); c_w["wd"] = pd.to_datetime(c_w["window_end"])
    _plot_policy_break(ax_c, c_w, "wd", "rdi", color="#D62728", lw=0.9, label="Cardiac 42k weekly RDI")
    c_w_expanded_only = _read_expanded_only("weekly_rdi_42k_expanded_only_rolling4_weekly.csv")
    if c_w_expanded_only is not None:
        c_w_expanded_only = c_w_expanded_only[c_w_expanded_only["valid"]].copy()
        c_w_expanded_only["wd"] = pd.to_datetime(c_w_expanded_only["window_end"])
        _plot_policy_break(
            ax_c,
            c_w_expanded_only,
            "wd",
            "rdi",
            color="#8e44ad",
            lw=0.85,
            ls="--",
            label="Cardiac expanded-only RDI",
        )
    ax_c.axhline(thr_resp, color="#1F77B4", lw=0.6, ls=":", alpha=0.7)
    ax_c.axhline(0.6357, color="#D62728", lw=0.6, ls=":", alpha=0.7)
    _add_policy_gap(ax_c)
    ax_c.axvspan(pd.Timestamp("2022-12-01"), pd.Timestamp("2023-01-31"),
                 color=EVENT_COLOR, alpha=0.4, lw=0, label="Dec 2022 reopening event")
    ax_c.set_title("C. WHU primary (Pearson + LGDI) vs cardiac 42k (Pearson RDI) \u2014 same-metric-family comparison",
                   loc="left", fontweight="bold")
    ax_c.set_ylabel("Pearson Profile Corr. / LGDI"); ax_c.set_xlabel("Window end date")
    ax_c.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax_c.grid(axis="y", color="#DDD", lw=0.5, alpha=0.7)
    ax_c.spines[["top", "right"]].set_visible(False)

    # ── Panel D: Unified-metric ROC ──
    ax_d = fig.add_subplot(gs[2, 0])
    from sklearn.metrics import roc_auc_score, roc_curve
    y_true = per_week["flu_event_window"].astype(int).values
    for name, score, color in [
        ("Pearson Profile Corr.", per_week["z_cosine"].values, "#1F77B4"),
        ("LGDI-only", per_week["z_lgdi"].values, "#D62728"),
        ("unified avg", per_week["unified_zsum"].values, "#2CA02C"),
    ]:
        if not np.isfinite(score).all():
            mask = np.isfinite(score)
            fpr, tpr, _ = roc_curve(y_true[mask], score[mask])
            auc = roc_auc_score(y_true[mask], score[mask])
        else:
            fpr, tpr, _ = roc_curve(y_true, score)
            auc = roc_auc_score(y_true, score)
        ax_d.plot(fpr, tpr, lw=1.4, color=color, label=f"{name} (AUC={auc:.3f})")
    ax_d.plot([0, 1], [0, 1], color="#888", lw=0.7, ls="--")
    ax_d.set_xlabel("False-positive rate"); ax_d.set_ylabel("Sensitivity")
    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1.02); ax_d.set_aspect("equal")
    ax_d.legend(fontsize=7, loc="lower right", frameon=False)
    ax_d.set_title("Pearson Profile Corr. vs LGDI ROC (FluNet weeks)", loc="left", fontweight="bold", pad=12)
    ax_d.text(-0.12, 1.08, "D", transform=ax_d.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax_d.grid(color="#DDD", lw=0.5, alpha=0.6)

    # ── Panel E: AUC bar ──
    ax_e = fig.add_subplot(gs[2, 1])
    auc_order = ["cosine_only", "lgdi_only", "unified_zsum", "unified_zmax",
                 "unified_and", "unified_or"]
    rows2 = unified.set_index("strategy").reindex(auc_order)
    x2 = np.arange(len(rows2))
    colors_aux = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF"]
    ax_e.bar(x2, rows2["auc"], color=colors_aux)
    ax_e.set_xticks(x2)
    ax_e.set_xticklabels(["Pearson", "LGDI", "U-Avg", "U-Max", "U-AND", "U-OR"], rotation=15)
    ax_e.axhline(0.5, color="#888", lw=0.7, ls="--")
    ax_e.set_ylim(0.40, 0.65)
    ax_e.set_ylabel("ROC-AUC")
    ax_e.set_title("ROC-AUC across strategies", loc="left", fontweight="bold", pad=12)
    ax_e.text(-0.12, 1.08, "E", transform=ax_e.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax_e.grid(axis="y", color="#DDD", lw=0.5, alpha=0.6)
    for i, v in enumerate(rows2["auc"]):
        ax_e.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)

    fig.suptitle("Figure 3. WHU primary LGDI and validation metric comparison",
                 fontsize=12, y=0.99, fontweight="bold")
    _save(fig, "Figure3_v3")


# ============================================================
#  Figure 4 v3 — Demote Venn; promote bars + forest + cardiac timeline + heatmap
# ============================================================

def build_figure4_v3() -> None:
    cardiac = json.loads((BASE / "revised_cardiac_validation_results.json").read_text(encoding="utf-8"))
    sa = cardiac["expanded_csv_wide_validation"]["sentinel_analysis"]
    h1 = sa["H1_2019_prepandemic"]["similarities"]
    q4 = sa["Q4_2022_reopening"]["similarities"]
    weekly = pd.read_csv(WEEKLY_DIR / "weekly_rdi_42k_rolling4_weekly.csv")
    weekly = weekly[weekly["valid"]].copy()
    weekly["wd"] = pd.to_datetime(weekly["window_end"])
    summary = json.loads((WEEKLY_DIR / "weekly_rdi_42k_summary.json").read_text(encoding="utf-8"))
    thr_w = summary["baseline"]["thresholds"]["rolling4_weekly"]["rdi_mean_plus_1_5sd"]
    ev_s = pd.Timestamp(summary["event_definition"]["event_start"])
    ev_e = pd.Timestamp(summary["event_definition"]["event_end"])

    fig = plt.figure(figsize=(15.5, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.06, right=0.98, top=0.94, bottom=0.07)

    # A: bars H1 vs Q4
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(GROUPS)); w = 0.4
    h1_v = [h1[g] for g in GROUPS]; q4_v = [q4[g] for g in GROUPS]
    b1 = ax_a.bar(x - w/2, h1_v, w, color="#7fb3d5", label="H1 2019 (pre-pandemic)", edgecolor="white")
    b2 = ax_a.bar(x + w/2, q4_v, w, color="#e59866", label="Q4 2022 (reopening)", edgecolor="white")
    ri = GROUPS.index("Respiratory")
    b1[ri].set_color("#1f618d"); b2[ri].set_color("#a04000")
    ax_a.set_xticks(x); ax_a.set_xticklabels(GROUPS, rotation=20, ha="right")
    ax_a.set_ylabel("Pearson similarity vs COVID reference")
    ax_a.set_title("A. Cardiac 42 795 cohort: respiratory dominance emerges in Q4 2022",
                   loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")
    ax_a.spines[["top", "right"]].set_visible(False)
    for vals, off in [(h1_v, -w/2), (q4_v, w/2)]:
        order = np.argsort(vals)[::-1]; rank = {i: r+1 for r, i in enumerate(order)}
        for i, v in enumerate(vals):
            ax_a.text(i + off, v + 0.02 if v >= 0 else v - 0.04, f"#{rank[i]}",
                      ha="center", fontsize=7, color="#34495e")
    ax_a.set_ylim(0, 1.10)

    # B: external 3-dataset forest plot
    ax_b = fig.add_subplot(gs[0, 1])
    files = [
        ("MIMIC-IV (n=1 964 influenza)", EXT_DIR / "mimic_influenza_profile_correlations.csv"),
        ("eICU-CRD (n=259 viral pneumonia)", EXT_DIR / "eicu_viral_pneumonia_profile_correlations.csv"),
        ("NWICU (n=3 315 COVID+)", EXT_DIR / "nwicu_covid_profile_correlations.csv"),
    ]
    n_groups = len(GROUPS); y_base = np.arange(n_groups)[::-1]
    offsets = np.linspace(-0.25, 0.25, len(files))
    markers = ["o", "s", "D"]
    colors_ds = ["#1F77B4", "#2CA02C", "#D62728"]
    for fi, (title, path) in enumerate(files):
        df = pd.read_csv(path)
        df["dg"] = df["group"].map(GROUP_KEY_MAP)
        df = df.set_index("dg").reindex(GROUPS).reset_index()
        rho = df["pearson_profile_correlation"].values
        lo = df["bootstrap_ci_low"].values; hi = df["bootstrap_ci_high"].values
        for i, (r, l, h) in enumerate(zip(rho, lo, hi)):
            ax_b.errorbar(r, y_base[i] + offsets[fi],
                          xerr=[[r - l], [h - r]], fmt=markers[fi],
                          color=colors_ds[fi], ecolor=colors_ds[fi],
                          elinewidth=0.9, capsize=2, markersize=4,
                          label=title if i == 0 else None)
    ax_b.axvline(0, color="black", lw=0.5)
    ax_b.set_yticks(y_base); ax_b.set_yticklabels(GROUPS)
    ax_b.set_xlabel("Pearson r vs reference profile (95 % bootstrap CI)")
    ax_b.set_title("B. External positive controls: respiratory-group dominance is method-dependent (not universal)",
                   loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7, loc="lower right")
    ax_b.spines[["top", "right"]].set_visible(False)

    # C: cardiac weekly RDI (Pearson) + LGDI dual-axis timeline zoom 2021-2024
    ax_c = fig.add_subplot(gs[1, 0])
    zoom = weekly[weekly["wd"].between("2021-06-01", "2024-12-31")]
    ax_c.plot(zoom["wd"], zoom["rdi"], color="#2c3e50", lw=1.0, label="Full 42k RDI (Pearson)")
    expanded_only = _read_expanded_only("weekly_rdi_42k_expanded_only_rolling4_weekly.csv")
    if expanded_only is not None:
        expanded_only = expanded_only[expanded_only["valid"]].copy()
        expanded_only["wd"] = pd.to_datetime(expanded_only["window_end"])
        expanded_only = expanded_only[expanded_only["wd"].between("2021-06-01", "2024-12-31")]
        ax_c.plot(expanded_only["wd"], expanded_only["rdi"], color="#8e44ad", lw=0.9, ls="--", label="Expanded-only RDI (Pearson)")
    ax_c.axhline(thr_w, color=RESP_COLOR, lw=0.9, ls="--",
                 label=f"Alert threshold \u03bc+1.5\u03c3 = {thr_w:.3f}")
    ax_c.axvspan(ev_s, ev_e, color=EVENT_COLOR, alpha=0.5, lw=0, label="Dec 2022 reopening")
    alert = zoom[zoom["rdi"] >= thr_w]
    ax_c.scatter(alert["wd"], alert["rdi"], s=16, color=RESP_COLOR, zorder=3, label="Alert weeks (RDI)")
    # Secondary y-axis: cardiac 42k LGDI for direct comparison
    lgdi_c = pd.read_csv(LGDI_DIR / "lgdi_rolling4_weekly.csv")
    lgdi_c = lgdi_c[lgdi_c["valid"]].copy()
    lgdi_c["wd"] = pd.to_datetime(lgdi_c["window_end"])
    lgdi_zoom_c = lgdi_c[lgdi_c["wd"].between("2021-06-01", "2024-12-31")]
    ax_c2 = ax_c.twinx()
    if len(lgdi_zoom_c):
        ax_c2.plot(lgdi_zoom_c["wd"], lgdi_zoom_c["lgdi"], color="#27ae60", lw=0.85,
                   ls=(0, (3, 2)), label="Cardiac 42k LGDI")
        ax_c2.set_ylabel("LGDI scalar", color="#27ae60", fontsize=8)
        ax_c2.tick_params(axis="y", labelcolor="#27ae60", labelsize=7)
        ax_c2.axhline(0, color="#27ae60", lw=0.4, ls=":", alpha=0.5)
        h2, l2 = ax_c2.get_legend_handles_labels()
    else:
        h2, l2 = [], []
    ax_c.set_ylabel("Weekly RDI (Pearson, 4-wk rolling)")
    ax_c.set_xlabel("Window end date")
    ax_c.set_title("C. Cardiac 42k: Pearson RDI vs LGDI 2021\u20132024",
                   loc="left", fontweight="bold")
    h1, l1 = ax_c.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax_c.spines[["top"]].set_visible(False)
    ax_c.grid(axis="y", color="#DDD", lw=0.5, alpha=0.7)

    # D: per-group similarity heatmap (z-score) 2022-2024 — using sim_<group> columns
    ax_d = fig.add_subplot(gs[1, 1])
    base_full = pd.read_csv(WEEKLY_DIR / "weekly_rdi_42k_rolling4_weekly.csv")
    base_full = base_full[base_full["valid"]].copy()
    base_full["wd"] = pd.to_datetime(base_full["window_end"])
    base_period = base_full[base_full["wd"].between("2016-01-01", "2018-12-31")]
    zh = base_full[base_full["wd"].between("2021-06-01", "2024-12-31")]
    H = []
    for g in GROUPS:
        col = f"sim_{g}"
        if col not in base_full.columns: H.append(np.zeros(len(zh))); continue
        mu = base_period[col].mean(); sd = base_period[col].std() or 1.0
        H.append(((zh[col] - mu) / sd).values)
    H = np.array(H)
    cmap = LinearSegmentedColormap.from_list("rwb2", ["#2c5aa0", "white", "#c0392b"], N=256)
    im = ax_d.imshow(H, aspect="auto", cmap=cmap, vmin=-3, vmax=3,
                     extent=[mdates.date2num(zh["wd"].min()), mdates.date2num(zh["wd"].max()),
                             len(GROUPS) - 0.5, -0.5])
    ax_d.set_yticks(range(len(GROUPS))); ax_d.set_yticklabels(GROUPS, fontsize=8)
    ax_d.xaxis_date(); ax_d.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_d.get_xticklabels(), rotation=35, ha="right", fontsize=7)
    cb = plt.colorbar(im, ax=ax_d, shrink=0.85, pad=0.02)
    cb.set_label("z vs 2016–2018", fontsize=7)
    ax_d.set_title("D. Cardiac cohort: per-group profile-similarity z-scores 2021–2024",
                   loc="left", fontweight="bold")

    fig.suptitle("Figure 4. Same-health-system cardiac overlap audit and external respiratory-infection validation",
                 fontsize=12, y=0.99, fontweight="bold")
    _save(fig, "Figure4_v3")


def build_figure_s18_hirid() -> None:
    """Supplementary Figure S18 — HiRID (Switzerland ICU) European geographic
    positive control. Single-panel forest plot of Pearson profile correlations
    of chronic-disease groups against the APACHE respiratory reference, with
    1000-bootstrap 95% CI. Mirrors Figure 4 Panel B layout but uses only HiRID
    data (single dataset).
    """
    hirid_csv = EXT_DIR / "hirid_respiratory_profile_correlations.csv"
    if not hirid_csv.exists():
        print(f"  [skip] FigureS18: {hirid_csv.name} not present; "
              f"run run_hirid_external_validation.py first")
        return

    df = pd.read_csv(hirid_csv)
    df["dg"] = df["group"].map(GROUP_KEY_MAP)
    df = df.set_index("dg").reindex(GROUPS).reset_index()

    fig = plt.figure(figsize=(7.5, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    y = np.arange(len(GROUPS))[::-1]
    rho = df["pearson_profile_correlation"].values
    lo = df["bootstrap_ci_low"].values
    hi = df["bootstrap_ci_high"].values
    color = "#FF7F0E"  # orange for European cohort
    n_ref = int(df["n_reference"].dropna().iloc[0]) if df["n_reference"].notna().any() else 0
    label = f"HiRID ICU (n={n_ref:,} APACHE respiratory, Switzerland)"
    for i, (r, l, h) in enumerate(zip(rho, lo, hi)):
        if np.isnan(r):
            ax.text(0.0, y[i], "  n.a.", color="#777", va="center", fontsize=8)
            continue
        ax.errorbar(r, y[i], xerr=[[r - l], [h - r]], fmt="^",
                    color=color, ecolor=color, elinewidth=1.0,
                    capsize=3, markersize=6,
                    label=label if i == 0 else None)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(GROUPS)
    ax.set_xlabel("Pearson r vs HiRID APACHE-respiratory reference profile (95% bootstrap CI)")
    ax.set_title("S18. HiRID ICU (Bern, Switzerland) — European geographic positive control",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.02,
             "Note: HiRID applies per-patient random date offsets; seasonal "
             "FluNet time-series validation is not applicable. Cross-admission "
             "patient linkage is not possible, so LGDI gap-based metrics were "
             "not computed for this cohort.",
             fontsize=7, color="#444", wrap=True)
    fig.suptitle("Figure S18. European ICU positive control (HiRID)",
                 fontsize=11, y=1.02, fontweight="bold")
    _save(fig, "FigureS18_hirid_european_validation")


def main() -> None:
    print("Building Figure 3 v3 ...")
    build_figure3_v3()
    print("Building Figure 4 v3 ...")
    build_figure4_v3()
    print("Building Figure S18 (HiRID) ...")
    build_figure_s18_hirid()
    print("Done.")


if __name__ == "__main__":
    main()
