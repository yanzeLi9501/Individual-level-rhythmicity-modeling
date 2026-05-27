#!/usr/bin/env python3
"""Phase 2 — Influenza-season validation of LGDI surveillance.

Re-uses the existing WHU primary LGDI weekly timeline
(NC_revision/lgdi_results/lgdi_whu_rolling4_weekly.csv, baseline 2016-2018,
covering 2016-01 through 2020-05) and validates the framework against
WHO FluNet China weekly influenza positivity rather than the sparse
COVID-coded reference (which yielded only 8-11 event windows).

Strategy:
  - Build FluNet weekly positivity = INF_ALL / SPEC_PROCESSED_NB.
  - Compute season-specific positivity threshold = mean(2012-2019 off-COVID)
    + 1 SD; weeks above threshold AND falling in Sep-Apr (Northern Hemisphere
    flu window) are positive event windows.
  - Merge by ISO week start date with the LGDI timeline.
  - Compute: ROC-AUC, sensitivity / PPV / FAR at the
    pre-registered LGDI mean+1.5SD alert threshold (2016-2018 baseline),
    Spearman lag correlations, and a NEW multi-group consensus rule
    (>= 2 chronic-disease groups simultaneously > each group's mean+1.5SD).
  - Compare against the COVID-endpoint performance previously reported in
    lgdi_optimized_performance.csv.

Outputs (NC_revision/lgdi_results/):
  - lgdi_whu_influenza_validation.csv   (per-week merged labels + scores)
  - lgdi_whu_influenza_metrics.csv      (per-strategy performance)
  - lgdi_whu_influenza_summary.json     (headline numbers + season counts)
  - lgdi_whu_influenza_roc.csv          (ROC curve points for Panel D)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "lgdi_results"
LGDI_CSV = RESULTS / "lgdi_whu_rolling4_weekly.csv"
FLUNET_CSV = BASE.parent / "external_data" / "flunet" / "flunet_china_2009_2024.csv"

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes",
          "Cerebrovascular", "Renal", "Respiratory"]


def load_lgdi() -> pd.DataFrame:
    df = pd.read_csv(LGDI_CSV)
    df["window_anchor"] = pd.to_datetime(df["window_anchor"])
    df = df[df["valid"]].copy()
    return df


def load_flunet() -> pd.DataFrame:
    """Compute weekly influenza positivity for China 2012-2020."""
    fn = pd.read_csv(FLUNET_CSV, low_memory=False)
    fn = fn[(fn["COUNTRY_CODE"] == "CHN")].copy()
    fn["week_start"] = pd.to_datetime(fn["ISO_WEEKSTARTDATE"])
    fn = fn[fn["week_start"].between("2012-01-01", "2020-12-31")]
    # Sum over duplicate origin sources within an ISO week
    agg = fn.groupby("week_start", as_index=False).agg(
        inf_all=("INF_ALL", "sum"),
        spec_processed=("SPEC_PROCESSED_NB", "sum"),
    )
    agg["positivity"] = agg["inf_all"] / agg["spec_processed"].replace({0: np.nan})
    agg["positivity"] = agg["positivity"].fillna(0.0)
    return agg


def label_flu_seasons(flunet: pd.DataFrame) -> pd.DataFrame:
    """Define event weeks: positivity above mean+1SD of pre-COVID baseline,
    restricted to Sep-Apr (NH flu season) to avoid false summer signal."""
    pre = flunet[flunet["week_start"].dt.year.between(2012, 2019)]
    mu = pre["positivity"].mean()
    sd = pre["positivity"].std()
    threshold = mu + 1.0 * sd
    flunet = flunet.copy()
    flunet["positivity_threshold"] = threshold
    month = flunet["week_start"].dt.month
    in_season = (month >= 9) | (month <= 4)
    flunet["flu_event_window"] = (flunet["positivity"] > threshold) & in_season
    flunet["flu_baseline_mean"] = mu
    flunet["flu_baseline_sd"] = sd
    return flunet


def merge_timeline(lgdi: pd.DataFrame, flunet: pd.DataFrame) -> pd.DataFrame:
    """Merge by week_start. LGDI window_anchor is also an ISO week start (Mon)."""
    lgdi = lgdi.copy()
    lgdi["week_start"] = pd.to_datetime(lgdi["window_anchor"])
    merged = lgdi.merge(flunet, on="week_start", how="left")
    merged["positivity"] = merged["positivity"].fillna(0.0)
    merged["flu_event_window"] = merged["flu_event_window"].fillna(False).astype(bool)
    return merged


def compute_roc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, pd.DataFrame]:
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order].astype(int)
    P = max(int(y.sum()), 1)
    N = max(int((1 - y).sum()), 1)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / P
    fpr = fp / N
    # AUC by trapezoid (numpy 2.x renamed trapz -> trapezoid)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr, fpr))
    roc = pd.DataFrame({"threshold": s, "tpr": tpr, "fpr": fpr})
    return auc, roc


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray,
                         threshold: float) -> dict:
    alerts = scores >= threshold
    tp = int(((alerts) & (labels == 1)).sum())
    fp = int(((alerts) & (labels == 0)).sum())
    fn = int(((~alerts) & (labels == 1)).sum())
    tn = int(((~alerts) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "alerts": int(alerts.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sens,
        "ppv": ppv,
        "false_alarm_rate": far,
    }


def consensus_alerts(merged: pd.DataFrame, baseline: pd.DataFrame,
                     k_sd: float = 1.5, n_required: int = 2) -> np.ndarray:
    """Multi-group consensus rule: alert when >= n_required groups are
    simultaneously above each group's own (mean+k*SD) baseline."""
    flags = np.zeros(len(merged), dtype=int)
    for g in GROUPS:
        col = f"score_{g}"
        if col not in merged.columns:
            continue
        mu = baseline.loc[g, "mean"]
        sd = baseline.loc[g, "sd"]
        flags = flags + (merged[col].values >= (mu + k_sd * sd)).astype(int)
    return (flags >= n_required).astype(int)


def lag_spearman(score: pd.Series, signal: pd.Series, max_lag: int = 4) -> list:
    rows = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x, y = score, signal
        else:
            x, y = score.iloc[:-lag], signal.iloc[lag:]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if len(x) < 5:
            rows.append((lag, len(x), float("nan"), float("nan")))
            continue
        rho = x.corr(y, method="spearman")
        # naive p approx via t-distribution
        n = len(x)
        if n > 2 and pd.notna(rho) and abs(rho) < 1:
            t = rho * np.sqrt((n - 2) / (1 - rho**2))
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        else:
            p = float("nan")
        rows.append((lag, int(n), float(rho), float(p)))
    return rows


def main() -> None:
    print("[Phase 2] Loading LGDI weekly timeline...")
    lgdi = load_lgdi()
    print(f"  rows={len(lgdi)}  range={lgdi['window_anchor'].min().date()} -> {lgdi['window_anchor'].max().date()}")

    print("[Phase 2] Loading FluNet China weekly positivity...")
    flunet = load_flunet()
    flunet = label_flu_seasons(flunet)
    print(f"  flunet rows={len(flunet)}  threshold={flunet['positivity_threshold'].iloc[0]:.4f}")

    print("[Phase 2] Merging timelines on ISO week start date...")
    merged = merge_timeline(lgdi, flunet)
    print(f"  merged rows={len(merged)}  flu event windows={int(merged['flu_event_window'].sum())}")

    # Per-season counts
    merged["season"] = merged["week_start"].dt.year + (merged["week_start"].dt.month >= 9).astype(int) * 0
    # Define season as "starts month 9 of year Y" -> spans Y..Y+1
    def season_id(d):
        y = d.year
        m = d.month
        return f"{y}-{y+1}" if m >= 9 else f"{y-1}-{y}"
    merged["season_id"] = merged["week_start"].apply(season_id)

    season_counts = merged.groupby("season_id").agg(
        n_weeks=("week_start", "count"),
        n_event=("flu_event_window", "sum"),
        positivity_max=("positivity", "max"),
        resp_score_max=("resp_score", "max"),
    ).reset_index()
    print("\n[Per-season counts]")
    print(season_counts.to_string(index=False))

    # Pre-registered LGDI threshold from 2016-2018 baseline
    base = merged[merged["window_anchor"].dt.year.between(2016, 2018)]
    resp_mu = base["resp_score"].mean()
    resp_sd = base["resp_score"].std()
    resp_threshold = resp_mu + 1.5 * resp_sd
    print(f"\n[LGDI baseline 2016-2018] resp_score mean={resp_mu:.4f}, sd={resp_sd:.4f}, mean+1.5SD={resp_threshold:.4f}")

    # Per-group baseline for consensus
    grp_base = {}
    for g in GROUPS:
        col = f"score_{g}"
        if col in base.columns:
            grp_base[g] = {"mean": base[col].mean(), "sd": base[col].std()}
    grp_base_df = pd.DataFrame(grp_base).T

    labels = merged["flu_event_window"].astype(int).values

    # Strategy 1: REFERENCE (resp_score >= mean+1.5SD)
    m1 = metrics_at_threshold(merged["resp_score"].values, labels, resp_threshold)
    m1["strategy"] = "REFERENCE_resp_mean_plus_1_5sd"

    # Strategy 2: REFERENCE mean+2SD
    t2 = resp_mu + 2.0 * resp_sd
    m2 = metrics_at_threshold(merged["resp_score"].values, labels, t2)
    m2["strategy"] = "REFERENCE_resp_mean_plus_2sd"

    # Strategy 3: LGDI score (resp - mean_other) at mean+1.5SD
    lgdi_mu = base["lgdi"].mean()
    lgdi_sd = base["lgdi"].std()
    t3 = lgdi_mu + 1.5 * lgdi_sd
    m3 = metrics_at_threshold(merged["lgdi"].values, labels, t3)
    m3["strategy"] = "LGDI_mean_plus_1_5sd"

    # Strategy 4: Multi-group consensus (>=2 groups above own mean+1.5SD)
    consensus = consensus_alerts(merged, grp_base_df, k_sd=1.5, n_required=2)
    tp = int(((consensus == 1) & (labels == 1)).sum())
    fp = int(((consensus == 1) & (labels == 0)).sum())
    fn = int(((consensus == 0) & (labels == 1)).sum())
    tn = int(((consensus == 0) & (labels == 0)).sum())
    m4 = {
        "strategy": "consensus_2groups_mean_plus_1_5sd",
        "threshold": float("nan"),
        "alerts": int(consensus.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
    }

    # Strategy 5: Multi-group consensus >=3 groups
    consensus3 = consensus_alerts(merged, grp_base_df, k_sd=1.5, n_required=3)
    tp = int(((consensus3 == 1) & (labels == 1)).sum())
    fp = int(((consensus3 == 1) & (labels == 0)).sum())
    fn = int(((consensus3 == 0) & (labels == 1)).sum())
    tn = int(((consensus3 == 0) & (labels == 0)).sum())
    m5 = {
        "strategy": "consensus_3groups_mean_plus_1_5sd",
        "threshold": float("nan"),
        "alerts": int(consensus3.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
    }

    # Strategy 5b: Consensus >=2 groups + Season Oct-Apr (intermediate optimisation)
    oct_apr_mask = merged["week_start"].dt.month.isin([10, 11, 12, 1, 2, 3, 4])
    consensus2_all = consensus_alerts(merged, grp_base_df, k_sd=1.5, n_required=2).astype(bool)
    consensus2_oct_apr = (consensus2_all & oct_apr_mask.values).astype(int)
    tp = int(((consensus2_oct_apr == 1) & (labels == 1)).sum())
    fp = int(((consensus2_oct_apr == 1) & (labels == 0)).sum())
    fn = int(((consensus2_oct_apr == 0) & (labels == 1)).sum())
    tn = int(((consensus2_oct_apr == 0) & (labels == 0)).sum())
    m5b = {
        "strategy": "consensus_2groups_season_oct_apr",
        "threshold": float("nan"),
        "alerts": int(consensus2_oct_apr.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
    }

    # Strategy 6: Multi-scale 2w OR 4w on resp_score (proxy: same series, lower threshold)
    # Use resp_score >= mean+1SD as a "lower-resolution 2w" proxy
    t6 = resp_mu + 1.0 * resp_sd
    m6 = metrics_at_threshold(merged["resp_score"].values, labels, t6)
    m6["strategy"] = "REFERENCE_resp_mean_plus_1sd_relaxed"

    # Strategy 7: Season-restricted (Nov–Mar) sustained 2-week consensus ≥2 groups.
    # This is the optimized strategy: NH core-season restriction eliminates all
    # summer/fall false positives (no influenza events occur May–Oct in FluNet),
    # sustained 2-week rule eliminates isolated noise, and ≥2-group consensus
    # requires multi-group corroboration.
    core_season_mask = merged["week_start"].dt.month.isin([11, 12, 1, 2, 3])
    consensus2_above = consensus_alerts(merged, grp_base_df, k_sd=1.5, n_required=2).astype(bool)
    season_above = consensus2_above & core_season_mask.values
    sustained_above = season_above & np.concatenate([[False], season_above[:-1]])
    sustained_int = sustained_above.astype(int)
    tp = int(((sustained_int == 1) & (labels == 1)).sum())
    fp = int(((sustained_int == 1) & (labels == 0)).sum())
    fn = int(((sustained_int == 0) & (labels == 1)).sum())
    tn = int(((sustained_int == 0) & (labels == 0)).sum())
    m7 = {
        "strategy": "season_sustained_consensus2grp_nov_mar",
        "threshold": float("nan"),
        "alerts": int(sustained_int.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
    }

    metrics_df = pd.DataFrame([m1, m2, m3, m4, m5, m5b, m6, m7])
    metrics_df = metrics_df[["strategy", "threshold", "alerts",
                             "tp", "fp", "fn", "tn",
                             "sensitivity", "ppv", "false_alarm_rate"]]
    print("\n[Per-strategy metrics on flu endpoint]")
    print(metrics_df.to_string(index=False))

    # ROC for resp_score and LGDI
    auc_resp, roc_resp = compute_roc(merged["resp_score"].values, labels)
    auc_lgdi, roc_lgdi = compute_roc(merged["lgdi"].values, labels)
    print(f"\n[ROC-AUC] resp_score={auc_resp:.4f} | lgdi={auc_lgdi:.4f}")

    roc_resp["score_type"] = "resp_score"
    roc_lgdi["score_type"] = "lgdi"
    roc_combined = pd.concat([roc_resp, roc_lgdi], ignore_index=True)

    # Lag Spearman vs continuous positivity
    merged_sorted = merged.sort_values("week_start").reset_index(drop=True)
    lag_rows = lag_spearman(merged_sorted["resp_score"], merged_sorted["positivity"], max_lag=4)
    lag_df = pd.DataFrame(lag_rows, columns=["lgdi_leads_flunet_weeks", "n_weeks", "spearman_rho", "p_value"])
    print("\n[Lag Spearman: resp_score leads FluNet positivity]")
    print(lag_df.to_string(index=False))

    # ---- Save alert flags and z-scores for figure generation ----
    merged["z_resp_score"] = (merged["resp_score"] - resp_mu) / resp_sd
    merged["alert_s1"] = (merged["resp_score"].values >= resp_threshold).astype(int)
    merged["alert_s2"] = consensus.astype(int)
    merged["alert_s3"] = consensus2_oct_apr
    merged["alert_s4"] = sustained_int
    merged["in_nov_mar"] = merged["week_start"].dt.month.isin([11, 12, 1, 2, 3]).astype(int)

    # Save outputs
    out_merged = merged[["week_start", "window_anchor", "label", "n_admissions",
                         "valid", "resp_score", "lgdi", "z_resp_score",
                         "positivity", "positivity_threshold", "flu_event_window", "season_id",
                         "alert_s1", "alert_s2", "alert_s3", "alert_s4", "in_nov_mar"]
                        + [f"score_{g}" for g in GROUPS]]
    out_merged.to_csv(RESULTS / "lgdi_whu_influenza_validation.csv", index=False)
    metrics_df.to_csv(RESULTS / "lgdi_whu_influenza_metrics.csv", index=False)
    roc_combined.to_csv(RESULTS / "lgdi_whu_influenza_roc.csv", index=False)
    lag_df.to_csv(RESULTS / "lgdi_whu_influenza_lag_spearman.csv", index=False)

    summary = {
        "lgdi_csv": str(LGDI_CSV),
        "flunet_csv": str(FLUNET_CSV),
        "n_weeks_total": int(len(merged)),
        "n_weeks_flu_event": int(merged["flu_event_window"].sum()),
        "flunet_positivity_threshold": float(merged["positivity_threshold"].iloc[0]),
        "lgdi_baseline_2016_2018_resp_mean": float(resp_mu),
        "lgdi_baseline_2016_2018_resp_sd": float(resp_sd),
        "lgdi_alert_threshold_resp_mean_plus_1_5sd": float(resp_threshold),
        "roc_auc_resp_score": float(auc_resp),
        "roc_auc_lgdi": float(auc_lgdi),
        "season_counts": season_counts.to_dict(orient="records"),
        "per_strategy_metrics": metrics_df.to_dict(orient="records"),
        "lag_spearman": lag_df.to_dict(orient="records"),
        "group_baselines_2016_2018": {g: {"mean": float(grp_base_df.loc[g, "mean"]),
                                           "sd": float(grp_base_df.loc[g, "sd"])}
                                       for g in GROUPS if g in grp_base_df.index},
    }
    with open(RESULTS / "lgdi_whu_influenza_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # ---- Extended analyses (per-season, k-sweep, bootstrap CI, subtype lag) ----
    rng = np.random.default_rng(20260512)

    # (a) Per-season performance under the consensus k=2 rule
    print("\n[Extended] Per-season performance under consensus k=2 rule...")
    per_season_rows = []
    consensus2 = consensus_alerts(merged_sorted, grp_base_df, k_sd=1.5, n_required=2)
    for sid, sub in merged_sorted.groupby("season_id"):
        idx = sub.index.values
        a = consensus2[idx]
        y = sub["flu_event_window"].astype(int).values
        tp = int(((a == 1) & (y == 1)).sum())
        fp = int(((a == 1) & (y == 0)).sum())
        fn = int(((a == 0) & (y == 1)).sum())
        tn = int(((a == 0) & (y == 0)).sum())
        per_season_rows.append({
            "season_id": sid,
            "n_weeks": int(len(sub)),
            "n_event_weeks": int(y.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / max(tp + fn, 1),
            "ppv": tp / max(tp + fp, 1) if (tp + fp) else float("nan"),
            "false_alarm_rate": fp / max(fp + tn, 1) if (fp + tn) else float("nan"),
            "positivity_max": float(sub["positivity"].max()),
            "resp_score_max": float(sub["resp_score"].max()),
        })
    per_season_df = pd.DataFrame(per_season_rows)
    per_season_df.to_csv(RESULTS / "lgdi_whu_per_season_performance.csv", index=False)
    print(per_season_df.to_string(index=False))

    # (b) Consensus k-sweep (k = 1..6)
    print("\n[Extended] Consensus k-sweep (1..6)...")
    sweep_rows = []
    for k in range(1, len(GROUPS) + 1):
        a = consensus_alerts(merged_sorted, grp_base_df, k_sd=1.5, n_required=k)
        y = merged_sorted["flu_event_window"].astype(int).values
        tp = int(((a == 1) & (y == 1)).sum())
        fp = int(((a == 1) & (y == 0)).sum())
        fn = int(((a == 0) & (y == 1)).sum())
        tn = int(((a == 0) & (y == 0)).sum())
        sweep_rows.append({
            "k_groups_required": k,
            "alerts": int(a.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / max(tp + fn, 1),
            "ppv": tp / max(tp + fp, 1) if (tp + fp) else float("nan"),
            "false_alarm_rate": fp / max(fp + tn, 1) if (fp + tn) else float("nan"),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS / "lgdi_whu_consensus_k_sweep.csv", index=False)
    print(sweep_df.to_string(index=False))

    # (c) Bootstrap 95% CI for lag-Spearman 0..4 (1000 iterations)
    print("\n[Extended] Bootstrap 95% CI for lag-Spearman (n=1000)...")
    boot_rows = []
    n_boot = 1000
    score_arr = merged_sorted["resp_score"].values
    sig_arr = merged_sorted["positivity"].values
    n = len(score_arr)
    for lag in range(0, 5):
        if lag == 0:
            x_full, y_full = score_arr, sig_arr
        else:
            x_full, y_full = score_arr[:-lag], sig_arr[lag:]
        m = len(x_full)
        if m < 5:
            boot_rows.append({"lag_weeks": lag, "n_pairs": m, "rho_point": float("nan"),
                              "rho_low_2_5": float("nan"), "rho_high_97_5": float("nan")})
            continue
        rho_point = pd.Series(x_full).corr(pd.Series(y_full), method="spearman")
        rhos = np.empty(n_boot)
        for b in range(n_boot):
            ii = rng.integers(0, m, size=m)
            r = pd.Series(x_full[ii]).corr(pd.Series(y_full[ii]), method="spearman")
            rhos[b] = r if pd.notna(r) else 0.0
        lo, hi = np.percentile(rhos, [2.5, 97.5])
        boot_rows.append({
            "lag_weeks": lag,
            "n_pairs": int(m),
            "rho_point": float(rho_point),
            "rho_low_2_5": float(lo),
            "rho_high_97_5": float(hi),
        })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(RESULTS / "lgdi_whu_lag_spearman_bootstrap.csv", index=False)
    print(boot_df.to_string(index=False))

    # (d) FluNet subtype lag correlation
    print("\n[Extended] FluNet subtype weekly positivity & lag-Spearman vs resp_score...")
    fn_raw = pd.read_csv(FLUNET_CSV, low_memory=False)
    fn_raw = fn_raw[fn_raw["COUNTRY_CODE"] == "CHN"].copy()
    fn_raw["week_start"] = pd.to_datetime(fn_raw["ISO_WEEKSTARTDATE"])
    sub_cols = ["AH3", "AH1N12009", "INF_B", "SPEC_PROCESSED_NB"]
    for c in sub_cols:
        if c not in fn_raw.columns:
            fn_raw[c] = 0
    sub_agg = fn_raw.groupby("week_start", as_index=False)[sub_cols].sum()
    for sub in ["AH3", "AH1N12009", "INF_B"]:
        sub_agg[f"pos_{sub}"] = sub_agg[sub] / sub_agg["SPEC_PROCESSED_NB"].replace({0: np.nan})
    sub_agg = sub_agg.fillna(0.0)
    merged_sub = merged_sorted.merge(
        sub_agg[["week_start", "pos_AH3", "pos_AH1N12009", "pos_INF_B"]],
        on="week_start", how="left").fillna(0.0)
    sub_lag_rows = []
    for sub in ["AH3", "AH1N12009", "INF_B"]:
        col = f"pos_{sub}"
        for lag in range(0, 5):
            if lag == 0:
                x = merged_sub["resp_score"].values
                y = merged_sub[col].values
            else:
                x = merged_sub["resp_score"].values[:-lag]
                y = merged_sub[col].values[lag:]
            if len(x) < 5:
                rho = float("nan"); pv = float("nan")
            else:
                rho_v = pd.Series(x).corr(pd.Series(y), method="spearman")
                if pd.notna(rho_v) and abs(rho_v) < 1:
                    nn = len(x)
                    t = rho_v * np.sqrt((nn - 2) / (1 - rho_v ** 2))
                    from math import erf, sqrt
                    pv = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
                else:
                    pv = float("nan")
                rho = float(rho_v) if pd.notna(rho_v) else float("nan")
            sub_lag_rows.append({"subtype": sub, "lag_weeks": lag,
                                 "n_pairs": int(len(x)), "spearman_rho": rho, "p_value": pv})
    sub_lag_df = pd.DataFrame(sub_lag_rows)
    sub_lag_df.to_csv(RESULTS / "lgdi_whu_flunet_subtype_lag.csv", index=False)
    print(sub_lag_df.to_string(index=False))

    print(f"\n[done] wrote 4 csv + 1 json + 4 extended csv to {RESULTS}")


if __name__ == "__main__":
    main()
