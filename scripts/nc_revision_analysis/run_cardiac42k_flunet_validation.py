#!/usr/bin/env python3
"""Cardiac 42k FluNet-anchored validation.

Applies the same PPV-optimized strategy framework used for WHU 32k
to the cardiac 42k cohort (2016-2024), computing:
  - Per-strategy metrics (PPV / Sens / FAR) for 5 strategies
  - ROC-AUC for resp_sim (Pearson) and rdi (LGDI analog)
  - Lag-Spearman correlation between resp_sim and FluNet positivity

Outputs: NC_revision/lgdi_results/lgdi_cardiac42k_influenza_summary.json
          NC_revision/lgdi_results/lgdi_cardiac42k_influenza_roc.csv
          NC_revision/lgdi_results/lgdi_cardiac42k_influenza_lag.csv
"""
from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "lgdi_results"
CARDIAC_CSV = BASE / "weekly_rdi_42k_results" / "weekly_rdi_42k_rolling4_weekly.csv"
FLUNET_CSV = BASE.parent / "external_data" / "flunet" / "flunet_china_2009_2024.csv"

GROUPS = ["Cardiovascular", "Hypertension", "Diabetes",
          "Cerebrovascular", "Renal", "Respiratory"]


# ------------------------------------------------------------------
def load_flunet() -> pd.DataFrame:
    fn = pd.read_csv(FLUNET_CSV, low_memory=False)
    fn = fn[fn["COUNTRY_CODE"] == "CHN"].copy()
    fn["week_start"] = pd.to_datetime(fn["ISO_WEEKSTARTDATE"])
    fn = fn[fn["week_start"].between("2012-01-01", "2024-12-31")]
    agg = fn.groupby("week_start", as_index=False).agg(
        inf_all=("INF_ALL", "sum"),
        spec_processed=("SPEC_PROCESSED_NB", "sum"),
    )
    agg["positivity"] = agg["inf_all"] / agg["spec_processed"].replace({0: np.nan})
    agg["positivity"] = agg["positivity"].fillna(0.0)
    return agg


def label_flu_seasons(flunet: pd.DataFrame) -> pd.DataFrame:
    pre = flunet[flunet["week_start"].dt.year.between(2012, 2019)]
    mu = pre["positivity"].mean()
    sd = pre["positivity"].std()
    threshold = mu + 1.0 * sd
    flunet = flunet.copy()
    flunet["positivity_threshold"] = threshold
    month = flunet["week_start"].dt.month
    in_season = (month >= 9) | (month <= 4)
    flunet["flu_event_window"] = (flunet["positivity"] > threshold) & in_season
    return flunet


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
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr, fpr))
    roc = pd.DataFrame({"threshold": s, "tpr": tpr, "fpr": fpr})
    return auc, roc


def lag_spearman(score: pd.Series, signal: pd.Series, max_lag: int = 4) -> list:
    rows = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x, y = score.reset_index(drop=True), signal.reset_index(drop=True)
        else:
            x = score.iloc[:-lag].reset_index(drop=True)
            y = signal.iloc[lag:].reset_index(drop=True)
        if len(x) < 5:
            rows.append((lag, len(x), float("nan"), float("nan")))
            continue
        rho = x.corr(y, method="spearman")
        n = len(x)
        if n > 2 and pd.notna(rho) and abs(rho) < 1:
            t = rho * np.sqrt((n - 2) / (1 - rho**2))
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        else:
            p = float("nan")
        rows.append((lag, int(n), float(rho), float(p)))
    return rows


def main() -> None:
    print("[Cardiac 42k] Loading weekly RDI data...")
    c = pd.read_csv(CARDIAC_CSV)
    c["wd"] = pd.to_datetime(c["window_end"])
    c = c[c["valid"]].copy()
    print(f"  rows={len(c)}  range={c.wd.min().date()} -> {c.wd.max().date()}")

    print("[Cardiac 42k] Loading FluNet China positivity...")
    flunet = load_flunet()
    flunet = label_flu_seasons(flunet)
    flu_threshold = float(flunet["positivity_threshold"].iloc[0])
    print(f"  FluNet threshold={flu_threshold:.4f}")

    # Merge on week: cardiac window_end is ISO week END (Sunday), FluNet is week START (Monday)
    # Align by: flu week_start == cardiac window_end - 6 days (or shift to nearest Monday)
    # Use merge_asof or approximate
    c["flu_week_start"] = c["wd"] - pd.Timedelta(days=6)  # window_end - 6d = window_start (Mon)
    merged = c.merge(
        flunet[["week_start", "positivity", "positivity_threshold", "flu_event_window"]],
        left_on="flu_week_start", right_on="week_start",
        how="left",
    )
    merged["positivity"] = merged["positivity"].fillna(0.0)
    merged["flu_event_window"] = merged["flu_event_window"].fillna(False).astype(bool)
    merged["month"] = merged["wd"].dt.month

    n_total = len(merged)
    n_event = int(merged["flu_event_window"].sum())
    print(f"  Merged rows={n_total}  flu event windows={n_event}")

    # Baseline 2016-2018
    base = merged[merged["wd"].between("2016-01-01", "2018-12-31")]
    resp_mu = float(base["resp_sim"].mean())
    resp_sd = float(base["resp_sim"].std())
    rdi_mu  = float(base["rdi"].mean())
    rdi_sd  = float(base["rdi"].std())
    print(f"  resp_sim baseline: mu={resp_mu:.4f} sd={resp_sd:.4f} thr1.5={resp_mu+1.5*resp_sd:.4f}")
    print(f"  rdi baseline:      mu={rdi_mu:.4f} sd={rdi_sd:.4f} thr1.5={rdi_mu+1.5*rdi_sd:.4f}")

    # Per-group baseline
    grp_mu = {g: float(base[f"sim_{g}"].mean()) for g in GROUPS}
    grp_sd = {g: float(base[f"sim_{g}"].std()) for g in GROUPS}

    labels = merged["flu_event_window"].astype(int).values
    N_EV = max(int(labels.sum()), 1)
    N_NEV = max(int((labels == 0).sum()), 1)

    def _m(pred):
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        n_ = tp + fp
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alerts": n_,
            "sensitivity": tp / N_EV,
            "ppv": tp / n_ if n_ > 0 else 0.0,
            "false_alarm_rate": fp / N_NEV,
        }

    # Strategy 1: Pearson (resp_sim) threshold only
    t1 = resp_mu + 1.5 * resp_sd
    s1 = {**_m((merged["resp_sim"] > t1).astype(int).values),
          "strategy": "REFERENCE_resp_mean_plus_1_5sd", "threshold": float(t1)}

    # Strategy 2: RDI threshold only
    t2 = rdi_mu + 1.5 * rdi_sd
    s2 = {**_m((merged["rdi"] > t2).astype(int).values),
          "strategy": "RDI_mean_plus_1_5sd", "threshold": float(t2)}

    # Strategy 3: Consensus >=2 groups (no season)
    cnt = sum((merged[f"sim_{g}"] > grp_mu[g] + 1.5 * grp_sd[g]).astype(int) for g in GROUPS)
    cons2 = (cnt >= 2).astype(int).values
    s3 = {**_m(cons2), "strategy": "consensus_2groups_mean_plus_1_5sd", "threshold": float("nan")}

    # Strategy 4: Consensus >=2 groups + Season (Oct-Apr)
    season_mask = merged["month"].isin([10, 11, 12, 1, 2, 3, 4]).values
    cons2_season = ((cnt >= 2).values & season_mask).astype(int)
    s4 = {**_m(cons2_season), "strategy": "consensus_2groups_season_oct_apr", "threshold": float("nan")}

    # Strategy 5 (OPTIMIZED): Season (Nov-Mar) + sustained 2-week + consensus >=2 groups
    nov_mar = merged["month"].isin([11, 12, 1, 2, 3]).values
    above5 = (cnt >= 2).values & nov_mar
    sustained5 = above5 & np.concatenate([[False], above5[:-1]])
    s5 = {**_m(sustained5.astype(int)),
          "strategy": "season_sustained_consensus2grp_nov_mar", "threshold": float("nan")}

    strats = [s1, s2, s3, s4, s5]
    print("\n[Per-strategy metrics]")
    for s in strats:
        print(f"  {s['strategy']:<48} ppv={s['ppv']:.3f}  sens={s['sensitivity']:.3f}  far={s['false_alarm_rate']:.3f}  tp={s['tp']}  fp={s['fp']}")

    # ROC curves
    resp_scores = merged["resp_sim"].values
    rdi_scores  = merged["rdi"].values
    auc_resp, roc_resp = compute_roc(resp_scores, labels)
    auc_rdi,  roc_rdi  = compute_roc(rdi_scores,  labels)
    print(f"\n[ROC-AUC] resp_sim={auc_resp:.4f} | rdi={auc_rdi:.4f}")
    roc_resp["score_type"] = "resp_sim"
    roc_rdi["score_type"] = "rdi"
    roc_combined = pd.concat([roc_resp, roc_rdi], ignore_index=True)

    # Lag Spearman
    ms = merged.sort_values("wd").reset_index(drop=True)
    lag_rows = lag_spearman(ms["resp_sim"], ms["positivity"], max_lag=4)
    lag_df = pd.DataFrame(lag_rows, columns=["lgdi_leads_flunet_weeks", "n_weeks", "spearman_rho", "p_value"])
    print("\n[Lag Spearman: resp_sim leads FluNet]")
    print(lag_df.to_string(index=False))

    # Event weeks by month
    ev_months = merged[merged["flu_event_window"]]["month"].value_counts().sort_index()
    print("\n[Event weeks by month]")
    print(ev_months.to_string())

    summary = {
        "cohort": "cardiac_42k",
        "cardiac_csv": str(CARDIAC_CSV),
        "flunet_csv": str(FLUNET_CSV),
        "n_weeks_total": int(n_total),
        "n_weeks_flu_event": int(n_event),
        "flunet_positivity_threshold": float(flu_threshold),
        "baseline_resp_mean": float(resp_mu),
        "baseline_resp_sd": float(resp_sd),
        "baseline_rdi_mean": float(rdi_mu),
        "baseline_rdi_sd": float(rdi_sd),
        "roc_auc_resp_sim": float(auc_resp),
        "roc_auc_rdi": float(auc_rdi),
        "per_strategy_metrics": strats,
        "lag_spearman": lag_df.to_dict(orient="records"),
        "group_baselines": {g: {"mean": grp_mu[g], "sd": grp_sd[g]} for g in GROUPS},
        "event_weeks_by_month": ev_months.to_dict(),
    }

    out_json = RESULTS / "lgdi_cardiac42k_influenza_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    roc_combined.to_csv(RESULTS / "lgdi_cardiac42k_influenza_roc.csv", index=False)
    lag_df.to_csv(RESULTS / "lgdi_cardiac42k_influenza_lag.csv", index=False)
    print(f"\n[done] wrote summary + roc + lag to {RESULTS}")


if __name__ == "__main__":
    main()
