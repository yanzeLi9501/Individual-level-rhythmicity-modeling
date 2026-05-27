"""
run_lgdi_lag_season_restricted.py
----------------------------------
Season-restricted lag-Spearman analysis for WHU-32k FluNet validation.

Rationale: The full-year analysis (all 213 weeks) includes 87 May-Oct
non-influenza weeks where resp_score sits at baseline floor (near zero)
and FluNet positivity is also zero. These correlated zeros reduce the
rank correlation spread for short lags (0-2), creating a floor effect.
Restricting to NH respiratory season (Sep-Apr, months 9-4) removes this
bias and provides a cleaner test of lead-lag structure within the period
when influenza can actually circulate.

Output: lgdi_results/lgdi_whu_lag_spearman_season_restricted.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "lgdi_results"
VALIDATION_CSV = RESULTS / "lgdi_whu_influenza_validation.csv"
OUTPUT_FULL = RESULTS / "lgdi_whu_lag_spearman_bootstrap.csv"
OUTPUT_SEASON = RESULTS / "lgdi_whu_lag_spearman_season_restricted.csv"

N_BOOT = 1000
RNG = np.random.default_rng(42)

# NH respiratory season months (Sep-Apr inclusive)
SEASON_MONTHS = {9, 10, 11, 12, 1, 2, 3, 4}


def bootstrap_lag_spearman(
    score: np.ndarray,
    signal: np.ndarray,
    lag: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    if lag == 0:
        x, y = score, signal
    else:
        x, y = score[:-lag], signal[lag:]
    m = len(x)
    if m < 10:
        return {
            "lag_weeks": lag,
            "n_pairs": m,
            "rho_point": float("nan"),
            "p_value": float("nan"),
            "rho_low_2_5": float("nan"),
            "rho_high_97_5": float("nan"),
        }
    rho_point, p_value = spearmanr(x, y)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, m, size=m)
        r, _ = spearmanr(x[ii], y[ii])
        rhos[b] = r if np.isfinite(r) else 0.0
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {
        "lag_weeks": int(lag),
        "n_pairs": int(m),
        "rho_point": float(rho_point),
        "p_value": float(p_value),
        "rho_low_2_5": float(lo),
        "rho_high_97_5": float(hi),
    }


def main() -> None:
    df = pd.read_csv(VALIDATION_CSV, parse_dates=["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    # Season filter: NH respiratory season (Sep-Apr)
    df_season = df[df["week_start"].dt.month.isin(SEASON_MONTHS)].reset_index(drop=True)
    n_all = len(df)
    n_season = len(df_season)
    n_event_all = int(df["flu_event_window"].sum())
    n_event_season = int(df_season["flu_event_window"].sum())

    print(f"Full-year:       {n_all} weeks, {n_event_all} event weeks")
    print(f"Season-restricted: {n_season} weeks (Sep-Apr), {n_event_season} event weeks")
    print(f"Weeks excluded (May-Aug): {n_all - n_season}")

    score_all = df["resp_score"].values.astype(float)
    sig_all = df["positivity"].values.astype(float)

    score_seas = df_season["resp_score"].values.astype(float)
    sig_seas = df_season["positivity"].values.astype(float)

    print("\n--- Full-year bootstrap lag-Spearman (reference) ---")
    rows_all = []
    for lag in range(5):
        r = bootstrap_lag_spearman(score_all, sig_all, lag, N_BOOT, RNG)
        rows_all.append(r)
        sig_str = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "NS")
        print(
            f"  lag-{lag}: rho={r['rho_point']:.3f}  p={r['p_value']:.4f} {sig_str}"
            f"  CI=[{r['rho_low_2_5']:.3f}, {r['rho_high_97_5']:.3f}]  n={r['n_pairs']}"
        )

    print("\n--- Season-restricted (Sep-Apr) bootstrap lag-Spearman ---")
    rows_season = []
    for lag in range(5):
        r = bootstrap_lag_spearman(score_seas, sig_seas, lag, N_BOOT, RNG)
        rows_season.append(r)
        sig_str = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "NS")
        print(
            f"  lag-{lag}: rho={r['rho_point']:.3f}  p={r['p_value']:.4f} {sig_str}"
            f"  CI=[{r['rho_low_2_5']:.3f}, {r['rho_high_97_5']:.3f}]  n={r['n_pairs']}"
        )

    out_df = pd.DataFrame(rows_season)
    out_df["subset"] = "season_sep_apr"
    out_df.to_csv(OUTPUT_SEASON, index=False)
    print(f"\nSaved season-restricted results: {OUTPUT_SEASON}")

    # Also save the comparison table
    comp_rows = []
    for i, lag in enumerate(range(5)):
        ra = rows_all[i]
        rs = rows_season[i]
        comp_rows.append({
            "lag_weeks": lag,
            "full_year_rho": ra["rho_point"],
            "full_year_p": ra["p_value"],
            "full_year_n": ra["n_pairs"],
            "season_rho": rs["rho_point"],
            "season_p": rs["p_value"],
            "season_n": rs["n_pairs"],
            "season_ci_low": rs["rho_low_2_5"],
            "season_ci_high": rs["rho_high_97_5"],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(RESULTS / "lgdi_whu_lag_spearman_comparison.csv", index=False)
    print(f"Saved comparison table: {RESULTS}/lgdi_whu_lag_spearman_comparison.csv")

    summary = {
        "n_full": n_all,
        "n_season_sep_apr": n_season,
        "n_event_full": n_event_all,
        "n_event_season": n_event_season,
        "full_year": rows_all,
        "season_restricted": rows_season,
    }
    (RESULTS / "lgdi_whu_lag_spearman_season_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
