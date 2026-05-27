from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf


ROOT = Path(__file__).resolve().parent
REBUILD_SCRIPTS = ROOT.parent / "RebuildRevision" / "scripts"
if str(REBUILD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REBUILD_SCRIPTS))

from rebuild_common import LGDI_DIR  # noqa: E402


OUT_DIR = ROOT / "analysis_outputs"
VALIDATION = LGDI_DIR / "lgdi_whu_influenza_validation.csv"
SEED = 20260525
N_BOOT = 5000


def paired_lag(x: pd.Series, y: pd.Series, lag: int = 4) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": x.shift(lag), "y": y}).dropna()
    return frame["x"].to_numpy(float), frame["y"].to_numpy(float)


def rho_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    stat = spearmanr(x, y)
    return float(stat.statistic), float(stat.pvalue)


def moving_block_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    block_weeks: int,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> np.ndarray:
    n = len(x)
    starts = np.arange(0, n - block_weeks + 1)
    n_blocks = int(np.ceil(n / block_weeks))
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(start, start + block_weeks) for start in chosen])[:n]
        values[i] = spearmanr(x[idx], y[idx]).statistic
    return values


def block_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    block_weeks: int,
    rng: np.random.Generator,
    n_perm: int = N_BOOT,
) -> float:
    n = len(x)
    blocks = [np.arange(start, min(start + block_weeks, n)) for start in range(0, n, block_weeks)]
    values = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        order = rng.permutation(len(blocks))
        idx = np.concatenate([blocks[j] for j in order])
        values[i] = spearmanr(x[idx], y).statistic
    return float((np.sum(np.abs(values) >= abs(observed)) + 1) / (n_perm + 1))


def circular_shift_p(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator, n_perm: int = N_BOOT) -> float:
    n = len(x)
    values = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shift = int(rng.integers(1, n))
        values[i] = spearmanr(np.roll(x, shift), y).statistic
    return float((np.sum(np.abs(values) >= abs(observed)) + 1) / (n_perm + 1))


def autocorr_rows(name: str, values: pd.Series) -> list[dict[str, object]]:
    series = values.astype(float).dropna()
    acf_values = acf(series.to_numpy(), nlags=8, fft=False)
    lb = acorr_ljungbox(series, lags=[4, 8, 12, 16], return_df=True)
    rows: list[dict[str, object]] = []
    for lag in [4, 8, 12, 16]:
        rows.append(
            {
                "series": name,
                "n_weeks": int(series.shape[0]),
                "lag": lag,
                "ljung_box_q": float(lb.loc[lag, "lb_stat"]),
                "ljung_box_p": float(lb.loc[lag, "lb_pvalue"]),
                "acf_lag1": float(acf_values[1]),
                "acf_lag4": float(acf_values[4]),
                "acf_lag8": float(acf_values[8]),
            }
        )
    return rows


def analyze_pair(
    label: str,
    x: pd.Series,
    y: pd.Series,
    block_weeks: int,
    seed_offset: int,
) -> dict[str, object]:
    x_lag, y_aligned = paired_lag(x, y, lag=4)
    observed, standard_p = rho_p(x_lag, y_aligned)
    rng = np.random.default_rng(SEED + seed_offset)
    boot = moving_block_bootstrap(x_lag, y_aligned, block_weeks, rng)
    ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
    if observed >= 0:
        sign_p = 2 * min(float(np.nanmean(boot <= 0)), float(np.nanmean(boot >= 0)))
    else:
        sign_p = 2 * min(float(np.nanmean(boot >= 0)), float(np.nanmean(boot <= 0)))
    sign_p = min(sign_p, 1.0)
    block_p = block_permutation_p(x_lag, y_aligned, observed, block_weeks, rng)
    shift_p = circular_shift_p(x_lag, y_aligned, observed, rng)
    return {
        "analysis": label,
        "block_weeks": block_weeks,
        "n_weeks": int(len(x_lag)),
        "spearman_rho": observed,
        "standard_p_recomputed": standard_p,
        "moving_block_ci_low": float(ci_low),
        "moving_block_ci_high": float(ci_high),
        "moving_block_sign_p": float(sign_p),
        "block_permutation_p": block_p,
        "circular_shift_p": shift_p,
    }


def main() -> int:
    df = pd.read_csv(VALIDATION)
    resp = df["resp_score"].astype(float)
    positivity = df["positivity"].astype(float)

    stl_resp = pd.Series(STL(resp, period=52, robust=True).fit().resid, index=df.index)
    stl_pos = pd.Series(STL(positivity, period=52, robust=True).fit().resid, index=df.index)

    sensitivity_rows = []
    for label, x, y, offset in [
        ("raw_resp_score_vs_positivity_lag4", resp, positivity, 0),
        ("stl_residual_resp_score_vs_positivity_lag4", stl_resp, stl_pos, 100),
    ]:
        for block_weeks in [4, 8]:
            sensitivity_rows.append(analyze_pair(label, x, y, block_weeks, offset + block_weeks))

    ljung_rows = []
    ljung_rows.extend(autocorr_rows("resp_score", resp))
    ljung_rows.extend(autocorr_rows("flunet_positivity", positivity))
    ljung_rows.extend(autocorr_rows("stl_resp_score_residual", stl_resp))
    ljung_rows.extend(autocorr_rows("stl_flunet_positivity_residual", stl_pos))

    OUT_DIR.mkdir(exist_ok=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    ljung = pd.DataFrame(ljung_rows)
    sensitivity.to_csv(OUT_DIR / "lag4_autocorr_seasonality_sensitivity.csv", index=False)
    ljung.to_csv(OUT_DIR / "lag4_ljung_box_autocorr.csv", index=False)

    def p_fmt(value: float) -> str:
        if value < 0.001:
            return "<0.001"
        return f"{value:.3f}"

    sensitivity_table = sensitivity.copy()
    sensitivity_table["analysis"] = sensitivity_table["analysis"].replace(
        {
            "raw_resp_score_vs_positivity_lag4": "Raw respiratory residual score",
            "stl_residual_resp_score_vs_positivity_lag4": "STL residualized series",
        }
    )
    sensitivity_table["spearman_rho"] = sensitivity_table["spearman_rho"].map(lambda v: f"{v:.3f}")
    sensitivity_table["standard_p_recomputed"] = sensitivity_table["standard_p_recomputed"].map(p_fmt)
    sensitivity_table["moving_block_95_ci"] = sensitivity.apply(
        lambda row: f"{row['moving_block_ci_low']:.3f} to {row['moving_block_ci_high']:.3f}",
        axis=1,
    )
    for col in ["moving_block_sign_p", "block_permutation_p", "circular_shift_p"]:
        sensitivity_table[col] = sensitivity[col].map(p_fmt)
    sensitivity_table = sensitivity_table[
        [
            "analysis",
            "block_weeks",
            "n_weeks",
            "spearman_rho",
            "standard_p_recomputed",
            "moving_block_95_ci",
            "moving_block_sign_p",
            "block_permutation_p",
            "circular_shift_p",
        ]
    ]
    sensitivity_table.to_csv(OUT_DIR / "lag4_autocorr_seasonality_sensitivity_table.csv", index=False)

    ljung_table = ljung.copy()
    ljung_table["series"] = ljung_table["series"].replace(
        {
            "resp_score": "Respiratory residual score",
            "flunet_positivity": "FluNet positivity",
            "stl_resp_score_residual": "STL respiratory residual",
            "stl_flunet_positivity_residual": "STL FluNet residual",
        }
    )
    ljung_table["ljung_box_q"] = ljung_table["ljung_box_q"].map(lambda v: f"{v:.1f}")
    ljung_table["ljung_box_p"] = ljung_table["ljung_box_p"].map(p_fmt)
    for col in ["acf_lag1", "acf_lag4", "acf_lag8"]:
        ljung_table[col] = ljung[col].map(lambda v: f"{v:.3f}")
    ljung_table.to_csv(OUT_DIR / "lag4_ljung_box_autocorr_table.csv", index=False)

    print(sensitivity.round(6).to_string(index=False))
    print()
    print(ljung.round(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
