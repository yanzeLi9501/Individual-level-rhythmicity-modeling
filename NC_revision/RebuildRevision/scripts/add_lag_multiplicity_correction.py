#!/usr/bin/env python3
"""Add Bonferroni and Benjamini-Hochberg corrections to LGDI lag tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def default_paths() -> tuple[Path, Path]:
    rebuild_dir = Path(__file__).resolve().parents[1]
    nc_revision_dir = rebuild_dir.parent
    input_csv = nc_revision_dir / "lgdi_results" / "lgdi_whu_influenza_lag_spearman.csv"
    output_dir = rebuild_dir / "outputs" / "analysis_outputs"
    return input_csv, output_dir


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q_values = np.full_like(values, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return q_values
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = finite_values.size
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    finite_q = np.empty_like(adjusted)
    finite_q[order] = adjusted
    q_values[finite_mask] = finite_q
    return q_values


def add_corrections(input_csv: Path, output_dir: Path, alpha: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_table = pd.read_csv(input_csv)
    if "p_value" not in lag_table.columns:
        raise KeyError(f"Missing p_value column in {input_csv}")
    test_count = int(lag_table["p_value"].notna().sum())
    bonferroni_alpha = alpha / max(test_count, 1)
    lag_table["p_bonferroni"] = (pd.to_numeric(lag_table["p_value"], errors="coerce") * test_count).clip(upper=1.0)
    lag_table["q_bh_fdr"] = benjamini_hochberg(lag_table["p_value"])
    lag_table["significant_bonferroni_alpha_0_05"] = pd.to_numeric(lag_table["p_value"], errors="coerce") <= bonferroni_alpha
    lag_table["significant_fdr_alpha_0_05"] = lag_table["q_bh_fdr"] <= alpha
    lag_table["multiplicity_family"] = "lag0_to_lag4_spearman"
    lag_table["bonferroni_alpha"] = bonferroni_alpha

    corrected_csv = output_dir / "lgdi_whu_influenza_lag_spearman_corrected.csv"
    lag_table.to_csv(corrected_csv, index=False)
    summary = {
        "generated_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_csv": str(input_csv),
        "output_csv": str(corrected_csv),
        "test_count": test_count,
        "family": "lag 0 through lag 4 Spearman tests",
        "alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "bonferroni_significant_lags": lag_table.loc[
            lag_table["significant_bonferroni_alpha_0_05"], "lgdi_leads_flunet_weeks"
        ].astype(int).tolist(),
        "fdr_significant_lags": lag_table.loc[
            lag_table["significant_fdr_alpha_0_05"], "lgdi_leads_flunet_weeks"
        ].astype(int).tolist(),
    }
    summary_path = output_dir / "lgdi_whu_influenza_lag_spearman_correction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    input_csv, output_dir = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=input_csv)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = add_corrections(args.input_csv, args.output_dir, args.alpha)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()