#!/usr/bin/env python3
"""Run LGDI XGBoost surveillance on the merged WHU+cardiac 51k cohort.

Mirrors `run_lgdi_whu_primary.py` but points at `merged_whu_cardiac_51k.csv`
(union of WHU primary 32k + expanded cardiac 42k → 51,633 unique record
numbers built by `build_merged_whu_cardiac.py`).

Uses `include_external=True` so that the external linkage outputs
(`merged_lgdi_nwicu_*`, `merged_lgdi_flunet_lag_correlation.csv`, etc.) are
also produced. Output prefix: `merged_lgdi`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from run_lgdi_surveillance import run_analysis  # type: ignore  # noqa: E402

MERGED = BASE / "merged_whu_cardiac_51k.csv"


def main() -> int:
    if not MERGED.exists():
        raise FileNotFoundError(MERGED)
    print(f"Running LGDI pipeline on merged 51k cohort: {MERGED}")
    summary = run_analysis(
        MERGED,
        prefix="merged_lgdi",
        include_external=True,
        cohort_label="WHU primary + expanded cardiac merged (51,633 record numbers)",
    )
    print("\n=== merged_lgdi summary ===")
    print(json.dumps(summary.get("cohort", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
