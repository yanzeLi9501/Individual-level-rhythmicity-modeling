#!/usr/bin/env python3
"""Audit HiRID dataset structure to support external validation analysis.

Run after `download_physionet.py` has populated
`external_data/physionet/hirid/1.1.1/`. Prints:
  - Directory tree (top 2 levels)
  - general_data.csv columns / dtypes / first 5 rows
  - Variable reference search for APACHE / diagnosis / admission-type variables
  - Observation table partition discovery

Output: stdout only (no files written).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SUBMIT_DIR = BASE_DIR.parent
HIRID_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "hirid" / "1.1.1"


def list_tree(root: Path, max_depth: int = 2) -> None:
    if not root.exists():
        print(f"[MISSING] HiRID root does not exist: {root}")
        return
    print(f"[TREE] {root}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        prefix = "  " * (depth - 1)
        size = f"  ({path.stat().st_size/1e6:.1f} MB)" if path.is_file() else "/"
        print(f"  {prefix}{rel.parts[-1]}{size}")


def find_file(root: Path, basename: str) -> Path | None:
    if not root.exists():
        return None
    matches = list(root.rglob(basename))
    return matches[0] if matches else None


def audit_general_data() -> None:
    path = find_file(HIRID_ROOT, "general_data.csv")
    if path is None:
        # Try .csv.gz
        path = find_file(HIRID_ROOT, "general_data.csv.gz")
    print("\n" + "=" * 70)
    print(f"[GENERAL_DATA] path={path}")
    if path is None:
        print("  NOT FOUND — cannot audit columns")
        return
    df = pd.read_csv(path, nrows=5)
    print(f"  columns ({len(df.columns)}): {list(df.columns)}")
    print("  dtypes:")
    for col, dt in df.dtypes.items():
        print(f"    {col}: {dt}")
    print("  head(5):")
    print(df.to_string(index=False, max_cols=20))
    # Full row count
    try:
        total = sum(1 for _ in open(path, "r", encoding="utf-8", errors="replace")) - 1
        print(f"  total_rows: {total:,}")
    except Exception as exc:
        print(f"  total_rows: <error: {exc}>")


def audit_variable_reference() -> None:
    for name in ("hirid_variable_reference.csv", "hirid_variable_reference_preprocessed.csv"):
        path = find_file(HIRID_ROOT, name)
        print("\n" + "=" * 70)
        print(f"[VAR_REF] {name} path={path}")
        if path is None:
            print("  NOT FOUND")
            continue
        df = pd.read_csv(path)
        print(f"  columns: {list(df.columns)}")
        print(f"  rows: {len(df):,}")
        # Search for diagnosis / APACHE / admission type rows
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            print("  no string columns to search")
            continue
        combined = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
        for keyword in ("apache", "diagnos", "admission type", "admit type",
                        "icu admission", "patient group", "category"):
            mask = combined.str.contains(keyword, na=False, regex=False)
            n = int(mask.sum())
            if n:
                print(f"  [keyword='{keyword}'] {n} matching variables")
                show = df.loc[mask].head(10)
                print(show.to_string(index=False, max_cols=10))


def audit_observation_tables() -> None:
    print("\n" + "=" * 70)
    print("[OBSERVATION_TABLES] discovery")
    candidates = []
    for sub in ("raw_stage", "merged_stage", "imputed_stage"):
        d = HIRID_ROOT / sub
        if d.exists():
            candidates.append(d)
        # Also search recursively
    if not candidates:
        # Recursive search for observation_tables dirs
        for d in HIRID_ROOT.rglob("observation_tables*"):
            if d.is_dir():
                candidates.append(d)
    if not candidates:
        print("  No raw_stage / merged_stage / observation_tables directories found")
        return
    for d in candidates:
        files = list(d.rglob("*.csv")) + list(d.rglob("*.csv.gz")) + list(d.rglob("*.parquet"))
        print(f"  {d.relative_to(HIRID_ROOT)}: {len(files)} data files")
        if files:
            first = files[0]
            size_mb = first.stat().st_size / 1e6
            print(f"    sample: {first.relative_to(HIRID_ROOT)} ({size_mb:.1f} MB)")
            # Try to read a few rows
            try:
                if first.suffix == ".parquet":
                    sample = pd.read_parquet(first)
                else:
                    sample = pd.read_csv(first, nrows=5)
                print(f"    columns: {list(sample.columns)}")
                print(f"    head(3):")
                print(sample.head(3).to_string(index=False, max_cols=10))
            except Exception as exc:
                print(f"    [read error] {exc}")


def main() -> None:
    print(f"HiRID audit — root: {HIRID_ROOT}")
    print(f"exists: {HIRID_ROOT.exists()}")
    list_tree(HIRID_ROOT, max_depth=2)
    audit_general_data()
    audit_variable_reference()
    audit_observation_tables()
    print("\n" + "=" * 70)
    print("[DONE] Use this output to update HIRID_FEATURES, APACHE variable ID,")
    print("       and disease group mapping in run_hirid_external_validation.py")


if __name__ == "__main__":
    main()
