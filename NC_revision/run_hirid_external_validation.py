#!/usr/bin/env python3
"""HiRID external validation: cross-sectional Pearson profile similarity.

Adds HiRID (Bern University Hospital ICU, Switzerland) as a European
geographic positive control for the chronic-disease utilisation-profile
analysis. Mirrors the schema used by
`external_positive_control_analysis.py` so the resulting CSV plugs into the
existing forest-plot code without modification.

IMPORTANT LIMITATIONS (HiRID-specific):
  - HiRID applies per-patient random date offsets (shifts each patient to
    2100-2200) → seasonal FluNet time-series analysis is NOT possible.
  - Each ICU admission has a unique ID; cross-admission patient linkage is
    NOT possible → LGDI gap-based metrics are NOT computed.
  - Reference group: APACHE respiratory category (NOT influenza or COVID+).
    Labelled as "HiRID ICU (APACHE respiratory)" in outputs.

Run AFTER `audit_hirid_data.py` to fill in the constants below if defaults
don't match the actual schema.

Inputs:
  external_data/physionet/hirid/1.1.1/general_data.csv (+ .csv.gz fallback)
  external_data/physionet/hirid/1.1.1/raw_stage/observation_tables/*.csv
    (optional, for LOS and APACHE group assignment)

Outputs (in NC_revision/external_positive_control_results/):
  hirid_respiratory_profile_correlations.csv
  hirid_respiratory_vs_nonresp_permutation.json
  hirid_analysis_admission_features.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse helpers from the main external analysis module
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from external_positive_control_analysis import (  # noqa: E402
    BOOTSTRAPS,
    BOOTSTRAP_SAMPLE_CAP,
    GROUP_COLUMNS,
    PERMUTATIONS,
    RNG_SEED,
    analyse_profiles,
)

SUBMIT_DIR = BASE_DIR.parent
HIRID_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "hirid" / "1.1.1"
OUT_DIR = BASE_DIR / "external_positive_control_results"
CACHE_DIR = OUT_DIR / "cache"

# --- ADJUST AFTER PHASE 1 AUDIT IF NEEDED -----------------------------------
# HiRID v1.1.1 general_data.csv columns (per official docs):
#   patientid, admissiontime, sex, age, height, discharge_status
GD_PATIENT_COL = "patientid"
GD_ADMIT_COL = "admissiontime"
GD_AGE_COL = "age"
GD_SEX_COL = "sex"
GD_HEIGHT_COL = "height"
GD_DISCHARGE_STATUS_COL = "discharge_status"  # 'alive' / 'dead' / 'unknown'

# APACHE patient-group variable ID in observation_tables.
# Per HiRID variable reference, common APACHE-II patient-group variable is
# 9990002 (or similar). Confirm via audit_hirid_data.py.
APACHE_VAR_ID: int | None = None  # e.g. 9990002 — set after audit
APACHE_VAR_COL = "variableid"     # column name in observation tables
APACHE_VALUE_COL = "value"        # numeric/string-coded APACHE category

# Mapping: APACHE category value -> chronic-group flag.
# Confirm category coding via audit (string vs integer). Typical HiRID:
#   surgical/medical split with subcategories. Adjust after audit.
APACHE_GROUP_MAP: dict[object, str] = {
    # value -> group_flag_name. Example placeholders; replace after audit:
    # 98: "chronic_respiratory",  # 'Respiratory'
    # 99: "cardiovascular",       # 'Cardiovascular'
    # 100: "kidney",              # 'Renal'
    # 101: "cerebrovascular",     # 'Neurological'
    # 102: "diabetes",            # 'Metabolic/Endocrine'
}

# Feature set (use only what's in general_data + cheap observation counts).
HIRID_FEATURES = [
    "age",
    "los_hours",
    "is_male",
    "discharge_dead",
    "height",
    "observation_count",
    "pharma_count",
]
# ----------------------------------------------------------------------------


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def find_file(root: Path, basename: str) -> Path | None:
    if not root.exists():
        return None
    for ext_basename in (basename, basename + ".gz"):
        matches = list(root.rglob(ext_basename))
        if matches:
            return matches[0]
    return None


def load_general_data() -> pd.DataFrame:
    path = find_file(HIRID_ROOT, "general_data.csv")
    if path is None:
        raise FileNotFoundError(f"general_data.csv not found under {HIRID_ROOT}")
    print(f"[load] general_data: {path}", flush=True)
    df = pd.read_csv(path)
    print(f"  rows={len(df):,}  columns={list(df.columns)}", flush=True)
    return df


def count_rows_per_patient(
    obs_root: Path,
    id_col: str,
    out_col: str,
    file_glob: str = "*.csv",
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Stream observation/pharma tables, accumulate per-patient row counts."""
    cache = CACHE_DIR / f"hirid_{out_col}.csv"
    if cache.exists():
        print(f"[cache] {out_col}: {cache.name}", flush=True)
        return pd.read_csv(cache)

    if not obs_root.exists():
        print(f"[warn] {obs_root} missing — {out_col} = 0 for all patients", flush=True)
        empty = pd.DataFrame({id_col: [], out_col: []})
        empty.to_csv(cache, index=False)
        return empty

    files = sorted(list(obs_root.rglob(file_glob)) + list(obs_root.rglob(file_glob + ".gz"))
                   + list(obs_root.rglob("*.parquet")))
    print(f"[count] {out_col}: scanning {len(files)} files in {obs_root}", flush=True)
    counts: pd.Series | None = None
    for fi, path in enumerate(files, start=1):
        try:
            if path.suffix == ".parquet":
                chunk = pd.read_parquet(path, columns=[id_col])
                values = chunk[id_col].dropna().astype("int64").value_counts()
                counts = values if counts is None else counts.add(values, fill_value=0)
            else:
                for chunk in pd.read_csv(path, usecols=[id_col], chunksize=chunksize):
                    values = chunk[id_col].dropna().astype("int64").value_counts()
                    counts = values if counts is None else counts.add(values, fill_value=0)
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}", flush=True)
        if fi % 25 == 0:
            print(f"  [count] {out_col}: {fi}/{len(files)} files", flush=True)
    if counts is None:
        result = pd.DataFrame({id_col: [], out_col: []})
    else:
        result = counts.rename_axis(id_col).reset_index(name=out_col)
        result[out_col] = result[out_col].astype(int)
    result.to_csv(cache, index=False)
    print(f"[cache-write] {out_col}: rows={len(result):,}", flush=True)
    return result


def compute_los_hours(obs_root: Path, id_col: str, dt_col: str = "datetime") -> pd.DataFrame:
    """Per-patient observation time span as a LOS proxy (hours)."""
    cache = CACHE_DIR / "hirid_los_hours.csv"
    if cache.exists():
        print(f"[cache] los_hours: {cache.name}", flush=True)
        return pd.read_csv(cache)
    if not obs_root.exists():
        print(f"[warn] {obs_root} missing — los_hours = NaN", flush=True)
        empty = pd.DataFrame({id_col: [], "los_hours": []})
        empty.to_csv(cache, index=False)
        return empty
    files = sorted(list(obs_root.rglob("*.csv")) + list(obs_root.rglob("*.csv.gz"))
                   + list(obs_root.rglob("*.parquet")))
    print(f"[los] scanning {len(files)} files for max-min datetime per patient", flush=True)
    mins: dict[int, pd.Timestamp] = {}
    maxs: dict[int, pd.Timestamp] = {}
    for fi, path in enumerate(files, start=1):
        try:
            if path.suffix == ".parquet":
                chunk = pd.read_parquet(path, columns=[id_col, dt_col])
                _update_minmax(chunk, id_col, dt_col, mins, maxs)
            else:
                for chunk in pd.read_csv(path, usecols=[id_col, dt_col],
                                         chunksize=500_000, parse_dates=[dt_col]):
                    _update_minmax(chunk, id_col, dt_col, mins, maxs)
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}", flush=True)
        if fi % 25 == 0:
            print(f"  [los] {fi}/{len(files)} files", flush=True)
    ids = sorted(mins.keys())
    rows = [
        {id_col: pid, "los_hours": (maxs[pid] - mins[pid]).total_seconds() / 3600.0}
        for pid in ids
    ]
    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    print(f"[cache-write] los_hours: rows={len(result):,}", flush=True)
    return result


def _update_minmax(chunk: pd.DataFrame, id_col: str, dt_col: str,
                   mins: dict, maxs: dict) -> None:
    chunk = chunk.dropna(subset=[id_col, dt_col])
    if chunk.empty:
        return
    if not np.issubdtype(chunk[dt_col].dtype, np.datetime64):
        chunk[dt_col] = pd.to_datetime(chunk[dt_col], errors="coerce")
        chunk = chunk.dropna(subset=[dt_col])
    grp = chunk.groupby(id_col)[dt_col].agg(["min", "max"])
    for pid, row in grp.iterrows():
        pid_int = int(pid)
        if pid_int not in mins or row["min"] < mins[pid_int]:
            mins[pid_int] = row["min"]
        if pid_int not in maxs or row["max"] > maxs[pid_int]:
            maxs[pid_int] = row["max"]


def assign_apache_groups(obs_root: Path, id_col: str) -> pd.DataFrame:
    """Extract APACHE patient group per patientid from observation tables."""
    cache = CACHE_DIR / "hirid_apache_groups.csv"
    if cache.exists():
        print(f"[cache] apache_groups: {cache.name}", flush=True)
        return pd.read_csv(cache)
    if APACHE_VAR_ID is None:
        print("[warn] APACHE_VAR_ID is None — skipping APACHE assignment.", flush=True)
        empty = pd.DataFrame({id_col: [], "apache_value": []})
        empty.to_csv(cache, index=False)
        return empty
    if not obs_root.exists():
        print(f"[warn] {obs_root} missing — APACHE groups unavailable", flush=True)
        empty = pd.DataFrame({id_col: [], "apache_value": []})
        empty.to_csv(cache, index=False)
        return empty
    files = sorted(list(obs_root.rglob("*.csv")) + list(obs_root.rglob("*.csv.gz"))
                   + list(obs_root.rglob("*.parquet")))
    rows: list[pd.DataFrame] = []
    for fi, path in enumerate(files, start=1):
        try:
            if path.suffix == ".parquet":
                chunk = pd.read_parquet(path, columns=[id_col, APACHE_VAR_COL, APACHE_VALUE_COL])
                hits = chunk[chunk[APACHE_VAR_COL] == APACHE_VAR_ID]
                if not hits.empty:
                    rows.append(hits[[id_col, APACHE_VALUE_COL]].rename(
                        columns={APACHE_VALUE_COL: "apache_value"}))
            else:
                for chunk in pd.read_csv(path, usecols=[id_col, APACHE_VAR_COL, APACHE_VALUE_COL],
                                         chunksize=500_000):
                    hits = chunk[chunk[APACHE_VAR_COL] == APACHE_VAR_ID]
                    if not hits.empty:
                        rows.append(hits[[id_col, APACHE_VALUE_COL]].rename(
                            columns={APACHE_VALUE_COL: "apache_value"}))
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}", flush=True)
        if fi % 25 == 0:
            print(f"  [apache] {fi}/{len(files)} files", flush=True)
    if rows:
        out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=[id_col], keep="first")
    else:
        out = pd.DataFrame({id_col: [], "apache_value": []})
    out.to_csv(cache, index=False)
    print(f"[cache-write] apache_groups: rows={len(out):,}", flush=True)
    return out


def build_admission_features() -> pd.DataFrame:
    gd = load_general_data()
    # Standardise columns
    if GD_PATIENT_COL not in gd.columns:
        raise KeyError(f"general_data missing patient column '{GD_PATIENT_COL}'. "
                       f"Available: {list(gd.columns)}")

    # Derived fields
    gd["is_male"] = (gd[GD_SEX_COL].astype(str).str.upper().str.startswith("M")).astype(int)
    gd["discharge_dead"] = (gd[GD_DISCHARGE_STATUS_COL].astype(str).str.lower()
                            .eq("dead")).astype(int)

    # LOS proxy from observation tables
    obs_root = HIRID_ROOT / "raw_stage" / "observation_tables"
    if not obs_root.exists():
        # Fall back to anywhere named observation_tables under hirid root
        cands = [p for p in HIRID_ROOT.rglob("observation_tables") if p.is_dir()]
        obs_root = cands[0] if cands else obs_root
    pharma_root = HIRID_ROOT / "raw_stage" / "pharma_records"
    if not pharma_root.exists():
        cands = [p for p in HIRID_ROOT.rglob("pharma_records") if p.is_dir()]
        pharma_root = cands[0] if cands else pharma_root

    los = compute_los_hours(obs_root, GD_PATIENT_COL)
    obs_counts = count_rows_per_patient(obs_root, GD_PATIENT_COL, "observation_count")
    pharma_counts = count_rows_per_patient(pharma_root, GD_PATIENT_COL, "pharma_count")

    # APACHE assignment → 6 group flags
    apache = assign_apache_groups(obs_root, GD_PATIENT_COL)
    for col in GROUP_COLUMNS:
        gd[col] = 0
    gd["apache_respiratory_reference"] = 0
    if not apache.empty:
        apache_lookup = dict(zip(apache[GD_PATIENT_COL].astype("int64"),
                                  apache["apache_value"]))
        for grp_value, grp_flag in APACHE_GROUP_MAP.items():
            if grp_flag not in gd.columns:
                gd[grp_flag] = 0
        pid_series = pd.to_numeric(gd[GD_PATIENT_COL], errors="coerce").astype("Int64")
        gd_apache_value = pid_series.map(apache_lookup)
        for grp_value, grp_flag in APACHE_GROUP_MAP.items():
            mask = gd_apache_value.eq(grp_value).fillna(False)
            gd.loc[mask, grp_flag] = 1
            if grp_flag == "chronic_respiratory":
                gd.loc[mask, "apache_respiratory_reference"] = 1
    else:
        print("[warn] No APACHE assignments — using FALLBACK: mortality split.", flush=True)
        # Fallback: cannot do disease-specific groups; mark respiratory_reference as 0
        # forest-plot row will be all NaN, signalling unavailable
        pass

    # Merge features
    feat = gd.merge(los, on=GD_PATIENT_COL, how="left")
    feat = feat.merge(obs_counts, on=GD_PATIENT_COL, how="left")
    feat = feat.merge(pharma_counts, on=GD_PATIENT_COL, how="left")
    feat["observation_count"] = feat["observation_count"].fillna(0).astype(int)
    feat["pharma_count"] = feat["pharma_count"].fillna(0).astype(int)
    # Keep height numeric / NaN
    feat[GD_HEIGHT_COL] = pd.to_numeric(feat[GD_HEIGHT_COL], errors="coerce")
    feat[GD_AGE_COL] = pd.to_numeric(feat[GD_AGE_COL], errors="coerce")

    # Rename to standardised column names expected by HIRID_FEATURES
    feat = feat.rename(columns={GD_HEIGHT_COL: "height"})

    return feat


def main() -> None:
    ensure_dirs()
    print(f"HiRID external validation — root: {HIRID_ROOT}")
    print(f"  APACHE_VAR_ID = {APACHE_VAR_ID}  (set non-None after audit)")
    print(f"  APACHE_GROUP_MAP entries = {len(APACHE_GROUP_MAP)}")

    if not HIRID_ROOT.exists() or not any(HIRID_ROOT.iterdir()):
        print(f"\n[ABORT] HiRID dataset not yet downloaded to {HIRID_ROOT}.")
        print("Run: python download_physionet.py -p <PHYSIONET_PASSWORD>")
        return

    feat = build_admission_features()
    feat_out = OUT_DIR / "hirid_analysis_admission_features.csv"
    feat.to_csv(feat_out, index=False)
    print(f"[write] {feat_out}  rows={len(feat):,}  cols={feat.shape[1]}")

    if "apache_respiratory_reference" not in feat.columns or feat["apache_respiratory_reference"].sum() == 0:
        print("\n[WARN] No respiratory reference admissions identified.")
        print("       Update APACHE_VAR_ID and APACHE_GROUP_MAP after running audit.")
        print("       Writing empty correlation file as a placeholder.")
        placeholder = pd.DataFrame([
            {"dataset": "HiRID ICU (APACHE respiratory)",
             "group": g, "n_reference": 0,
             "n_group_excluding_reference": 0,
             "pearson_profile_correlation": float("nan"),
             "bootstrap_ci_low": float("nan"),
             "bootstrap_ci_high": float("nan"),
             "feature_count": len(HIRID_FEATURES)}
            for g in GROUP_COLUMNS
        ])
        placeholder.to_csv(OUT_DIR / "hirid_respiratory_profile_correlations.csv", index=False)
        with open(OUT_DIR / "hirid_respiratory_vs_nonresp_permutation.json", "w") as f:
            json.dump({"status": "skipped: no APACHE reference",
                       "n_reference": 0, "n_respiratory": 0,
                       "n_nonrespiratory_chronic": 0}, f, indent=2)
        return

    corr_df, perm = analyse_profiles(
        label="HiRID ICU (APACHE respiratory)",
        df=feat,
        features=HIRID_FEATURES,
        reference_col="apache_respiratory_reference",
        group_cols=GROUP_COLUMNS,
    )
    corr_out = OUT_DIR / "hirid_respiratory_profile_correlations.csv"
    corr_df.to_csv(corr_out, index=False)
    print(f"[write] {corr_out}")
    print(corr_df.to_string(index=False))

    perm_out = OUT_DIR / "hirid_respiratory_vs_nonresp_permutation.json"
    with open(perm_out, "w") as f:
        json.dump(perm, f, indent=2, default=float)
    print(f"[write] {perm_out}")
    print(json.dumps(perm, indent=2, default=float))


if __name__ == "__main__":
    main()
