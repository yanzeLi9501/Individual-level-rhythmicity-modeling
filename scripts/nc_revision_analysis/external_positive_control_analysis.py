#!/usr/bin/env python3
"""External positive-control analyses for the NC revision.

This script uses already available external EHR datasets to address reviewer
requests for a better studied respiratory-pathogen endpoint. It computes
standardized healthcare-utilization profiles, frames the similarity metric as
Pearson correlation, and adds bootstrap confidence intervals plus a simple
permutation contrast for respiratory vs. non-respiratory chronic admissions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SUBMIT_DIR = BASE_DIR.parent
OUT_DIR = BASE_DIR / "external_positive_control_results"
CACHE_DIR = OUT_DIR / "cache"

MIMIC_ROOT = Path(r"data\mimic-iv-2.2\mimic-iv-2.2")
EICU_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "eicu-crd" / "2.0"
NWICU_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "nwicu-northwestern-icu" / "0.1.0"
MIMIC_ED_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "mimic-iv-ed" / "2.2"
MIMIC31_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "mimiciv" / "3.1"
ZIGONG_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "icu-infection-zigong-fourth"
FLUNET_ROOT = SUBMIT_DIR / "external_data" / "flunet"
NWICU_EMAR_REL = Path("data") / "nw_hosp" / "emar.csv.gz"
NWICU_EMAR_SHA256 = "8b29dbbdae7e17b89df00133c5b04ce05fcf929e71cce8f43d1cd5646a02d731"

RNG_SEED = 20260512
BOOTSTRAPS = 1000
PERMUTATIONS = 1000
BOOTSTRAP_SAMPLE_CAP = 5000


MIMIC_FEATURES = [
    "age",
    "los_days",
    "hospital_expire_flag",
    "emergency_admission",
    "ed_registered",
    "diagnosis_count",
    "procedure_count",
    "prescription_count",
    "lab_count",
    "microbiology_count",
    "poe_count",
    "transfer_count",
    "icu_stay_count",
]

EICU_FEATURES = [
    "age",
    "hospital_los_days",
    "unit_los_days",
    "hospital_mortality",
    "diagnosis_count",
    "lab_count",
    "medication_count",
    "treatment_count",
    "respiratory_care_count",
    "infusion_drug_count",
    "intake_output_count",
    "nurse_assessment_count",
    "unit_visit_number",
]

NWICU_FEATURES = [
    "age",
    "los_days",
    "hospital_expire_flag",
    "emergency_admission",
    "ed_registered",
    "diagnosis_count",
    "emar_count",
    "prescription_count",
    "lab_count",
    "icu_stay_count",
    "icu_total_los_days",
    "procedure_event_count",
    "chart_event_count",
]

GROUP_COLUMNS = [
    "chronic_respiratory",
    "cardiovascular",
    "hypertension",
    "diabetes",
    "cerebrovascular",
    "kidney",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalized_icd(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )


def numeric_icd3(codes: pd.Series) -> pd.Series:
    return pd.to_numeric(codes.str.extract(r"^(\d{3})")[0], errors="coerce")


def count_by_id(
    path: Path,
    id_col: str,
    out_col: str,
    chunksize: int = 1_000_000,
    cache_prefix: str | None = None,
) -> pd.DataFrame:
    prefix = f"{cache_prefix}_" if cache_prefix else ""
    cache = CACHE_DIR / f"{prefix}{path.stem}_{out_col}.csv"
    if cache.exists():
        print(f"[cache] {out_col}: {cache.name}", flush=True)
        return pd.read_csv(cache)

    counts: pd.Series | None = None
    print(f"[count] {out_col}: {path}", flush=True)
    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=[id_col], chunksize=chunksize), start=1):
        values = chunk[id_col].dropna().astype("int64").value_counts()
        counts = values if counts is None else counts.add(values, fill_value=0)
        if chunk_idx % 5 == 0:
            print(f"[count] {out_col}: {chunk_idx:,} chunks", flush=True)

    if counts is None:
        result = pd.DataFrame({id_col: [], out_col: []})
    else:
        result = counts.rename_axis(id_col).reset_index(name=out_col)
        result[out_col] = result[out_col].astype(int)
    result.to_csv(cache, index=False)
    print(f"[cache-write] {out_col}: {cache.name} rows={len(result):,}", flush=True)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_row_count(path: Path) -> int:
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for rows, _ in enumerate(handle, start=1):
            pass
    return rows


def audit_nwicu_emar_file() -> dict[str, object]:
    path = NWICU_ROOT / NWICU_EMAR_REL
    audit: dict[str, object] = {
        "nwicu_emar_path": str(path),
        "nwicu_emar_exists": path.exists(),
        "nwicu_emar_expected_sha256": NWICU_EMAR_SHA256,
    }
    if not path.exists():
        audit.update({"nwicu_emar_sha256": "missing", "nwicu_emar_sha256_ok": False, "nwicu_emar_gzip_ok": False, "nwicu_emar_gzip_rows": 0})
        return audit
    actual = sha256_file(path)
    audit["nwicu_emar_size_mb"] = round(path.stat().st_size / 1024 / 1024, 1)
    audit["nwicu_emar_sha256"] = actual
    audit["nwicu_emar_sha256_ok"] = actual == NWICU_EMAR_SHA256
    try:
        rows = gzip_row_count(path)
        audit["nwicu_emar_gzip_ok"] = True
        audit["nwicu_emar_gzip_rows"] = rows
    except Exception as exc:
        audit["nwicu_emar_gzip_ok"] = False
        audit["nwicu_emar_gzip_error"] = f"{type(exc).__name__}: {exc}"
        audit["nwicu_emar_gzip_rows"] = 0
    return audit


def pearson_profile(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    good = np.isfinite(vec_a) & np.isfinite(vec_b)
    if good.sum() < 3:
        return float("nan")
    a = vec_a[good]
    b = vec_b[good]
    if np.isclose(a.std(ddof=0), 0) or np.isclose(b.std(ddof=0), 0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_profile_ci(
    ref_values: np.ndarray,
    group_values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = BOOTSTRAPS,
) -> tuple[float, float]:
    if len(ref_values) == 0 or len(group_values) == 0:
        return float("nan"), float("nan")
    ref_size = min(len(ref_values), BOOTSTRAP_SAMPLE_CAP)
    group_size = min(len(group_values), BOOTSTRAP_SAMPLE_CAP)
    estimates = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        ref_sample = ref_values[rng.integers(0, len(ref_values), ref_size)]
        group_sample = group_values[rng.integers(0, len(group_values), group_size)]
        estimates[idx] = pearson_profile(ref_sample.mean(axis=0), group_sample.mean(axis=0))
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def permutation_resp_contrast(
    z_values: pd.DataFrame,
    reference_mask: pd.Series,
    respiratory_mask: pd.Series,
    chronic_mask: pd.Series,
    rng: np.random.Generator,
    n_perm: int = PERMUTATIONS,
) -> dict[str, float | int]:
    valid = z_values.notna().all(axis=1)
    ref_idx = reference_mask & valid
    analysis_idx = chronic_mask & ~reference_mask & valid
    resp_idx = respiratory_mask & analysis_idx
    nonresp_idx = analysis_idx & ~respiratory_mask

    if ref_idx.sum() == 0 or resp_idx.sum() == 0 or nonresp_idx.sum() == 0:
        return {
            "n_reference": int(ref_idx.sum()),
            "n_respiratory": int(resp_idx.sum()),
            "n_nonrespiratory_chronic": int(nonresp_idx.sum()),
            "observed_difference": float("nan"),
            "p_two_sided": float("nan"),
            "n_permutations": 0,
        }

    ref_vec = z_values.loc[ref_idx].to_numpy().mean(axis=0)
    pool = z_values.loc[analysis_idx].to_numpy()
    labels = respiratory_mask.loc[analysis_idx].to_numpy(dtype=bool)
    resp_count = int(labels.sum())
    total_sum = pool.sum(axis=0)
    nonresp_count = len(labels) - resp_count

    observed = pearson_profile(ref_vec, pool[labels].mean(axis=0)) - pearson_profile(
        ref_vec, pool[~labels].mean(axis=0)
    )

    permuted = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        selected = rng.choice(len(labels), size=resp_count, replace=False)
        selected_sum = pool[selected].sum(axis=0)
        resp_mean = selected_sum / resp_count
        nonresp_mean = (total_sum - selected_sum) / nonresp_count
        permuted[idx] = pearson_profile(ref_vec, resp_mean) - pearson_profile(ref_vec, nonresp_mean)

    p_value = (np.sum(np.abs(permuted) >= abs(observed)) + 1) / (n_perm + 1)
    return {
        "n_reference": int(ref_idx.sum()),
        "n_respiratory": int(resp_idx.sum()),
        "n_nonrespiratory_chronic": int(nonresp_idx.sum()),
        "observed_difference": float(observed),
        "p_two_sided": float(p_value),
        "n_permutations": n_perm,
        "permuted_difference_mean": float(np.nanmean(permuted)),
        "permuted_difference_sd": float(np.nanstd(permuted, ddof=1)),
    }


def analyse_profiles(
    label: str,
    df: pd.DataFrame,
    features: list[str],
    reference_col: str,
    group_cols: list[str] = GROUP_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(RNG_SEED)
    work = df.copy()
    for feature in features:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    feature_frame = work[features].replace([np.inf, -np.inf], np.nan)
    means = feature_frame.mean(axis=0)
    stds = feature_frame.std(axis=0, ddof=0).replace(0, np.nan)
    z_values = (feature_frame - means) / stds
    valid = z_values.notna().all(axis=1)
    ref_mask = work[reference_col].fillna(False).astype(bool) & valid
    ref_values = z_values.loc[ref_mask].to_numpy()
    ref_vec = ref_values.mean(axis=0)

    rows = []
    for group in group_cols:
        group_mask = work[group].fillna(False).astype(bool) & ~work[reference_col].fillna(False).astype(bool) & valid
        group_values = z_values.loc[group_mask].to_numpy()
        group_vec = group_values.mean(axis=0) if len(group_values) else np.full(len(features), np.nan)
        corr = pearson_profile(ref_vec, group_vec)
        ci_low, ci_high = bootstrap_profile_ci(ref_values, group_values, rng)
        rows.append(
            {
                "dataset": label,
                "group": group,
                "n_reference": int(ref_mask.sum()),
                "n_group_excluding_reference": int(group_mask.sum()),
                "pearson_profile_correlation": corr,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "feature_count": len(features),
            }
        )

    chronic_mask = work[group_cols].fillna(False).astype(bool).any(axis=1)
    perm = permutation_resp_contrast(
        z_values=z_values,
        reference_mask=work[reference_col].fillna(False).astype(bool),
        respiratory_mask=work["chronic_respiratory"].fillna(False).astype(bool),
        chronic_mask=chronic_mask,
        rng=rng,
    )
    perm["dataset"] = label
    perm["reference_col"] = reference_col
    return pd.DataFrame(rows), perm


def mimic_group_flags(dx: pd.DataFrame) -> pd.DataFrame:
    codes = normalized_icd(dx["icd_code"])
    code3 = numeric_icd3(codes)
    version = pd.to_numeric(dx["icd_version"], errors="coerce")

    is_icd9 = version.eq(9)
    is_icd10 = version.eq(10)
    flags = pd.DataFrame({"hadm_id": dx["hadm_id"]})
    flags["influenza_reference"] = (is_icd9 & code3.isin([487, 488])) | (
        is_icd10 & codes.str.startswith(("J09", "J10", "J11"))
    )
    flags["chronic_respiratory"] = (
        is_icd9 & code3.between(490, 496, inclusive="both")
    ) | (is_icd10 & codes.str.startswith(("J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47")))
    flags["cardiovascular"] = (is_icd9 & code3.between(390, 459, inclusive="both")) | (
        is_icd10 & codes.str.startswith("I")
    )
    flags["hypertension"] = (is_icd9 & code3.between(401, 405, inclusive="both")) | (
        is_icd10 & codes.str.startswith(("I10", "I11", "I12", "I13", "I15"))
    )
    flags["diabetes"] = (is_icd9 & code3.eq(250)) | (
        is_icd10 & codes.str.startswith(("E10", "E11", "E12", "E13", "E14"))
    )
    flags["cerebrovascular"] = (is_icd9 & code3.between(430, 438, inclusive="both")) | (
        is_icd10 & codes.str.startswith(("I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"))
    )
    flags["kidney"] = (is_icd9 & code3.between(580, 589, inclusive="both")) | (
        is_icd10 & codes.str.startswith(("N17", "N18", "N19"))
    )
    return flags.groupby("hadm_id", as_index=False)[["influenza_reference", *GROUP_COLUMNS]].max()


def nwicu_group_flags(dx: pd.DataFrame) -> pd.DataFrame:
    codes = normalized_icd(dx["icd_code"])
    code3 = numeric_icd3(codes)
    version = pd.to_numeric(dx["icd_version"], errors="coerce")

    is_icd9 = version.eq(9)
    is_icd10 = version.eq(10)
    flags = pd.DataFrame({"hadm_id": dx["hadm_id"]})
    flags["covid_reference"] = is_icd10 & codes.str.startswith("U07")
    flags["chronic_respiratory"] = (
        is_icd9 & code3.between(490, 496, inclusive="both")
    ) | (is_icd10 & codes.str.startswith(("J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47")))
    flags["cardiovascular"] = (is_icd9 & code3.between(390, 459, inclusive="both")) | (
        is_icd10 & codes.str.startswith("I")
    )
    flags["hypertension"] = (is_icd9 & code3.between(401, 405, inclusive="both")) | (
        is_icd10 & codes.str.startswith(("I10", "I11", "I12", "I13", "I15"))
    )
    flags["diabetes"] = (is_icd9 & code3.eq(250)) | (
        is_icd10 & codes.str.startswith(("E10", "E11", "E12", "E13", "E14"))
    )
    flags["cerebrovascular"] = (is_icd9 & code3.between(430, 438, inclusive="both")) | (
        is_icd10 & codes.str.startswith(("I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"))
    )
    flags["kidney"] = (is_icd9 & code3.between(580, 589, inclusive="both")) | (
        is_icd10 & codes.str.startswith(("N17", "N18", "N19"))
    )
    return flags.groupby("hadm_id", as_index=False)[["covid_reference", *GROUP_COLUMNS]].max()


def build_mimic_dataset() -> pd.DataFrame:
    admissions = pd.read_csv(MIMIC_ROOT / "hosp" / "admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    patients = pd.read_csv(MIMIC_ROOT / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    dx = pd.read_csv(MIMIC_ROOT / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})

    df = admissions.merge(patients, on="subject_id", how="left")
    df["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    df["emergency_admission"] = df["admission_type"].fillna("").str.contains(
        "EMERGENCY|URGENT|EW EMER", case=False, regex=True
    ).astype(int)
    df["ed_registered"] = df["edregtime"].notna().astype(int)

    df = df.merge(mimic_group_flags(dx), on="hadm_id", how="left")
    df = df.merge(dx.groupby("hadm_id").size().reset_index(name="diagnosis_count"), on="hadm_id", how="left")

    count_specs = [
        (MIMIC_ROOT / "hosp" / "procedures_icd.csv.gz", "hadm_id", "procedure_count"),
        (MIMIC_ROOT / "hosp" / "prescriptions.csv.gz", "hadm_id", "prescription_count"),
        (MIMIC_ROOT / "hosp" / "labevents.csv.gz", "hadm_id", "lab_count"),
        (MIMIC_ROOT / "hosp" / "microbiologyevents.csv.gz", "hadm_id", "microbiology_count"),
        (MIMIC_ROOT / "hosp" / "poe.csv.gz", "hadm_id", "poe_count"),
        (MIMIC_ROOT / "hosp" / "transfers.csv.gz", "hadm_id", "transfer_count"),
        (MIMIC_ROOT / "icu" / "icustays.csv.gz", "hadm_id", "icu_stay_count"),
    ]
    for path, id_col, out_col in count_specs:
        df = df.merge(count_by_id(path, id_col, out_col), on="hadm_id", how="left")

    for col in ["influenza_reference", *GROUP_COLUMNS]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in MIMIC_FEATURES:
        if col not in ["age", "los_days", "hospital_expire_flag", "emergency_admission", "ed_registered"]:
            df[col] = df[col].fillna(0)
    return df


def build_nwicu_dataset() -> pd.DataFrame:
    hosp_root = NWICU_ROOT / "data" / "nw_hosp"
    icu_root = NWICU_ROOT / "data" / "nw_icu"
    admissions = pd.read_csv(hosp_root / "admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    patients = pd.read_csv(hosp_root / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    dx = pd.read_csv(hosp_root / "diagnoses_icd.csv.gz", dtype={"icd_code": str})

    df = admissions.merge(patients, on="subject_id", how="left")
    df["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    df["emergency_admission"] = df["admission_type"].fillna("").str.contains(
        "EMERGENCY|URGENT|EW EMER", case=False, regex=True
    ).astype(int)
    df["ed_registered"] = df["edregtime"].notna().astype(int)

    df = df.merge(nwicu_group_flags(dx), on="hadm_id", how="left")
    df = df.merge(dx.groupby("hadm_id").size().reset_index(name="diagnosis_count"), on="hadm_id", how="left")

    count_specs = [
        (hosp_root / "emar.csv.gz", "hadm_id", "emar_count"),
        (hosp_root / "prescriptions.csv.gz", "hadm_id", "prescription_count"),
        (hosp_root / "labevents.csv.gz", "hadm_id", "lab_count"),
        (icu_root / "icustays.csv.gz", "hadm_id", "icu_stay_count"),
        (icu_root / "procedureevents.csv.gz", "hadm_id", "procedure_event_count"),
        (icu_root / "chartevents.csv.gz", "hadm_id", "chart_event_count"),
    ]
    for path, id_col, out_col in count_specs:
        df = df.merge(count_by_id(path, id_col, out_col, cache_prefix="nwicu"), on="hadm_id", how="left")

    icu_stays = pd.read_csv(icu_root / "icustays.csv.gz", usecols=["hadm_id", "los"])
    icu_los = (
        icu_stays.assign(los=pd.to_numeric(icu_stays["los"], errors="coerce").fillna(0))
        .groupby("hadm_id", as_index=False)["los"]
        .sum()
        .rename(columns={"los": "icu_total_los_days"})
    )
    df = df.merge(icu_los, on="hadm_id", how="left")

    for col in ["covid_reference", *GROUP_COLUMNS]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in NWICU_FEATURES:
        if col not in ["age", "los_days", "hospital_expire_flag", "emergency_admission", "ed_registered"]:
            df[col] = df[col].fillna(0)
    return df


def parse_eicu_age(age: object) -> float:
    if pd.isna(age):
        return float("nan")
    text = str(age).strip()
    if text.startswith(">"):
        return 90.0
    return float(text) if text else float("nan")


def eicu_group_flags(dx: pd.DataFrame, reference_name: str) -> pd.DataFrame:
    codes = normalized_icd(dx["icd9code"])
    text = dx["diagnosisstring"].fillna("").astype(str).str.lower()

    flags = pd.DataFrame({"patientunitstayid": dx["patientunitstayid"]})
    strict_flu = codes.str.contains(r"(?:^|[,; ]+)(?:487|488|J09|J10|J11)", regex=True)
    viral_pneumonia = codes.str.contains(r"(?:^|[,; ]+)(?:480|J12)", regex=True) | (
        text.str.contains("viral", regex=False) & text.str.contains("pneumonia", regex=False)
    )
    flags[reference_name] = strict_flu if reference_name == "influenza_reference" else viral_pneumonia
    flags["chronic_respiratory"] = text.str.contains(
        "copd|chronic obstructive|emphysema|asthma|chronic bronchitis", regex=True
    ) | codes.str.contains(r"(?:^|[,; ]+)(?:491|492|493|496|J44|J45)", regex=True)
    flags["cardiovascular"] = text.str.contains("cardiovascular|coronary|heart failure|arrhythmia|cardiac", regex=True) | codes.str.contains(
        r"(?:^|[,; ]+)(?:39\d|4[0-5]\d|I\d\d)", regex=True
    )
    flags["hypertension"] = text.str.contains("hypertension", regex=False) | codes.str.contains(
        r"(?:^|[,; ]+)(?:401|402|403|404|405|I10|I11|I12|I13|I15)", regex=True
    )
    flags["diabetes"] = text.str.contains("diabetes", regex=False) | codes.str.contains(
        r"(?:^|[,; ]+)(?:250|E10|E11|E12|E13|E14)", regex=True
    )
    flags["cerebrovascular"] = text.str.contains("stroke|cerebrovascular|intracranial hemorrhage", regex=True) | codes.str.contains(
        r"(?:^|[,; ]+)(?:43\d|I6\d)", regex=True
    )
    flags["kidney"] = text.str.contains("renal|kidney|dialysis", regex=True) | codes.str.contains(
        r"(?:^|[,; ]+)(?:58\d|N17|N18|N19)", regex=True
    )
    return flags.groupby("patientunitstayid", as_index=False)[[reference_name, *GROUP_COLUMNS]].max()


def build_eicu_dataset(reference_name: str = "influenza_reference") -> pd.DataFrame:
    patient = pd.read_csv(EICU_ROOT / "patient.csv.gz")
    dx = pd.read_csv(EICU_ROOT / "diagnosis.csv.gz", dtype={"icd9code": str, "diagnosisstring": str})

    df = patient[[
        "patientunitstayid",
        "age",
        "hospitaldischargeoffset",
        "unitdischargeoffset",
        "hospitaldischargestatus",
        "unitvisitnumber",
    ]].copy()
    df["age"] = df["age"].map(parse_eicu_age)
    df["hospital_los_days"] = pd.to_numeric(df["hospitaldischargeoffset"], errors="coerce") / 1440
    df["unit_los_days"] = pd.to_numeric(df["unitdischargeoffset"], errors="coerce") / 1440
    df["hospital_mortality"] = df["hospitaldischargestatus"].fillna("").str.lower().eq("expired").astype(int)
    df["unit_visit_number"] = pd.to_numeric(df["unitvisitnumber"], errors="coerce").fillna(1)
    df = df.merge(eicu_group_flags(dx, reference_name), on="patientunitstayid", how="left")
    df = df.merge(dx.groupby("patientunitstayid").size().reset_index(name="diagnosis_count"), on="patientunitstayid", how="left")

    count_specs = [
        (EICU_ROOT / "lab.csv.gz", "patientunitstayid", "lab_count"),
        (EICU_ROOT / "medication.csv.gz", "patientunitstayid", "medication_count"),
        (EICU_ROOT / "treatment.csv.gz", "patientunitstayid", "treatment_count"),
        (EICU_ROOT / "respiratoryCare.csv.gz", "patientunitstayid", "respiratory_care_count"),
        (EICU_ROOT / "infusionDrug.csv.gz", "patientunitstayid", "infusion_drug_count"),
        (EICU_ROOT / "intakeOutput.csv.gz", "patientunitstayid", "intake_output_count"),
        (EICU_ROOT / "nurseAssessment.csv.gz", "patientunitstayid", "nurse_assessment_count"),
    ]
    for path, id_col, out_col in count_specs:
        df = df.merge(count_by_id(path, id_col, out_col), on="patientunitstayid", how="left")

    for col in [reference_name, *GROUP_COLUMNS]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in EICU_FEATURES:
        if col not in ["age", "hospital_los_days", "unit_los_days", "hospital_mortality", "unit_visit_number"]:
            df[col] = df[col].fillna(0)
    return df


def plot_correlations(results: pd.DataFrame, output_path: Path, title: str) -> None:
    order = results.sort_values("pearson_profile_correlation", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(order))
    ax.barh(y, order["pearson_profile_correlation"], color="#4477AA")
    lower = order["pearson_profile_correlation"] - order["bootstrap_ci_low"]
    upper = order["bootstrap_ci_high"] - order["pearson_profile_correlation"]
    ax.errorbar(order["pearson_profile_correlation"], y, xerr=[lower, upper], fmt="none", ecolor="#222222", capsize=3)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order["group"].str.replace("_", " "))
    ax.set_xlabel("Pearson correlation of standardized utilization profiles")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def validate_external_files() -> pd.DataFrame:
    rows = []
    for label, root in [
        ("MIMIC-IV v2.2", MIMIC_ROOT),
        ("MIMIC-IV-ED v2.2", MIMIC_ED_ROOT),
        ("NWICU v0.1.0", NWICU_ROOT),
        ("eICU-CRD v2.0", EICU_ROOT),
        ("MIMIC-IV v3.1 partial", MIMIC31_ROOT),
        ("Zigong Critical Care", ZIGONG_ROOT),
        ("WHO FluNet local files", FLUNET_ROOT),
    ]:
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        part_files = [p for p in files if p.name.endswith(".part")]
        html_like = []
        if label.startswith("WHO"):
            for path in files:
                if path.suffix.lower() == ".csv":
                    head = path.read_text(encoding="utf-8", errors="ignore")[:200].lower()
                    if "<!doctype html" in head or "<html" in head:
                        html_like.append(path.name)
        row = {
            "dataset": label,
            "root": str(root),
            "exists": root.exists(),
            "file_count": len(files),
            "part_file_count": len(part_files),
            "total_mb": round(sum(p.stat().st_size for p in files) / 1024 / 1024, 1),
            "html_like_csv_files": ";".join(html_like),
        }
        if label.startswith("NWICU"):
            row.update(audit_nwicu_emar_file())
        rows.append(row)
    status = pd.DataFrame(rows)
    status.to_csv(OUT_DIR / "external_data_file_status.csv", index=False)
    return status


def read_flunet_csv(path: Path) -> pd.DataFrame:
    head = path.read_text(encoding="utf-8", errors="ignore")[:200].lower()
    if "<!doctype html" in head or "<html" in head:
        raise ValueError(f"WHO FluNet file is HTML, not CSV: {path}")
    return pd.read_csv(path, low_memory=False)


def aggregate_flunet_weekly(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    work = frame.copy()
    for col in ["ISO_YEAR", "ISO_WEEK", "SPEC_PROCESSED_NB", "SPEC_RECEIVED_NB", "INF_A", "INF_B", "INF_ALL", "INF_NEGATIVE", "RSV", "HUMAN_CORONA"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "INF_ALL" not in work.columns:
        work["INF_ALL"] = np.nan
    positive = work["INF_ALL"].where(work["INF_ALL"].notna(), work.get("INF_A", 0).fillna(0) + work.get("INF_B", 0).fillna(0))
    work["influenza_positive"] = pd.to_numeric(positive, errors="coerce").fillna(0)
    processed = work.get("SPEC_PROCESSED_NB", pd.Series(index=work.index, dtype=float))
    fallback_processed = work["influenza_positive"] + work.get("INF_NEGATIVE", pd.Series(index=work.index, dtype=float)).fillna(0)
    work["specimens_processed"] = processed.where(processed.notna() & processed.gt(0), fallback_processed).fillna(0)
    work["rsv_positive"] = work.get("RSV", pd.Series(index=work.index, dtype=float)).fillna(0)
    work["human_coronavirus_positive"] = work.get("HUMAN_CORONA", pd.Series(index=work.index, dtype=float)).fillna(0)
    grouped = (
        work.groupby(["ISO_YEAR", "ISO_WEEK"], as_index=False)[["specimens_processed", "influenza_positive", "rsv_positive", "human_coronavirus_positive"]]
        .sum()
        .sort_values(["ISO_YEAR", "ISO_WEEK"])
    )
    grouped["region"] = region
    grouped["influenza_percent_positive"] = np.where(
        grouped["specimens_processed"] > 0,
        grouped["influenza_positive"] / grouped["specimens_processed"] * 100,
        np.nan,
    )
    grouped["week_start"] = pd.to_datetime(
        grouped["ISO_YEAR"].astype(int).astype(str) + "-W" + grouped["ISO_WEEK"].astype(int).astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    grouped["influenza_percent_positive_4week"] = grouped["influenza_percent_positive"].rolling(4, min_periods=1).mean()
    return grouped[[
        "region",
        "ISO_YEAR",
        "ISO_WEEK",
        "week_start",
        "specimens_processed",
        "influenza_positive",
        "influenza_percent_positive",
        "influenza_percent_positive_4week",
        "rsv_positive",
        "human_coronavirus_positive",
    ]]


def build_who_flunet_analysis() -> dict[str, object]:
    china_path = FLUNET_ROOT / "flunet_china_2009_2024.csv"
    global_path = FLUNET_ROOT / "flunet_global_2009_2024.csv"
    china = read_flunet_csv(china_path)
    global_frame = read_flunet_csv(global_path)
    rest_world = global_frame[global_frame["COUNTRY_CODE"].fillna("").astype(str).str.upper().ne("CHN")].copy()

    weekly = pd.concat(
        [
            aggregate_flunet_weekly(china, "China"),
            aggregate_flunet_weekly(rest_world, "World excluding China"),
        ],
        ignore_index=True,
    )
    weekly.to_csv(OUT_DIR / "who_flunet_weekly_china_vs_rest_world.csv", index=False)

    annual = (
        weekly.groupby(["region", "ISO_YEAR"], as_index=False)[["specimens_processed", "influenza_positive", "rsv_positive", "human_coronavirus_positive"]]
        .sum()
        .sort_values(["region", "ISO_YEAR"])
    )
    annual["influenza_percent_positive"] = np.where(
        annual["specimens_processed"] > 0,
        annual["influenza_positive"] / annual["specimens_processed"] * 100,
        np.nan,
    )
    peak = weekly.loc[weekly.groupby("region")["influenza_percent_positive_4week"].idxmax(), ["region", "ISO_YEAR", "ISO_WEEK", "influenza_percent_positive_4week"]]
    peak = peak.rename(columns={"ISO_YEAR": "peak_iso_year", "ISO_WEEK": "peak_iso_week", "influenza_percent_positive_4week": "peak_4week_percent_positive"})
    annual.to_csv(OUT_DIR / "who_flunet_annual_china_vs_rest_world.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=False)
    for region, color in [("China", "#D65F5F"), ("World excluding China", "#4C78A8")]:
        sub = annual[annual["region"].eq(region)]
        axes[0].plot(sub["ISO_YEAR"], sub["influenza_percent_positive"], marker="o", linewidth=1.8, label=region, color=color)
        recent = weekly[weekly["region"].eq(region) & weekly["ISO_YEAR"].between(2018, 2024)]
        axes[1].plot(recent["week_start"], recent["influenza_percent_positive_4week"], linewidth=1.2, label=region, color=color)
    axes[0].set_title("WHO FluNet annual influenza positivity")
    axes[0].set_ylabel("Positive specimens (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].set_title("WHO FluNet weekly influenza positivity, 2018-2024")
    axes[1].set_ylabel("4-week mean positive specimens (%)")
    axes[1].set_xlabel("ISO week start")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "who_flunet_china_vs_rest_world.png", dpi=300)
    plt.close(fig)

    latest_year = int(annual["ISO_YEAR"].max())
    summary_rows = []
    for region in ["China", "World excluding China"]:
        sub = annual[annual["region"].eq(region)]
        latest = sub[sub["ISO_YEAR"].eq(latest_year)].iloc[0].to_dict() if not sub[sub["ISO_YEAR"].eq(latest_year)].empty else {}
        peak_row = peak[peak["region"].eq(region)].iloc[0].to_dict()
        summary_rows.append({
            "region": region,
            "years": f"{int(sub['ISO_YEAR'].min())}-{int(sub['ISO_YEAR'].max())}",
            "total_specimens_processed": int(sub["specimens_processed"].sum()),
            "total_influenza_positive": int(sub["influenza_positive"].sum()),
            "overall_percent_positive": float(sub["influenza_positive"].sum() / sub["specimens_processed"].sum() * 100),
            "latest_year": latest_year,
            "latest_year_percent_positive": float(latest.get("influenza_percent_positive", np.nan)),
            "peak_iso_year": int(peak_row["peak_iso_year"]),
            "peak_iso_week": int(peak_row["peak_iso_week"]),
            "peak_4week_percent_positive": float(peak_row["peak_4week_percent_positive"]),
        })
    summary = {
        "source": "WHO FluNet xMart VIW_FNT CSV export",
        "china_file": str(china_path),
        "global_file": str(global_path),
        "regions": summary_rows,
        "output_weekly_csv": str(OUT_DIR / "who_flunet_weekly_china_vs_rest_world.csv"),
        "output_annual_csv": str(OUT_DIR / "who_flunet_annual_china_vs_rest_world.csv"),
        "output_figure": str(OUT_DIR / "who_flunet_china_vs_rest_world.png"),
        "interpretation": "WHO FluNet provides independent aggregate respiratory-virus surveillance context; it is not patient-level EHR validation and is kept separate from profile-correlation analyses.",
    }
    (OUT_DIR / "who_flunet_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_summary(
    status: pd.DataFrame,
    mimic_results: pd.DataFrame,
    mimic_perm: dict[str, float | int],
    nwicu_results: pd.DataFrame,
    nwicu_perm: dict[str, float | int],
    eicu_results: pd.DataFrame,
    eicu_perm: dict[str, float | int],
    who_summary: dict[str, object],
) -> None:
    summary = {
        "generated_on": "2026-05-12",
        "analysis_scope": {
            "primary_positive_control": "MIMIC-IV v2.2 influenza-coded hospitalizations",
            "external_covid_validation": "NWICU v0.1.0 COVID-coded admissions from a US hospital",
            "secondary_external_icu_positive_control": "eICU-CRD v2.0 viral-pneumonia-coded ICU stays; strict influenza ICD codes were not present",
            "who_surveillance_validation": "WHO FluNet aggregate influenza surveillance split into China and world excluding China",
            "metric": "Pearson correlation of standardized healthcare-utilization profiles",
            "feature_count_mimic": len(MIMIC_FEATURES),
            "feature_count_nwicu": len(NWICU_FEATURES),
            "feature_count_eicu": len(EICU_FEATURES),
            "bootstrap_replicates": BOOTSTRAPS,
            "permutation_replicates": PERMUTATIONS,
        },
        "file_status": status.to_dict(orient="records"),
        "mimic_top_group": mimic_results.sort_values("pearson_profile_correlation", ascending=False).head(1).to_dict(orient="records"),
        "mimic_permutation": mimic_perm,
        "nwicu_top_group": nwicu_results.sort_values("pearson_profile_correlation", ascending=False).head(1).to_dict(orient="records"),
        "nwicu_permutation": nwicu_perm,
        "eicu_top_group": eicu_results.sort_values("pearson_profile_correlation", ascending=False).head(1).to_dict(orient="records"),
        "eicu_permutation": eicu_perm,
        "who_flunet_summary": who_summary,
        "limitations": [
            "MIMIC-IV dates are de-identified/shifted, so this analysis should be framed as an influenza diagnostic positive-control profile analysis rather than a direct calendar-week FluNet alignment.",
            "NWICU provides a COVID-coded US hospital cohort, but its dates are also de-identified/shifted, so it supports cross-system endpoint replication rather than real-time calendar surveillance.",
            "WHO FluNet is aggregate surveillance data and is therefore analyzed separately from patient-level EHR profile correlations.",
            "eICU has limited calendar information and only 2014-2015 coverage, so it is a secondary ICU replication rather than a seasonal time-series validation.",
        ],
    }
    (OUT_DIR / "external_positive_control_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_covid_dataset_audit(nwicu_perm: dict[str, float | int]) -> None:
    rows = [
        {
            "dataset": "NWICU v0.1.0",
            "country_or_region": "United States",
            "local_status": "downloaded_complete",
            "covid_endpoint_available": "yes_ICD10_U07",
            "use_in_revision": "main_external_covid_validation",
            "reference_n_used": nwicu_perm.get("n_reference"),
            "reason": "Only downloaded non-China public EHR dataset with a sizable COVID-coded admission endpoint; emar.csv.gz is included after SHA256 and gzip integrity verification.",
        },
        {
            "dataset": "MIMIC-IV v2.2 + MIMIC-IV-ED v2.2",
            "country_or_region": "United States",
            "local_status": "downloaded_complete",
            "covid_endpoint_available": "no_pre_covid_release_window",
            "use_in_revision": "influenza_positive_control_only",
            "reference_n_used": "1964 influenza admissions",
            "reason": "Coverage precedes COVID; useful for Reviewer #1 seasonal respiratory-pathogen endpoint.",
        },
        {
            "dataset": "eICU-CRD v2.0",
            "country_or_region": "United States multi-center ICU",
            "local_status": "downloaded_complete",
            "covid_endpoint_available": "no_2014_2015_pre_covid",
            "use_in_revision": "viral_pneumonia_positive_control",
            "reference_n_used": "259 viral-pneumonia ICU stays",
            "reason": "Pre-COVID but externally replicates respiratory-infection profile patterns in a US ICU network.",
        },
        {
            "dataset": "Zigong Critical Care Database",
            "country_or_region": "China",
            "local_status": "downloaded_zip",
            "covid_endpoint_available": "not_other_country_endpoint",
            "use_in_revision": "not_used_for_foreign_covid_validation",
            "reference_n_used": "NA",
            "reason": "Useful as a China critical-care comparison, but does not answer the requested other-country COVID validation gap.",
        },
        {
            "dataset": "MIMIC-IV v3.1 partial",
            "country_or_region": "United States",
            "local_status": "partial_download_with_part_files",
            "covid_endpoint_available": "not_validated_due_partial_download",
            "use_in_revision": "not_used",
            "reference_n_used": "NA",
            "reason": "Incomplete local copy; v2.2 and NWICU provide cleaner reproducible inputs.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "external_covid_dataset_audit.csv", index=False)


def main() -> None:
    ensure_dirs()
    print("[phase] validate external files", flush=True)
    status = validate_external_files()
    print("[phase] WHO FluNet China/rest-of-world", flush=True)
    who_summary = build_who_flunet_analysis()

    print("[phase] MIMIC-IV influenza profile", flush=True)
    mimic_df = build_mimic_dataset()
    mimic_df[["hadm_id", "influenza_reference", *GROUP_COLUMNS, *MIMIC_FEATURES]].to_csv(
        OUT_DIR / "mimic_analysis_admission_features.csv", index=False
    )
    mimic_results, mimic_perm = analyse_profiles(
        "MIMIC-IV v2.2",
        mimic_df,
        MIMIC_FEATURES,
        "influenza_reference",
    )
    mimic_results.to_csv(OUT_DIR / "mimic_influenza_profile_correlations.csv", index=False)
    (OUT_DIR / "mimic_respiratory_vs_nonresp_permutation.json").write_text(
        json.dumps(mimic_perm, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_correlations(
        mimic_results,
        OUT_DIR / "mimic_influenza_profile_correlations.png",
        "MIMIC-IV influenza positive-control profile",
    )

    print("[phase] NWICU COVID profile", flush=True)
    nwicu_df = build_nwicu_dataset()
    nwicu_df[["hadm_id", "covid_reference", *GROUP_COLUMNS, *NWICU_FEATURES]].to_csv(
        OUT_DIR / "nwicu_analysis_admission_features.csv", index=False
    )
    nwicu_results, nwicu_perm = analyse_profiles(
        "NWICU v0.1.0",
        nwicu_df,
        NWICU_FEATURES,
        "covid_reference",
    )
    nwicu_results.to_csv(OUT_DIR / "nwicu_covid_profile_correlations.csv", index=False)
    (OUT_DIR / "nwicu_covid_respiratory_vs_nonresp_permutation.json").write_text(
        json.dumps(nwicu_perm, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_correlations(
        nwicu_results,
        OUT_DIR / "nwicu_covid_profile_correlations.png",
        "NWICU COVID-coded external validation profile",
    )

    print("[phase] eICU viral-pneumonia profile", flush=True)
    eicu_reference_col = "viral_pneumonia_reference"
    eicu_df = build_eicu_dataset(eicu_reference_col)
    eicu_df[["patientunitstayid", eicu_reference_col, *GROUP_COLUMNS, *EICU_FEATURES]].to_csv(
        OUT_DIR / "eicu_analysis_stay_features.csv", index=False
    )
    eicu_results, eicu_perm = analyse_profiles(
        "eICU-CRD v2.0",
        eicu_df,
        EICU_FEATURES,
        eicu_reference_col,
    )
    eicu_results.to_csv(OUT_DIR / "eicu_viral_pneumonia_profile_correlations.csv", index=False)
    (OUT_DIR / "eicu_respiratory_vs_nonresp_permutation.json").write_text(
        json.dumps(eicu_perm, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_correlations(
        eicu_results,
        OUT_DIR / "eicu_viral_pneumonia_profile_correlations.png",
        "eICU viral pneumonia positive-control profile",
    )

    print("[phase] write summaries", flush=True)
    write_summary(status, mimic_results, mimic_perm, nwicu_results, nwicu_perm, eicu_results, eicu_perm, who_summary)
    write_covid_dataset_audit(nwicu_perm)
    print(f"Wrote results to {OUT_DIR}")
    print("MIMIC top group:")
    print(mimic_results.sort_values("pearson_profile_correlation", ascending=False).head(3).to_string(index=False))
    print("MIMIC permutation:", mimic_perm)
    print("NWICU top group:")
    print(nwicu_results.sort_values("pearson_profile_correlation", ascending=False).head(3).to_string(index=False))
    print("NWICU permutation:", nwicu_perm)
    print("eICU top group:")
    print(eicu_results.sort_values("pearson_profile_correlation", ascending=False).head(3).to_string(index=False))
    print("eICU permutation:", eicu_perm)


if __name__ == "__main__":
    main()