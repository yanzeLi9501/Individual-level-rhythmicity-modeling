from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NC_ROOT = ROOT.parent
OUTPUT = ROOT / "analysis_outputs"

COHORT = Path(
    os.environ.get(
        "WHU_PRIMARY_COHORT_CSV",
        NC_ROOT / "data" / "private" / "whu_primary_for_lgdi.csv",
    )
)
DIAGNOSIS = Path(
    os.environ.get(
        "WHU_DIAGNOSIS_CSV",
        NC_ROOT / "data" / "private" / "diagnosis.csv",
    )
)


def normalize_mrn(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\D", "", str(value))
    if not text:
        return None
    return str(int(text))


def normalize_icd(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    match = re.search(r"[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?", text)
    return "" if match is None else match.group(0)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT, usecols=["病案号"], dtype=str, low_memory=False)
    cohort["mrn_norm"] = cohort["病案号"].map(normalize_mrn)
    cohort_mrns = set(cohort["mrn_norm"].dropna())

    usecols = ["病案号", "诊断类型", "ICD10编码", "ICD10名称", "是否主诊断"]
    diag = pd.read_csv(DIAGNOSIS, usecols=usecols, encoding="gbk", dtype=str, low_memory=False)
    diag["mrn_norm"] = diag["病案号"].map(normalize_mrn)
    diag = diag[diag["mrn_norm"].isin(cohort_mrns)].copy()
    diag["icd10_norm"] = diag["ICD10编码"].map(normalize_icd)
    diag = diag[diag["icd10_norm"] != ""].copy()

    diag["any_j"] = diag["icd10_norm"].str.match(r"^J", na=False)
    diag["acute_j09_j18"] = diag["icd10_norm"].str.match(r"^J(09|1[0-8])", na=False)
    diag["influenza_j09_j11"] = diag["icd10_norm"].str.match(r"^J(09|10|11)", na=False)
    diag["pneumonia_j12_j18"] = diag["icd10_norm"].str.match(r"^J(1[2-8])", na=False)

    def patient_set(flag: str) -> set[str]:
        return set(diag.loc[diag[flag], "mrn_norm"])

    coded_patients = set(diag["mrn_norm"])
    any_j = patient_set("any_j")
    acute = patient_set("acute_j09_j18")
    influenza = patient_set("influenza_j09_j11")
    pneumonia = patient_set("pneumonia_j12_j18")

    summary = {
        "cohort_admissions": int(len(cohort)),
        "cohort_patients": int(len(cohort_mrns)),
        "patients_with_any_icd10_record": int(len(coded_patients)),
        "patients_with_any_j_code": int(len(any_j)),
        "patients_with_acute_j09_j18_code": int(len(acute)),
        "patients_with_j09_j11_influenza_code": int(len(influenza)),
        "patients_with_j12_j18_pneumonia_code": int(len(pneumonia)),
        "acute_j09_j18_percent_of_all_cohort_patients": 100.0 * len(acute) / len(cohort_mrns),
        "acute_j09_j18_percent_of_patients_with_icd10_records": 100.0 * len(acute) / len(coded_patients)
        if coded_patients
        else None,
        "acute_j09_j18_percent_of_any_j_patients": 100.0 * len(acute) / len(any_j) if any_j else None,
        "j09_j11_percent_of_all_cohort_patients": 100.0 * len(influenza) / len(cohort_mrns),
        "j12_j18_percent_of_all_cohort_patients": 100.0 * len(pneumonia) / len(cohort_mrns),
    }

    code_counts = (
        diag.loc[diag["acute_j09_j18"], "icd10_norm"]
        .value_counts()
        .rename_axis("icd10_code")
        .reset_index(name="diagnosis_records")
    )
    code_counts.to_csv(OUTPUT / "whu_j09_j18_code_counts.csv", index=False)
    with open(OUTPUT / "whu_respiratory_icd_overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
