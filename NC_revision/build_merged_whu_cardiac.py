#!/usr/bin/env python3
"""Build a WHU-primary plus expanded-cardiac union table keyed by patient record number."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
WHU_TABLE = BASE_DIR.parent / "data" / "private" / "readmission_output" / "all_admissions.csv"
CARDIAC_TABLE = BASE_DIR / "expanded_cardiac_wide_table.csv"
OUTPUT_TABLE = BASE_DIR / "merged_whu_cardiac_51k.csv"
SUMMARY_JSON = BASE_DIR / "merged_whu_cardiac_51k_summary.json"
SOURCE_COUNTS_CSV = BASE_DIR / "merged_whu_cardiac_51k_source_counts.csv"
PACKAGE_ANALYSIS = BASE_DIR / "resubmission_package_20260512" / "analysis_outputs"

EXPECTED_COLUMNS = ["病案号", "入院时间", "出院时间", "入院日期", "出院日期", "主要诊断", "日期"]
LABS = ["lab_WBC", "lab_CRP", "lab_HGB", "lab_ALB", "lab_CREA", "lab_GLU", "lab_K", "lab_Na"]
CARDIAC_LABS = {
    "白细胞": "lab_WBC",
    "超敏C反应蛋白": "lab_CRP",
    "血红蛋白": "lab_HGB",
    "白蛋白": "lab_ALB",
    "肌酐": "lab_CREA",
    "空腹血糖": "lab_GLU",
    "钾": "lab_K",
    "钠": "lab_Na",
}


def read_csv_best(path: Path, **kwargs: object) -> pd.DataFrame:
    best: tuple[int, pd.DataFrame] | None = None
    errors: list[str] = []
    read_kwargs = dict(kwargs)
    dtype = dict(read_kwargs.pop("dtype", {}) or {})
    for id_col in ["病案号", "病案号_norm", "mrn", "patient_id"]:
        dtype.setdefault(id_col, str)
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=dtype, low_memory=False, **read_kwargs)
        except Exception as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
            continue
        score = sum(col in frame.columns for col in EXPECTED_COLUMNS)
        if best is None or score > best[0]:
            best = (score, frame)
        if score >= 3:
            return frame
    if best is not None:
        return best[1]
    raise UnicodeError(f"Could not read {path}: {'; '.join(errors)}")


def normalize_mrn(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.replace("\ufeff", "", regex=False).str.replace("\u3000", " ", regex=False).str.strip()
    text = text.map(lambda value: re.sub(r"\s+", " ", value))
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.mask(text.str.lower().isin({"", "nan", "none", "null", "na", "n/a"}), "")


def coalesce_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    pieces = []
    for column in columns:
        if column in frame.columns:
            pieces.append(frame[column].fillna("").astype(str))
    if not pieces:
        return pd.Series("", index=frame.index)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out + " " + piece
    return out.str.strip()


def prepare_whu() -> pd.DataFrame:
    raw = read_csv_best(WHU_TABLE)
    data = pd.DataFrame(index=raw.index)
    data["病案号_norm"] = normalize_mrn(raw["病案号"])
    admit = pd.to_datetime(raw.get("入院日期"), errors="coerce")
    fallback_date = pd.to_datetime(raw.get("日期"), errors="coerce")
    data["入院时间"] = admit.fillna(fallback_date)
    data["出院时间"] = pd.to_datetime(raw.get("出院日期"), errors="coerce")
    data["主要诊断"] = coalesce_text(raw, ["诊断文本", "EMR_初步诊断", "EMR_出院记录"])
    data["上次诊断"] = ""
    for lab in LABS:
        data[lab] = pd.to_numeric(raw[lab], errors="coerce") if lab in raw.columns else np.nan
    data["source"] = "whu_primary"
    data = data[data["病案号_norm"].ne("") & data["入院时间"].notna()].copy()
    data = data.drop_duplicates(subset=["病案号_norm", "入院时间", "出院时间", "主要诊断"])
    return data


def prepare_cardiac() -> pd.DataFrame:
    raw = read_csv_best(CARDIAC_TABLE)
    data = pd.DataFrame(index=raw.index)
    data["病案号_norm"] = normalize_mrn(raw["病案号"])
    data["入院时间"] = pd.to_datetime(raw["入院时间"], errors="coerce")
    data["出院时间"] = pd.to_datetime(raw["出院时间"], errors="coerce")
    data["主要诊断"] = coalesce_text(raw, ["主要诊断"])
    data["上次诊断"] = coalesce_text(raw, ["上次诊断"])
    for source_col, target_col in CARDIAC_LABS.items():
        data[target_col] = pd.to_numeric(raw[source_col], errors="coerce") if source_col in raw.columns else np.nan
    data["source"] = "expanded_cardiac"
    data = data[data["病案号_norm"].ne("") & data["入院时间"].notna()].copy()
    data = data.drop_duplicates(subset=["病案号_norm", "入院时间", "出院时间", "主要诊断"])
    return data


def source_counts(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, frame in combined.groupby("patient_source", dropna=False):
        rows.append({
            "patient_source": source,
            "admissions": int(len(frame)),
            "unique_patient_record_numbers": int(frame["病案号_norm"].nunique()),
            "admissions_2019": int(frame[frame["入院时间"].dt.year.eq(2019)].shape[0]),
            "unique_patients_2019": int(frame.loc[frame["入院时间"].dt.year.eq(2019), "病案号_norm"].nunique()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    whu = prepare_whu()
    cardiac = prepare_cardiac()
    whu_patients = set(whu["病案号_norm"].dropna())
    cardiac_patients = set(cardiac["病案号_norm"].dropna())
    overlap = whu_patients & cardiac_patients
    whu_only = whu_patients - cardiac_patients
    cardiac_only = cardiac_patients - whu_patients

    whu_out = whu.copy()
    whu_out["patient_source"] = np.where(whu_out["病案号_norm"].isin(overlap), "both_prefer_whu", "whu_primary_only")
    cardiac_out = cardiac[cardiac["病案号_norm"].isin(cardiac_only)].copy()
    cardiac_out["patient_source"] = "expanded_cardiac_only"
    combined = pd.concat([whu_out, cardiac_out], ignore_index=True, sort=False)
    combined = combined.sort_values(["病案号_norm", "入院时间", "出院时间"]).copy()
    combined["病案号"] = combined["病案号_norm"]
    ordered_cols = ["病案号", "病案号_norm", "入院时间", "出院时间", "主要诊断", "上次诊断", "source", "patient_source", *LABS]
    combined[ordered_cols].to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")

    counts = source_counts(combined)
    counts.to_csv(SOURCE_COUNTS_CSV, index=False, encoding="utf-8-sig")
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "inputs": {"whu_primary": str(WHU_TABLE), "expanded_cardiac": str(CARDIAC_TABLE)},
        "patient_record_number_union": {
            "whu_primary_unique": int(len(whu_patients)),
            "expanded_cardiac_unique": int(len(cardiac_patients)),
            "overlap": int(len(overlap)),
            "whu_primary_only": int(len(whu_only)),
            "expanded_cardiac_only": int(len(cardiac_only)),
            "union": int(len(whu_patients | cardiac_patients)),
        },
        "admission_rows": {
            "whu_primary_prepared": int(len(whu)),
            "expanded_cardiac_prepared": int(len(cardiac)),
            "merged_prefer_whu_for_overlap": int(len(combined)),
        },
        "date_range": f"{combined['入院时间'].min().date()} to {combined['入院时间'].max().date()}",
        "shared_features": ["los_days derived from dates", *LABS],
        "outputs": {"merged_table": str(OUTPUT_TABLE), "source_counts": str(SOURCE_COUNTS_CSV)},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    PACKAGE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    for path in [SUMMARY_JSON, SOURCE_COUNTS_CSV]:
        shutil.copy2(path, PACKAGE_ANALYSIS / path.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
