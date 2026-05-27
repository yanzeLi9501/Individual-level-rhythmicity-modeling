from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "expanded_cardiac_table_schema.csv"
ORIGINAL_WIDE_PATH = Path(
    r"data\readmission_output\figures_v2\merge\final_preprocessed_data_new.csv"
)

EXPANDED_WIDE_PATH = BASE_DIR / "expanded_cardiac_wide_table.csv"
COMBINED_WIDE_PATH = BASE_DIR / "combined_cardiac_wide_table.csv"
SUMMARY_JSON_PATH = BASE_DIR / "expanded_cardiac_wide_table_summary.json"
SUMMARY_MD_PATH = BASE_DIR / "expanded_cardiac_wide_table_summary.md"

MISSING_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "无",
    "未填写",
    "未提供",
    "不详",
    "未知",
    "-",
    "--",
    "---",
    "/",
    "\\",
}

VALIDATION_COLUMNS = [
    "病案号",
    "入院时间",
    "出院时间",
    "主要诊断",
    "入院科室",
    "出院科室",
    "上次诊断",
    "上次入院科室",
    "上次出院科室",
    "上次入院时间",
    "上次出院时间",
    "时间差",
    "白细胞",
    "超敏C反应蛋白",
    "血红蛋白",
    "白蛋白",
    "肌酐",
    "空腹血糖",
    "钾",
    "钠",
]

LAB_COLUMNS = [
    "白细胞",
    "超敏C反应蛋白",
    "血红蛋白",
    "白蛋白",
    "肌酐",
    "空腹血糖",
    "钾",
    "钠",
]

LAB_EXACT_MAP = {
    "白细胞": "白细胞",
    "超敏C反应蛋白": "超敏C反应蛋白",
    "血红蛋白": "血红蛋白",
    "白蛋白": "白蛋白",
    "肌酐": "肌酐",
    "肌酐(酶法)": "肌酐",
    "空腹血糖": "空腹血糖",
    "钾": "钾",
    "钠": "钠",
}

CHUNK_SIZE = 200_000


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in MISSING_VALUES:
        return ""
    return text


def normalize_id(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def split_identifier_values(value: object) -> list[str]:
    text = normalize_id(value)
    if not text:
        return []
    tokens = re.split(r"\s*(?:\|\||[;；,，、])\s*", text)
    return [normalize_id(token) for token in tokens if normalize_id(token)]


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def log(message: str) -> None:
    print(message, flush=True)


def parse_datetime_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def parse_numeric(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = clean_text(value)
    if not text:
        return np.nan
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def read_schema() -> pd.DataFrame:
    schema = pd.read_csv(SCHEMA_PATH)
    schema["path"] = schema["relative_path"].map(lambda value: BASE_DIR / str(value))
    return schema[schema["path"].map(Path.exists)].copy()


def table_rows(schema: pd.DataFrame, table_names: set[str]) -> pd.DataFrame:
    return schema[schema["table_name"].isin(table_names)].copy()


def chunk_reader(path: Path, encoding: str, wanted_columns: set[str]):
    return pd.read_csv(
        path,
        encoding=encoding,
        encoding_errors="replace",
        usecols=lambda column: column in wanted_columns,
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )


def normalize_identifier_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in ["病案号", "门诊号", "住院号", "就诊标识（医渡云计算）"]:
        if column not in dataframe.columns:
            dataframe[column] = ""
        dataframe[column] = dataframe[column].map(normalize_id)
    return dataframe


def collect_outpatient_to_mrn(schema: pd.DataFrame) -> tuple[dict[str, str], dict[str, int]]:
    wanted_columns = {"病案号", "患者病案号（病案首页）", "门诊号"}
    source_tables = {"患者", "索引", "病案首页", "病案首页出院诊断"}
    mapping_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows_scanned = 0

    for _, table_row in table_rows(schema, source_tables).iterrows():
        log(f"[ids] scanning {table_row['table_name']}: {table_row['path'].name}")
        for chunk in chunk_reader(table_row["path"], str(table_row["encoding"]), wanted_columns):
            mrn_series = chunk.get("病案号", pd.Series([""] * len(chunk))).map(normalize_id)
            if "患者病案号（病案首页）" in chunk.columns:
                fallback_mrn = chunk["患者病案号（病案首页）"].map(normalize_id)
                mrn_series = mrn_series.mask(mrn_series.eq(""), fallback_mrn)
            outpatient_series = chunk.get("门诊号", pd.Series([""] * len(chunk))).map(normalize_id)
            for medical_record_number, outpatient_value in zip(mrn_series, outpatient_series):
                rows_scanned += 1
                if not medical_record_number or not outpatient_value:
                    continue
                for token in split_identifier_values(outpatient_value):
                    mapping_counts[token][medical_record_number] += 1

    resolved: dict[str, str] = {}
    ambiguous = 0
    for outpatient_value, counter in mapping_counts.items():
        if not counter:
            continue
        most_common = counter.most_common(2)
        if len(most_common) == 1 or most_common[0][1] >= most_common[1][1]:
            resolved[outpatient_value] = most_common[0][0]
        else:
            ambiguous += 1
    return resolved, {"outpatient_values": len(mapping_counts), "resolved": len(resolved), "ambiguous": ambiguous, "rows_scanned": rows_scanned}


def fill_mrn_from_outpatient(dataframe: pd.DataFrame, outpatient_to_mrn: dict[str, str]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    if "病案号" not in dataframe.columns:
        dataframe["病案号"] = ""
    if "门诊号" not in dataframe.columns:
        dataframe["门诊号"] = ""
    dataframe["病案号"] = dataframe["病案号"].map(normalize_id)
    missing = dataframe["病案号"].eq("")
    if missing.any():
        dataframe.loc[missing, "病案号"] = dataframe.loc[missing, "门诊号"].map(
            lambda value: next(
                (outpatient_to_mrn[token] for token in split_identifier_values(value) if token in outpatient_to_mrn),
                "",
            )
        )
    return dataframe


def map_rows_to_admissions(dataframe: pd.DataFrame, lookup: dict[str, str | None]) -> pd.Series:
    dataframe = normalize_identifier_columns(dataframe)
    result = pd.Series("", index=dataframe.index, dtype=object)
    medical_record_number = dataframe["病案号"]

    visit_id = dataframe["就诊标识（医渡云计算）"]
    visit_mask = result.eq("") & medical_record_number.ne("") & visit_id.ne("")
    if visit_mask.any():
        keys = "v|" + medical_record_number[visit_mask] + "|" + visit_id[visit_mask]
        result.loc[visit_mask] = keys.map(lookup).fillna("").map(lambda value: value or "")

    inpatient_number = dataframe["住院号"]
    inpatient_mask = result.eq("") & medical_record_number.ne("") & inpatient_number.ne("")
    if inpatient_mask.any():
        keys = "i|" + medical_record_number[inpatient_mask] + "|" + inpatient_number[inpatient_mask]
        result.loc[inpatient_mask] = keys.map(lookup).fillna("").map(lambda value: value or "")

    outpatient_mask = result.eq("") & medical_record_number.ne("") & dataframe["门诊号"].ne("")
    if outpatient_mask.any():
        result.loc[outpatient_mask] = dataframe.loc[outpatient_mask].apply(
            lambda row: next(
                (
                    lookup.get(f"o|{row['病案号']}|{token}")
                    for token in split_identifier_values(row.get("门诊号", ""))
                    if lookup.get(f"o|{row['病案号']}|{token}")
                ),
                "",
            ),
            axis=1,
        )
    return result


def add_admission_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = normalize_identifier_columns(dataframe)
    admit_dates = parse_datetime_series(dataframe.get("入院时间", pd.Series([""] * len(dataframe))))
    discharge_dates = parse_datetime_series(dataframe.get("出院时间", pd.Series([""] * len(dataframe))))
    outpatient_norm = dataframe["门诊号"].map(lambda value: "|".join(split_identifier_values(value)))
    dataframe["admit_dt"] = admit_dates
    dataframe["discharge_dt"] = discharge_dates
    dataframe["admission_key"] = "d|" + dataframe["病案号"] + "|" + admit_dates.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("") + "|" + discharge_dates.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    visit_mask = dataframe["就诊标识（医渡云计算）"].ne("")
    dataframe.loc[visit_mask, "admission_key"] = "v|" + dataframe.loc[visit_mask, "病案号"] + "|" + dataframe.loc[visit_mask, "就诊标识（医渡云计算）"]

    inpatient_mask = ~visit_mask & dataframe["住院号"].ne("")
    dataframe.loc[inpatient_mask, "admission_key"] = "i|" + dataframe.loc[inpatient_mask, "病案号"] + "|" + dataframe.loc[inpatient_mask, "住院号"]

    outpatient_mask = ~visit_mask & ~inpatient_mask & outpatient_norm.ne("")
    dataframe.loc[outpatient_mask, "admission_key"] = "o|" + dataframe.loc[outpatient_mask, "病案号"] + "|" + outpatient_norm[outpatient_mask]
    return dataframe


def admission_lab_lookup(admissions: pd.DataFrame) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for _, row in admissions.iterrows():
        medical_record_number = normalize_id(row.get("病案号", ""))
        admission_key = clean_text(row.get("admission_key", ""))
        candidates: list[str] = []
        visit_id = normalize_id(row.get("就诊标识（医渡云计算）", ""))
        inpatient_number = normalize_id(row.get("住院号", ""))
        outpatient_number = normalize_id(row.get("门诊号", ""))
        if visit_id:
            candidates.append(f"v|{medical_record_number}|{visit_id}")
        if inpatient_number:
            candidates.append(f"i|{medical_record_number}|{inpatient_number}")
        for token in split_identifier_values(outpatient_number):
            candidates.append(f"o|{medical_record_number}|{token}")
        for candidate in candidates:
            existing = lookup.get(candidate)
            if existing is None and candidate in lookup:
                continue
            if existing and existing != admission_key:
                lookup[candidate] = None
            else:
                lookup[candidate] = admission_key
    return lookup


def match_row_to_admission(row: pd.Series, lookup: dict[str, str | None]) -> str:
    medical_record_number = normalize_id(row.get("病案号", ""))
    if not medical_record_number:
        return ""
    visit_id = normalize_id(row.get("就诊标识（医渡云计算）", ""))
    if visit_id:
        key = lookup.get(f"v|{medical_record_number}|{visit_id}")
        if key:
            return key
    inpatient_number = normalize_id(row.get("住院号", ""))
    if inpatient_number:
        key = lookup.get(f"i|{medical_record_number}|{inpatient_number}")
        if key:
            return key
    for token in split_identifier_values(row.get("门诊号", "")):
        key = lookup.get(f"o|{medical_record_number}|{token}")
        if key:
            return key
    return ""


def collapse_admissions(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    dataframe = add_admission_keys(dataframe)
    dataframe = dataframe[dataframe["病案号"].ne("") & dataframe["admit_dt"].notna()].copy()
    dataframe["source_priority"] = dataframe["record_source"].map({"病案首页": 0, "病历": 1}).fillna(2)
    dataframe = dataframe.sort_values(["source_priority", "admit_dt"])
    aggregate_columns = [
        "病案号",
        "门诊号",
        "住院号",
        "就诊标识（医渡云计算）",
        "入院时间",
        "出院时间",
        "主要诊断",
        "入院科室",
        "出院科室",
        "性别",
        "出生日期",
        "record_source",
    ]
    grouped = dataframe.groupby("admission_key", sort=False)
    rows: list[dict[str, object]] = []
    for admission_key, group in grouped:
        row = {"admission_key": admission_key}
        for column in aggregate_columns:
            row[column] = first_nonempty(group[column]) if column in group.columns else ""
        rows.append(row)
    collapsed = pd.DataFrame(rows)
    collapsed = add_admission_keys(collapsed)
    return collapsed


def load_homepage_admissions(schema: pd.DataFrame, outpatient_to_mrn: dict[str, str]) -> pd.DataFrame:
    wanted_columns = {
        "病案号",
        "患者病案号（病案首页）",
        "门诊号",
        "住院号",
        "就诊标识（医渡云计算）",
        "出院诊断主诊断（病案首页）",
        "入院时间（病案首页）",
        "出院时间（病案首页）",
        "入院科室（病案首页）",
        "出院科室（病案首页）",
        "患者性别（病案首页）",
        "患者出生日期（病案首页）",
    }
    pieces: list[pd.DataFrame] = []
    for _, table_row in table_rows(schema, {"病案首页"}).iterrows():
        for chunk in chunk_reader(table_row["path"], str(table_row["encoding"]), wanted_columns):
            if "患者病案号（病案首页）" in chunk.columns:
                mrn_series = chunk["病案号"].map(normalize_id) if "病案号" in chunk.columns else pd.Series([""] * len(chunk), index=chunk.index)
                chunk["病案号"] = mrn_series.mask(
                    mrn_series.eq(""),
                    chunk["患者病案号（病案首页）"].map(normalize_id),
                )
            renamed = chunk.rename(
                columns={
                    "出院诊断主诊断（病案首页）": "主要诊断",
                    "入院时间（病案首页）": "入院时间",
                    "出院时间（病案首页）": "出院时间",
                    "入院科室（病案首页）": "入院科室",
                    "出院科室（病案首页）": "出院科室",
                    "患者性别（病案首页）": "性别",
                    "患者出生日期（病案首页）": "出生日期",
                }
            )
            renamed["record_source"] = "病案首页"
            renamed = fill_mrn_from_outpatient(renamed, outpatient_to_mrn)
            pieces.append(renamed)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def load_medical_record_admissions(schema: pd.DataFrame, outpatient_to_mrn: dict[str, str]) -> pd.DataFrame:
    wanted_columns = {
        "病案号",
        "门诊号",
        "住院号",
        "就诊标识（医渡云计算）",
        "入院(就诊)时间",
        "出院时间",
        "主诊断（医渡云计算）",
        "出院诊断",
        "入院（就诊）科室",
        "出院科室",
        "性别",
        "出生日期",
    }
    collapsed_chunks: list[pd.DataFrame] = []
    for _, table_row in table_rows(schema, {"病历"}).iterrows():
        for chunk in chunk_reader(table_row["path"], str(table_row["encoding"]), wanted_columns):
            renamed = chunk.rename(
                columns={
                    "入院(就诊)时间": "入院时间",
                    "主诊断（医渡云计算）": "主要诊断",
                    "入院（就诊）科室": "入院科室",
                }
            )
            if "主要诊断" not in renamed.columns and "出院诊断" in renamed.columns:
                renamed["主要诊断"] = renamed["出院诊断"]
            elif "出院诊断" in renamed.columns:
                renamed["主要诊断"] = renamed["主要诊断"].where(renamed["主要诊断"].map(clean_text).ne(""), renamed["出院诊断"])
            renamed["record_source"] = "病历"
            renamed = fill_mrn_from_outpatient(renamed, outpatient_to_mrn)
            collapsed_chunks.append(collapse_admissions(renamed))
    return pd.concat(collapsed_chunks, ignore_index=True) if collapsed_chunks else pd.DataFrame()


def map_lab_item(raw_item: object, standardized_item: object) -> str:
    raw_text = clean_text(raw_item)
    standardized_text = clean_text(standardized_item)
    if raw_text in LAB_EXACT_MAP:
        return LAB_EXACT_MAP[raw_text]
    if "静脉血" not in standardized_text:
        return ""
    if "超敏C反应蛋白" in standardized_text or "hs-CRP" in standardized_text:
        return "超敏C反应蛋白"
    if "血红蛋白(Hb)" in standardized_text:
        return "血红蛋白"
    if "白蛋白(ALB)" in standardized_text:
        return "白蛋白"
    if "肌酐" in standardized_text or "Crea" in standardized_text:
        return "肌酐"
    if "葡萄糖" in standardized_text and "空腹" in standardized_text:
        return "空腹血糖"
    if "钾离子" in standardized_text:
        return "钾"
    if "钠离子" in standardized_text:
        return "钠"
    if "白细胞" in standardized_text and "尿" not in standardized_text and "粪" not in standardized_text:
        return "白细胞"
    return ""


def load_diagnosis_text(schema: pd.DataFrame, admissions: pd.DataFrame) -> dict[str, str]:
    lookup = admission_lab_lookup(admissions)
    wanted_columns = {
        "病案号",
        "门诊号",
        "住院号",
        "就诊标识（医渡云计算）",
        "出院诊断顺位（病案首页）",
        "出院诊断（病案首页）",
        "诊断名称（原始字段）",
        "ICD10名称",
        "诊断顺位",
        "是否主诊断",
    }
    records: list[pd.DataFrame] = []
    scan_stats = Counter()
    for _, table_row in table_rows(schema, {"病案首页出院诊断", "诊断"}).iterrows():
        log(f"[diagnosis] scanning {table_row['table_name']}: {table_row['path'].name}")
        for chunk in chunk_reader(table_row["path"], str(table_row["encoding"]), wanted_columns):
            scan_stats["rows"] += len(chunk)
            chunk = normalize_identifier_columns(chunk)
            if "出院诊断（病案首页）" in chunk.columns:
                diagnosis_values = chunk["出院诊断（病案首页）"]
                order_values = chunk.get("出院诊断顺位（病案首页）", pd.Series([""] * len(chunk)))
            else:
                diagnosis_values = chunk.get("诊断名称（原始字段）", pd.Series([""] * len(chunk)))
                if "ICD10名称" in chunk.columns:
                    diagnosis_values = diagnosis_values.where(diagnosis_values.map(clean_text).ne(""), chunk["ICD10名称"])
                order_values = chunk.get("诊断顺位", pd.Series([""] * len(chunk)))
            admission_keys = map_rows_to_admissions(chunk, lookup)
            diagnosis_frame = pd.DataFrame(
                {
                    "admission_key": admission_keys,
                    "diagnosis": diagnosis_values.map(clean_text),
                    "order_number": order_values.map(parse_numeric).fillna(999).astype(int),
                }
            )
            diagnosis_frame = diagnosis_frame[diagnosis_frame["admission_key"].ne("") & diagnosis_frame["diagnosis"].ne("")]
            if not diagnosis_frame.empty:
                records.append(diagnosis_frame)
                scan_stats["matched_rows"] += len(diagnosis_frame)

    log(f"[diagnosis] rows={scan_stats['rows']:,}; matched={scan_stats['matched_rows']:,}")
    if not records:
        return {}
    diagnosis_records = pd.concat(records, ignore_index=True)
    diagnosis_records = diagnosis_records.sort_values(["admission_key", "order_number"])
    merged: dict[str, str] = {}
    for admission_key, group in diagnosis_records.groupby("admission_key", sort=False):
        seen: set[str] = set()
        ordered: list[str] = []
        for diagnosis in group["diagnosis"]:
            if diagnosis not in seen:
                seen.add(diagnosis)
                ordered.append(diagnosis)
        merged[admission_key] = "; ".join(ordered)
    return merged


def map_lab_items(raw_items: pd.Series, standardized_items: pd.Series) -> pd.Series:
    raw_clean = raw_items.map(clean_text)
    standardized_clean = standardized_items.map(clean_text)
    result = raw_clean.map(LAB_EXACT_MAP).fillna("")
    venous = standardized_clean.str.contains("静脉血", na=False)
    fill_mask = result.eq("") & venous & (standardized_clean.str.contains("超敏C反应蛋白|hs-CRP", case=False, na=False))
    result.loc[fill_mask] = "超敏C反应蛋白"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains(r"血红蛋白\(Hb\)", na=False)
    result.loc[fill_mask] = "血红蛋白"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains(r"白蛋白\(ALB\)", na=False)
    result.loc[fill_mask] = "白蛋白"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains("肌酐|Crea", case=False, na=False)
    result.loc[fill_mask] = "肌酐"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains("葡萄糖", na=False) & standardized_clean.str.contains("空腹", na=False)
    result.loc[fill_mask] = "空腹血糖"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains("钾离子", na=False)
    result.loc[fill_mask] = "钾"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains("钠离子", na=False)
    result.loc[fill_mask] = "钠"
    fill_mask = result.eq("") & venous & standardized_clean.str.contains("白细胞", na=False) & ~standardized_clean.str.contains("尿|粪", na=False)
    result.loc[fill_mask] = "白细胞"
    return result


def load_selected_labs(schema: pd.DataFrame, admissions: pd.DataFrame, outpatient_to_mrn: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    lookup = admission_lab_lookup(admissions)
    wanted_columns = {
        "病案号",
        "门诊号",
        "住院号",
        "就诊标识（医渡云计算）",
        "检验报告时间",
        "检验标本采样时间",
        "检验子项名称",
        "检验结果(定量)",
        "检验值（医院原始值）",
        "检验子项目名称（医渡标准化）",
    }
    best_values: dict[tuple[str, str], tuple[pd.Timestamp, float]] = {}
    scan_stats = Counter()
    lab_tables = schema[schema["table_name"].map(lambda name: str(name).startswith("检验."))].copy()

    for _, table_row in lab_tables.iterrows():
        scan_stats["files"] += 1
        log(f"[labs] scanning {table_row['table_name']}: {table_row['path'].name}")
        for chunk in chunk_reader(table_row["path"], str(table_row["encoding"]), wanted_columns):
            scan_stats["rows"] += len(chunk)
            chunk = normalize_identifier_columns(chunk)
            chunk = fill_mrn_from_outpatient(chunk, outpatient_to_mrn)
            raw_items = chunk.get("检验子项名称", pd.Series([""] * len(chunk)))
            standardized_items = chunk.get("检验子项目名称（医渡标准化）", pd.Series([""] * len(chunk)))
            chunk["target_lab"] = map_lab_items(raw_items, standardized_items)
            chunk = chunk[chunk["target_lab"].ne("")].copy()
            if chunk.empty:
                continue
            scan_stats["target_rows"] += len(chunk)
            report_time = parse_datetime_series(
                chunk.get("检验报告时间", pd.Series([""] * len(chunk))).where(
                    chunk.get("检验报告时间", pd.Series([""] * len(chunk))).map(clean_text).ne(""),
                    chunk.get("检验标本采样时间", pd.Series([""] * len(chunk))),
                )
            )
            quantitative = chunk.get("检验结果(定量)", pd.Series([np.nan] * len(chunk))).map(parse_numeric)
            original_value = chunk.get("检验值（医院原始值）", pd.Series([np.nan] * len(chunk))).map(parse_numeric)
            values = quantitative.where(quantitative.notna(), original_value)
            chunk["report_time"] = report_time
            chunk["lab_value"] = values
            chunk = chunk[chunk["lab_value"].notna()].copy()
            scan_stats["numeric_target_rows"] += len(chunk)
            chunk["admission_key"] = map_rows_to_admissions(chunk, lookup)
            unmatched = chunk["admission_key"].eq("").sum()
            scan_stats["unmatched_target_rows"] += int(unmatched)
            chunk = chunk[chunk["admission_key"].ne("")].copy()
            if chunk.empty:
                continue
            chunk["report_time"] = chunk["report_time"].fillna(pd.Timestamp.max)
            reduced = (
                chunk.sort_values("report_time")
                .drop_duplicates(["admission_key", "target_lab"], keep="first")
                [["admission_key", "target_lab", "report_time", "lab_value"]]
            )
            for admission_key, target_lab, timestamp, lab_value in reduced.itertuples(index=False, name=None):
                dict_key = (admission_key, target_lab)
                previous = best_values.get(dict_key)
                if previous is None or timestamp < previous[0]:
                    best_values[dict_key] = (timestamp, float(lab_value))

        log(
            "[labs] cumulative rows="
            f"{scan_stats['rows']:,}; target={scan_stats['target_rows']:,}; "
            f"numeric={scan_stats['numeric_target_rows']:,}; matched pairs={len(best_values):,}"
        )

    records = [
        {"admission_key": admission_key, "lab_name": lab_name, "lab_value": value}
        for (admission_key, lab_name), (_, value) in best_values.items()
    ]
    if not records:
        return pd.DataFrame(columns=["admission_key", *LAB_COLUMNS]), dict(scan_stats)
    labs = pd.DataFrame(records).pivot_table(index="admission_key", columns="lab_name", values="lab_value", aggfunc="first").reset_index()
    for column in LAB_COLUMNS:
        if column not in labs.columns:
            labs[column] = np.nan
    return labs[["admission_key", *LAB_COLUMNS]], dict(scan_stats)


def add_previous_visit_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["入院时间"] = parse_datetime_series(dataframe["入院时间"])
    dataframe["出院时间"] = parse_datetime_series(dataframe["出院时间"])
    dataframe = dataframe.sort_values(["病案号", "入院时间", "出院时间"]).reset_index(drop=True)
    grouped = dataframe.groupby("病案号", sort=False)
    dataframe["上次诊断"] = grouped["主要诊断"].shift(1).fillna("首诊记录")
    dataframe["上次入院科室"] = grouped["入院科室"].shift(1)
    dataframe["上次出院科室"] = grouped["出院科室"].shift(1)
    dataframe["上次入院时间"] = grouped["入院时间"].shift(1)
    dataframe["上次出院时间"] = grouped["出院时间"].shift(1)
    dataframe["时间差"] = (dataframe["入院时间"] - dataframe["上次出院时间"]).dt.days
    dataframe["时间差"] = dataframe["时间差"].fillna(0).clip(lower=0).astype(int)
    for column in ["入院时间", "出院时间", "上次入院时间", "上次出院时间"]:
        dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce").dt.strftime("%Y-%m-%d")
    dataframe[["上次入院科室", "上次出院科室"]] = dataframe[["上次入院科室", "上次出院科室"]].fillna("")
    return dataframe


def build_expanded_table(schema: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    outpatient_to_mrn, mapping_stats = collect_outpatient_to_mrn(schema)
    homepage = load_homepage_admissions(schema, outpatient_to_mrn)
    medical_records = load_medical_record_admissions(schema, outpatient_to_mrn)
    admissions = collapse_admissions(pd.concat([homepage, medical_records], ignore_index=True))

    diagnosis_by_admission = load_diagnosis_text(schema, admissions)
    if diagnosis_by_admission:
        diagnosis_series = admissions["admission_key"].map(diagnosis_by_admission).fillna("")
        admissions["主要诊断"] = admissions["主要诊断"].where(admissions["主要诊断"].map(clean_text).ne(""), diagnosis_series)
        admissions["主要诊断"] = [
            diagnosis if clean_text(extra) == "" or clean_text(extra) in clean_text(diagnosis) else f"{clean_text(diagnosis)}; {clean_text(extra)}"
            for diagnosis, extra in zip(admissions["主要诊断"], diagnosis_series)
        ]

    labs, lab_stats = load_selected_labs(schema, admissions, outpatient_to_mrn)
    expanded = admissions.merge(labs, on="admission_key", how="left")
    expanded = add_previous_visit_columns(expanded)
    for column in VALIDATION_COLUMNS:
        if column not in expanded.columns:
            expanded[column] = np.nan
    expanded = expanded[VALIDATION_COLUMNS].copy()

    stats = {
        "outpatient_to_mrn": mapping_stats,
        "homepage_rows_loaded": int(len(homepage)),
        "medical_record_visit_rows_loaded": int(len(medical_records)),
        "expanded_admissions": int(len(expanded)),
        "expanded_unique_patients": int(expanded["病案号"].nunique()),
        "expanded_date_min": str(pd.to_datetime(expanded["入院时间"], errors="coerce").min().date()),
        "expanded_date_max": str(pd.to_datetime(expanded["入院时间"], errors="coerce").max().date()),
        "lab_scan": lab_stats,
        "lab_nonmissing": {column: int(expanded[column].notna().sum()) for column in LAB_COLUMNS},
    }
    return expanded, stats


def load_original_table() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_WIDE_PATH, usecols=VALIDATION_COLUMNS, low_memory=False)
    original["病案号"] = original["病案号"].map(normalize_id)
    return original.copy()


def build_combined_table(original: pd.DataFrame, expanded: pd.DataFrame) -> pd.DataFrame:
    original = original.copy()
    expanded = expanded.copy()
    original["病案号"] = original["病案号"].map(normalize_id)
    expanded["病案号"] = expanded["病案号"].map(normalize_id)
    original["_source_priority"] = 0
    expanded["_source_priority"] = 1
    combined = pd.concat([original, expanded], ignore_index=True)
    combined["_admit"] = parse_datetime_series(combined["入院时间"])
    combined["_discharge"] = parse_datetime_series(combined["出院时间"])
    combined["_dedup_key"] = (
        combined["病案号"].map(normalize_id)
        + "|"
        + combined["_admit"].dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + combined["_discharge"].dt.strftime("%Y-%m-%d").fillna("")
    )
    combined = combined.sort_values(["_source_priority", "_admit"])
    combined = combined.drop_duplicates("_dedup_key", keep="first")
    combined = combined.drop(columns=["_source_priority", "_admit", "_discharge", "_dedup_key"])
    return combined[VALIDATION_COLUMNS].copy()


def summarize_table(dataframe: pd.DataFrame, prefix: str) -> dict[str, object]:
    admit_times = parse_datetime_series(dataframe["入院时间"])
    return {
        f"{prefix}_admissions": int(len(dataframe)),
        f"{prefix}_unique_patients": int(dataframe["病案号"].map(normalize_id).nunique()),
        f"{prefix}_date_min": str(admit_times.min().date()) if admit_times.notna().any() else "",
        f"{prefix}_date_max": str(admit_times.max().date()) if admit_times.notna().any() else "",
        f"{prefix}_lab_nonmissing": {column: int(dataframe[column].notna().sum()) for column in LAB_COLUMNS},
    }


def write_summary(summary: dict[str, object]) -> None:
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Expanded cardiac wide-table build summary",
        "",
        "## Cohort sizes",
        "",
        f"- Original 20K wide admissions: {summary['original']['original_admissions']:,}",
        f"- Original unique 病案号 in validation wide table: {summary['original']['original_unique_patients']:,}",
        f"- Expanded CSV-derived admissions: {summary['expanded']['expanded_admissions']:,}",
        f"- Expanded CSV-derived unique 病案号 in validation wide table: {summary['expanded']['expanded_unique_patients']:,}",
        f"- Combined deduplicated admissions: {summary['combined']['combined_admissions']:,}",
        f"- Combined deduplicated unique 病案号 in combined wide table: {summary['combined']['combined_unique_patients']:,}",
        "",
        "## Date ranges",
        "",
        f"- Original: {summary['original']['original_date_min']} to {summary['original']['original_date_max']}",
        f"- Expanded: {summary['expanded']['expanded_date_min']} to {summary['expanded']['expanded_date_max']}",
        f"- Combined: {summary['combined']['combined_date_min']} to {summary['combined']['combined_date_max']}",
        "",
        "## Selected lab availability in expanded table",
        "",
    ]
    for column, count in summary["expanded"]["expanded_lab_nonmissing"].items():
        lines.append(f"- {column}: {count:,}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{EXPANDED_WIDE_PATH.name}`",
            f"- `{COMBINED_WIDE_PATH.name}`",
            f"- `{SUMMARY_JSON_PATH.name}`",
        ]
    )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    schema = read_schema()
    expanded, expanded_stats = build_expanded_table(schema)
    original = load_original_table()
    combined = build_combined_table(original, expanded)

    expanded.to_csv(EXPANDED_WIDE_PATH, index=False, encoding="utf-8-sig")
    combined.to_csv(COMBINED_WIDE_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "original": summarize_table(original, "original"),
        "expanded": {**summarize_table(expanded, "expanded"), **expanded_stats},
        "combined": summarize_table(combined, "combined"),
        "paths": {
            "expanded_wide_table": str(EXPANDED_WIDE_PATH),
            "combined_wide_table": str(COMBINED_WIDE_PATH),
        },
    }
    write_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())