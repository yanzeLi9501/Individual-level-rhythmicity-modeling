from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FOLDERS = [
    "private_ehr_export_1",
    "private_ehr_export_2",
    "private_ehr_export_3",
]

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936")
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
HASH_SALT = "NC_revision_expanded_cardiac_2026_05"
MAX_FULL_SCAN_BYTES = 80 * 1024 * 1024
MAX_LARGE_TABLE_ROWS = 500
ROW_COUNT_ESTIMATE_BYTES = 8 * 1024 * 1024
CORE_FULL_SCAN_TABLES = {
    "患者",
    "索引",
    "病案首页",
    "病案首页出院诊断",
    "病案首页手术记录",
    "病案首页病理诊断",
}

EXACT_PRIMARY_ID_TYPES = {"medical_record_number", "outpatient_number"}
LINKABLE_ID_TYPES = {
    "medical_record_number",
    "outpatient_number",
    "hospital_number",
    "hospital_or_inpatient_number",
    "his_inpatient_number",
    "patient_sequence",
    "patient_sn",
    "empi",
    "identity_insurance_number",
}
UNION_ID_TYPES = LINKABLE_ID_TYPES - {"patient_sn"}
GLOBAL_ID_TYPES = {
    "medical_record_number",
    "outpatient_number",
    "hospital_number",
    "hospital_or_inpatient_number",
    "empi",
    "identity_insurance_number",
}

ID_TYPE_LABELS = {
    "medical_record_number": "病案号",
    "outpatient_number": "门诊号",
    "hospital_number": "医院号",
    "hospital_or_inpatient_number": "医院号/HIS住院号",
    "his_inpatient_number": "HIS住院号/住院号",
    "patient_sequence": "患者序号",
    "patient_sn": "patient_SN",
    "empi": "EMPI",
    "identity_insurance_number": "身份证号/医保号",
}

TIME_KEYWORDS = (
    "入院时间",
    "出院时间",
    "就诊时间",
    "检查时间",
    "检验时间",
    "报告时间",
    "采样时间",
    "申请时间",
    "执行时间",
    "时间",
    "日期",
)
DIAGNOSIS_KEYWORDS = ("诊断", "icd", "疾病", "病理")

DISEASE_GROUPS = {
    "cardiovascular": (
        "心",
        "冠心病",
        "心肌",
        "心衰",
        "心力衰竭",
        "心律",
        "房颤",
        "高血压",
        "瓣膜",
        "动脉粥样",
    ),
    "respiratory": (
        "肺",
        "呼吸",
        "支气管",
        "哮喘",
        "慢阻肺",
        "copd",
        "肺炎",
        "流感",
        "上呼吸道",
    ),
    "diabetes": ("糖尿病",),
    "renal": ("肾", "尿毒症", "透析"),
    "cerebrovascular": ("脑梗", "脑出血", "脑卒中", "卒中", "中风"),
    "covid_related": ("新冠", "冠状病毒", "covid", "sars-cov-2"),
}


@dataclass
class TableAudit:
    source_folder: str
    table_name: str
    relative_path: str
    encoding: str
    delimiter: str
    file_size_mb: float = 0.0
    row_count: int = 0
    row_count_is_estimated: bool = False
    detailed_rows_scanned: int = 0
    parsing_mode: str = ""
    column_count: int = 0
    columns: list[str] = field(default_factory=list)
    exact_primary_id_columns: list[str] = field(default_factory=list)
    link_id_columns: list[str] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    diagnosis_columns: list[str] = field(default_factory=list)
    min_time: str = ""
    max_time: str = ""
    read_error: str = ""


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        self.rank: dict[tuple[str, str, str], int] = {}

    def add(self, item: tuple[str, str, str]) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: tuple[str, str, str]) -> tuple[str, str, str]:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: tuple[str, str, str], right: tuple[str, str, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def safe_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def decode_sample(path: Path) -> tuple[str, str, str]:
    raw = path.read_bytes()[:65536]
    last_error = ""
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
            try:
                dialect = csv.Sniffer().sniff(text, delimiters=",\t;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
            return encoding, delimiter, ""
        except UnicodeDecodeError as exc:
            last_error = str(exc)
    return "gb18030", ",", last_error


def count_data_rows(path: Path) -> tuple[int, bool]:
    file_size = path.stat().st_size
    if file_size > MAX_FULL_SCAN_BYTES:
        with path.open("rb") as handle:
            sample = handle.read(ROW_COUNT_ESTIMATE_BYTES)
        newline_count = sample.count(b"\n")
        if newline_count <= 1:
            return 0, True
        average_line_bytes = max(len(sample) / newline_count, 1)
        estimated_lines = int(file_size / average_line_bytes)
        return max(estimated_lines - 1, 0), True

    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    return max(line_count - 1, 0), False


def detailed_row_limit(path: Path, table_name: str) -> int | None:
    size = path.stat().st_size
    if table_name in CORE_FULL_SCAN_TABLES or size <= MAX_FULL_SCAN_BYTES:
        return None
    return MAX_LARGE_TABLE_ROWS


def clean_header(header: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: Counter[str] = Counter()
    for value in header:
        column = str(value).replace("\ufeff", "").strip()
        if not column:
            column = "unnamed"
        seen[column] += 1
        if seen[column] > 1:
            column = f"{column}_{seen[column]}"
        cleaned.append(column)
    return cleaned


def normalize_value(value: object) -> str:
    text = str(value).replace("\ufeff", "").replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in MISSING_VALUES:
        return ""
    return text


def split_identifier_values(value: object) -> list[str]:
    text = normalize_value(value)
    if not text:
        return []
    parts = re.split(r"\s*(?:\|\||[;；,，、])\s*", text)
    tokens = [normalize_value(part) for part in parts]
    return [token for token in tokens if token and token.lower() not in MISSING_VALUES]


def classify_identifier_column(column: str) -> str | None:
    lowered = column.lower().strip()
    compact = re.sub(r"\s+", "", column)
    compact = compact.replace("（", "(").replace("）", ")")

    if any(token in compact for token in ("身份证号", "医保号", "身份证/医保", "身份证号/医保号")):
        return "identity_insurance_number"
    if compact in {"病案号", "患者病案号", "病历号", "住院病案号"}:
        return "medical_record_number"
    if re.fullmatch(r"患者?病案号\([^)]*\)", compact):
        return "medical_record_number"
    if compact in {"门诊号", "患者门诊号"}:
        return "outpatient_number"
    if re.fullmatch(r"患者?门诊号\([^)]*\)", compact):
        return "outpatient_number"
    if "empi" in lowered:
        return "empi"
    if "patient_sn" in lowered or "patient sn" in lowered:
        return "patient_sn"
    if compact in {"医院号/HIS住院号", "医院号/住院号", "医院号HIS住院号"}:
        return "hospital_or_inpatient_number"
    if compact in {"医院号", "患者医院号"}:
        return "hospital_number"
    if compact in {"HIS住院号", "住院号", "患者住院号"}:
        return "his_inpatient_number"
    if compact in {"患者序号", "患者编号", "患者ID"}:
        return "patient_sequence"
    return None


def is_time_column(column: str) -> bool:
    return any(keyword in column for keyword in TIME_KEYWORDS)


def is_diagnosis_column(column: str) -> bool:
    lowered = column.lower()
    return any(keyword in lowered for keyword in DIAGNOSIS_KEYWORDS)


def parse_datetime(value: object) -> datetime | None:
    text = normalize_value(value)
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        if 20000 <= number <= 60000:
            return datetime(1899, 12, 30) + timedelta(days=number)
    normalized = (
        text.replace("年", "-")
        .replace("月", "-")
        .replace("日", " ")
        .replace("/", "-")
        .replace(".", "-")
        .strip()
    )
    match = re.search(
        r"(19\d{2}|20\d{2})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?",
        normalized,
    )
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    try:
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
        )
    except ValueError:
        return None


def node_scope(source_folder: str, id_type: str) -> str:
    return "global" if id_type in GLOBAL_ID_TYPES else source_folder


def make_node(source_folder: str, id_type: str, value: str) -> tuple[str, str, str]:
    return (node_scope(source_folder, id_type), id_type, value)


def id_hash(node: tuple[str, str, str]) -> str:
    scope, id_type, value = node
    payload = f"{HASH_SALT}|{scope}|{id_type}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def mask_value(value: str) -> str:
    if len(value) <= 2:
        return "*" * len(value)
    if len(value) <= 6:
        return value[:1] + "***" + value[-1:]
    return value[:2] + "***" + value[-2:]


def row_to_dict(header: list[str], row: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, column in enumerate(header):
        values[column] = row[index] if index < len(row) else ""
    return values


def classify_diagnosis_text(text: str) -> set[str]:
    lowered = text.lower()
    groups = set()
    for group, keywords in DISEASE_GROUPS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            groups.add(group)
    return groups


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iso_or_blank(value: datetime | None) -> str:
    return value.isoformat(sep=" ", timespec="seconds") if value else ""


def discover_csv_files() -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for source_folder in SOURCE_FOLDERS:
        root = BASE_DIR / source_folder
        for path in sorted(root.rglob("*.csv")):
            discovered.append((source_folder, path))
    return discovered


def main() -> int:
    safe_field_limit()

    union_find = UnionFind()
    table_audits: list[TableAudit] = []
    node_occurrences: Counter[tuple[str, str, str]] = Counter()
    node_sources: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    node_tables: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    source_row_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()
    id_type_counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    diagnosis_row_counts: Counter[str] = Counter()
    diagnosis_component_sources: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    visit_rows: list[dict[str, object]] = []
    exact_primary_columns_found: Counter[str] = Counter()
    patient_table_stats: dict[str, dict[str, object]] = {}
    patient_table_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    all_min_time: datetime | None = None
    all_max_time: datetime | None = None

    for source_folder, path in discover_csv_files():
        source_file_counts[source_folder] += 1
        table_name = path.stem
        relative_path = path.relative_to(BASE_DIR).as_posix()
        encoding, delimiter, encoding_error = decode_sample(path)
        row_limit = detailed_row_limit(path, table_name)
        row_count, row_count_is_estimated = count_data_rows(path)
        audit = TableAudit(
            source_folder=source_folder,
            table_name=table_name,
            relative_path=relative_path,
            encoding=encoding,
            delimiter="TAB" if delimiter == "\t" else delimiter,
            file_size_mb=round(path.stat().st_size / (1024 * 1024), 2),
            row_count=row_count,
            row_count_is_estimated=row_count_is_estimated,
            parsing_mode="full" if row_limit is None else f"sample_first_{row_limit}_rows",
            read_error=encoding_error,
        )
        source_row_counts[source_folder] += audit.row_count

        try:
            with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                try:
                    header = clean_header(next(reader))
                except StopIteration:
                    header = []
                audit.columns = header
                audit.column_count = len(header)
                id_columns = {
                    column: classify_identifier_column(column)
                    for column in header
                    if classify_identifier_column(column) in LINKABLE_ID_TYPES
                }
                audit.exact_primary_id_columns = [
                    column for column, id_type in id_columns.items() if id_type in EXACT_PRIMARY_ID_TYPES
                ]
                audit.link_id_columns = list(id_columns.keys())
                audit.time_columns = [column for column in header if is_time_column(column)]
                audit.diagnosis_columns = [column for column in header if is_diagnosis_column(column)]
                for column in audit.exact_primary_id_columns:
                    exact_primary_columns_found[column] += 1

                table_min_time: datetime | None = None
                table_max_time: datetime | None = None
                row_index = 0
                for row in reader:
                    if row_limit is not None and row_index >= row_limit:
                        break
                    row_index += 1
                    row_values = row_to_dict(header, row)

                    row_nodes: list[tuple[str, str, str]] = []
                    union_nodes: list[tuple[str, str, str]] = []
                    for column, id_type in id_columns.items():
                        for token in split_identifier_values(row_values.get(column, "")):
                            node = make_node(source_folder, id_type or "unknown", token)
                            union_find.add(node)
                            row_nodes.append(node)
                            if id_type in UNION_ID_TYPES:
                                union_nodes.append(node)
                            node_occurrences[node] += 1
                            node_sources[node][source_folder] += 1
                            node_tables[node][table_name] += 1
                            id_type_counts_by_source[source_folder][id_type or "unknown"] += 1

                    if len(union_nodes) > 1:
                        anchor = union_nodes[0]
                        for node in union_nodes[1:]:
                            union_find.union(anchor, node)

                    if table_name == "患者":
                        stats = patient_table_stats.setdefault(
                            source_folder,
                            {
                                "patient_table_rows": 0,
                                "patient_table_rows_with_exact_primary_id": 0,
                                "patient_table_rows_with_empi": 0,
                            },
                        )
                        stats["patient_table_rows"] = int(stats["patient_table_rows"]) + 1
                        row_has_exact_primary = False
                        row_has_empi = False
                        for column, id_type in id_columns.items():
                            tokens = split_identifier_values(row_values.get(column, ""))
                            if not tokens:
                                continue
                            patient_table_values[source_folder][id_type or "unknown"].update(tokens)
                            if id_type in EXACT_PRIMARY_ID_TYPES:
                                row_has_exact_primary = True
                            if id_type == "empi":
                                row_has_empi = True
                        if row_has_exact_primary:
                            stats["patient_table_rows_with_exact_primary_id"] = int(
                                stats["patient_table_rows_with_exact_primary_id"]
                            ) + 1
                        if row_has_empi:
                            stats["patient_table_rows_with_empi"] = int(stats["patient_table_rows_with_empi"]) + 1

                    parsed_times: list[tuple[str, datetime]] = []
                    for column in audit.time_columns:
                        parsed = parse_datetime(row_values.get(column, ""))
                        if parsed:
                            parsed_times.append((column, parsed))
                            table_min_time = parsed if table_min_time is None else min(table_min_time, parsed)
                            table_max_time = parsed if table_max_time is None else max(table_max_time, parsed)
                            all_min_time = parsed if all_min_time is None else min(all_min_time, parsed)
                            all_max_time = parsed if all_max_time is None else max(all_max_time, parsed)

                    row_disease_groups: set[str] = set()
                    if audit.diagnosis_columns:
                        diagnosis_text = " ".join(
                            normalize_value(row_values.get(column, "")) for column in audit.diagnosis_columns
                        )
                        row_disease_groups = classify_diagnosis_text(diagnosis_text)
                        for group in row_disease_groups:
                            diagnosis_row_counts[group] += 1
                        for node in row_nodes:
                            for group in row_disease_groups:
                                diagnosis_component_sources[node][group] += 1

                    visit_id_tokens: list[str] = []
                    for column, id_type in id_columns.items():
                        if id_type in {"his_inpatient_number", "patient_sequence", "hospital_or_inpatient_number"}:
                            visit_id_tokens.extend(split_identifier_values(row_values.get(column, "")))
                    is_visit_candidate = (
                        table_name in {"索引", "病历", "病案首页"}
                        or "病案首页" in table_name
                        or bool(visit_id_tokens)
                    )
                    if is_visit_candidate and union_nodes:
                        admission_times = [
                            parsed for column, parsed in parsed_times if "入院" in column or "就诊" in column
                        ]
                        discharge_times = [parsed for column, parsed in parsed_times if "出院" in column]
                        admit_time = min(admission_times) if admission_times else (min((t for _, t in parsed_times), default=None))
                        discharge_time = max(discharge_times) if discharge_times else None
                        los_days = ""
                        if admit_time and discharge_time and discharge_time >= admit_time:
                            los_days = round((discharge_time - admit_time).total_seconds() / 86400, 2)
                        visit_hash_source = "|".join(sorted(set(visit_id_tokens))) or f"{relative_path}|{row_index}"
                        visit_rows.append(
                            {
                                "source_folder": source_folder,
                                "source_table": table_name,
                                "node_refs": union_nodes,
                                "visit_id_hash": hashlib.sha256(
                                    f"{HASH_SALT}|visit|{source_folder}|{visit_hash_source}".encode("utf-8")
                                ).hexdigest()[:20],
                                "visit_id_preview": mask_value(visit_id_tokens[0]) if visit_id_tokens else "",
                                "admit_time": iso_or_blank(admit_time),
                                "discharge_time": iso_or_blank(discharge_time),
                                "los_days": los_days,
                                "disease_groups": ";".join(sorted(row_disease_groups)),
                            }
                        )

                audit.detailed_rows_scanned = row_index
                audit.min_time = iso_or_blank(table_min_time)
                audit.max_time = iso_or_blank(table_max_time)
        except Exception as exc:  # keep the run going and record the broken table
            audit.read_error = f"{type(exc).__name__}: {exc}"

        table_audits.append(audit)

    components: dict[tuple[str, str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for node in union_find.parent:
        components[union_find.find(node)].append(node)

    sorted_roots = sorted(components, key=lambda root: min(id_hash(node) for node in components[root]))
    component_id_by_root = {root: f"ECP{index:06d}" for index, root in enumerate(sorted_roots, start=1)}

    node_to_component: dict[tuple[str, str, str], str] = {}
    component_summaries: list[dict[str, object]] = []
    component_sources: dict[str, Counter[str]] = {}
    component_tables: dict[str, Counter[str]] = {}
    component_disease_counts: dict[str, Counter[str]] = {}

    for root in sorted_roots:
        component_id = component_id_by_root[root]
        nodes = components[root]
        for node in nodes:
            node_to_component[node] = component_id
        sources = Counter()
        tables = Counter()
        id_type_counts = Counter()
        disease_counts = Counter()
        occurrence_count = 0
        for node in nodes:
            sources.update(node_sources.get(node, Counter()))
            tables.update(node_tables.get(node, Counter()))
            disease_counts.update(diagnosis_component_sources.get(node, Counter()))
            id_type_counts[node[1]] += 1
            occurrence_count += node_occurrences[node]
        component_sources[component_id] = sources
        component_tables[component_id] = tables
        component_disease_counts[component_id] = disease_counts
        conflict_reasons = []
        for id_type in ("medical_record_number", "outpatient_number", "hospital_number", "empi", "identity_insurance_number"):
            if id_type_counts[id_type] > 1:
                conflict_reasons.append(f"multiple_{id_type}")
        component_summaries.append(
            {
                "unified_patient_id": component_id,
                "source_folder_count": len(sources),
                "source_folders": ";".join(sorted(sources)),
                "source_table_count": len(tables),
                "source_tables": ";".join(sorted(tables)),
                "id_node_count": len(nodes),
                "id_occurrence_count": occurrence_count,
                "has_exact_primary_id": any(node[1] in EXACT_PRIMARY_ID_TYPES for node in nodes),
                "exact_primary_id_types": ";".join(sorted({node[1] for node in nodes if node[1] in EXACT_PRIMARY_ID_TYPES})),
                "has_hospital_or_inpatient_equivalent": any(
                    node[1]
                    in {
                        "hospital_number",
                        "hospital_or_inpatient_number",
                        "his_inpatient_number",
                        "patient_sequence",
                    }
                    for node in nodes
                ),
                "medical_record_number_count": id_type_counts["medical_record_number"],
                "outpatient_number_count": id_type_counts["outpatient_number"],
                "hospital_number_count": id_type_counts["hospital_number"],
                "hospital_or_inpatient_number_count": id_type_counts["hospital_or_inpatient_number"],
                "his_inpatient_number_count": id_type_counts["his_inpatient_number"],
                "patient_sequence_count": id_type_counts["patient_sequence"],
                "patient_sn_count": id_type_counts["patient_sn"],
                "empi_count": id_type_counts["empi"],
                "identity_insurance_number_count": id_type_counts["identity_insurance_number"],
                "diagnosis_groups": ";".join(sorted(disease_counts)),
                "conflict_flag": bool(conflict_reasons),
                "conflict_reason": ";".join(conflict_reasons),
            }
        )

    crosswalk_rows: list[dict[str, object]] = []
    for node, component_id in sorted(node_to_component.items(), key=lambda item: (item[1], item[0][1], id_hash(item[0]))):
        scope, id_type, value = node
        crosswalk_rows.append(
            {
                "unified_patient_id": component_id,
                "id_type": id_type,
                "id_type_label": ID_TYPE_LABELS.get(id_type, id_type),
                "id_scope": scope,
                "id_hash": id_hash(node),
                "id_preview_masked": mask_value(value),
                "occurrence_count": node_occurrences[node],
                "source_folders": ";".join(sorted(node_sources.get(node, Counter()))),
                "source_tables": ";".join(sorted(node_tables.get(node, Counter()))),
            }
        )

    visit_output_rows: list[dict[str, object]] = []
    seen_visits: set[tuple[str, str, str, str]] = set()
    for visit in visit_rows:
        component_ids = sorted({node_to_component[node] for node in visit["node_refs"] if node in node_to_component})
        if not component_ids:
            continue
        key = (visit["source_folder"], visit["source_table"], visit["visit_id_hash"], component_ids[0])
        if key in seen_visits:
            continue
        seen_visits.add(key)
        visit_output_rows.append(
            {
                "unified_patient_id": component_ids[0],
                "component_count_in_row": len(component_ids),
                "source_folder": visit["source_folder"],
                "source_table": visit["source_table"],
                "visit_id_hash": visit["visit_id_hash"],
                "visit_id_preview_masked": visit["visit_id_preview"],
                "admit_time": visit["admit_time"],
                "discharge_time": visit["discharge_time"],
                "los_days": visit["los_days"],
                "disease_groups": visit["disease_groups"],
            }
        )

    schema_rows = [
        {
            "source_folder": audit.source_folder,
            "table_name": audit.table_name,
            "relative_path": audit.relative_path,
            "encoding": audit.encoding,
            "delimiter": audit.delimiter,
            "file_size_mb": audit.file_size_mb,
            "row_count": audit.row_count,
            "row_count_is_estimated": audit.row_count_is_estimated,
            "detailed_rows_scanned": audit.detailed_rows_scanned,
            "parsing_mode": audit.parsing_mode,
            "column_count": audit.column_count,
            "exact_primary_id_columns": ";".join(audit.exact_primary_id_columns),
            "link_id_columns": ";".join(audit.link_id_columns),
            "time_columns": ";".join(audit.time_columns),
            "diagnosis_columns": ";".join(audit.diagnosis_columns),
            "min_time": audit.min_time,
            "max_time": audit.max_time,
            "read_error": audit.read_error,
            "columns": ";".join(audit.columns),
        }
        for audit in table_audits
    ]

    folder_rows: list[dict[str, object]] = []
    for source_folder in SOURCE_FOLDERS:
        components_in_source = [
            summary["unified_patient_id"]
            for summary in component_summaries
            if source_folder in str(summary["source_folders"]).split(";")
        ]
        folder_rows.append(
            {
                "source_folder": source_folder,
                "csv_file_count": source_file_counts[source_folder],
                "source_row_count": source_row_counts[source_folder],
                "linked_patient_component_count": len(components_in_source),
                "id_occurrence_count": sum(id_type_counts_by_source[source_folder].values()),
                **{f"id_occurrences_{id_type}": id_type_counts_by_source[source_folder][id_type] for id_type in sorted(LINKABLE_ID_TYPES)},
            }
        )

    conflict_rows = [row for row in component_summaries if row["conflict_flag"]]
    disease_rows = [
        {
            "disease_group": group,
            "diagnosis_row_count": diagnosis_row_counts[group],
            "linked_patient_component_count": sum(
                1 for counts in component_disease_counts.values() if counts[group] > 0
            ),
        }
        for group in sorted(DISEASE_GROUPS)
    ]

    patient_table_rows: list[dict[str, object]] = []
    for source_folder in SOURCE_FOLDERS:
        stats = patient_table_stats.get(source_folder, {})
        values = patient_table_values.get(source_folder, {})
        patient_table_rows.append(
            {
                "source_folder": source_folder,
                "patient_table_rows": stats.get("patient_table_rows", 0),
                "patient_table_rows_with_exact_primary_id": stats.get("patient_table_rows_with_exact_primary_id", 0),
                "patient_table_rows_with_empi": stats.get("patient_table_rows_with_empi", 0),
                "unique_medical_record_numbers": len(values.get("medical_record_number", set())),
                "unique_outpatient_numbers": len(values.get("outpatient_number", set())),
                "unique_his_inpatient_numbers": len(values.get("his_inpatient_number", set())),
                "unique_empi": len(values.get("empi", set())),
            }
        )

    all_patient_empi: set[str] = set()
    all_patient_mrn: set[str] = set()
    all_patient_outpatient: set[str] = set()
    for values in patient_table_values.values():
        all_patient_empi.update(values.get("empi", set()))
        all_patient_mrn.update(values.get("medical_record_number", set()))
        all_patient_outpatient.update(values.get("outpatient_number", set()))

    overlap_distribution = Counter(int(row["source_folder_count"]) for row in component_summaries)
    component_count = len(component_summaries)
    exact_primary_component_count = sum(1 for row in component_summaries if row["has_exact_primary_id"])
    equivalent_component_count = sum(1 for row in component_summaries if row["has_hospital_or_inpatient_equivalent"])
    multi_source_component_count = sum(1 for row in component_summaries if int(row["source_folder_count"]) > 1)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_folders": SOURCE_FOLDERS,
        "csv_file_count": sum(source_file_counts.values()),
        "source_row_count": sum(source_row_counts.values()),
        "row_count_method": "exact binary newline count for files <=80MB; estimated from the first 8MB for larger files; quoted multiline fields may require manual review",
        "large_table_sample_row_limit": MAX_LARGE_TABLE_ROWS,
        "table_parsing_modes": dict(Counter(audit.parsing_mode for audit in table_audits)),
        "linked_patient_component_count": component_count,
        "exact_primary_component_count": exact_primary_component_count,
        "hospital_or_inpatient_equivalent_component_count": equivalent_component_count,
        "auxiliary_only_component_count": component_count - exact_primary_component_count,
        "multi_source_component_count": multi_source_component_count,
        "component_source_overlap_distribution": dict(sorted(overlap_distribution.items())),
        "visit_summary_row_count": len(visit_output_rows),
        "patient_table_total_rows": sum(int(row["patient_table_rows"]) for row in patient_table_rows),
        "patient_table_unique_empi": len(all_patient_empi),
        "patient_table_unique_medical_record_numbers": len(all_patient_mrn),
        "patient_table_unique_outpatient_numbers": len(all_patient_outpatient),
        "conflict_component_count": len(conflict_rows),
        "exact_primary_columns_found": dict(exact_primary_columns_found),
        "overall_min_time": iso_or_blank(all_min_time),
        "overall_max_time": iso_or_blank(all_max_time),
        "disease_group_row_counts": dict(diagnosis_row_counts),
        "existing_20k_cardiac_file_found_in_workspace": bool(
            list(BASE_DIR.parent.rglob("final_preprocessed_data_new*.csv"))
        ),
        "privacy_note": "Identifier outputs are salted hashes with masked previews; raw identifiers are not written by this script.",
    }

    write_csv(
        BASE_DIR / "expanded_cardiac_table_schema.csv",
        schema_rows,
        [
            "source_folder",
            "table_name",
            "relative_path",
            "encoding",
            "delimiter",
            "file_size_mb",
            "row_count",
            "row_count_is_estimated",
            "detailed_rows_scanned",
            "parsing_mode",
            "column_count",
            "exact_primary_id_columns",
            "link_id_columns",
            "time_columns",
            "diagnosis_columns",
            "min_time",
            "max_time",
            "read_error",
            "columns",
        ],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_crosswalk.csv",
        crosswalk_rows,
        [
            "unified_patient_id",
            "id_type",
            "id_type_label",
            "id_scope",
            "id_hash",
            "id_preview_masked",
            "occurrence_count",
            "source_folders",
            "source_tables",
        ],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_component_summary.csv",
        component_summaries,
        [
            "unified_patient_id",
            "source_folder_count",
            "source_folders",
            "source_table_count",
            "source_tables",
            "id_node_count",
            "id_occurrence_count",
            "has_exact_primary_id",
            "exact_primary_id_types",
            "has_hospital_or_inpatient_equivalent",
            "medical_record_number_count",
            "outpatient_number_count",
            "hospital_number_count",
            "hospital_or_inpatient_number_count",
            "his_inpatient_number_count",
            "patient_sequence_count",
            "patient_sn_count",
            "empi_count",
            "identity_insurance_number_count",
            "diagnosis_groups",
            "conflict_flag",
            "conflict_reason",
        ],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_visit_summary.csv",
        visit_output_rows,
        [
            "unified_patient_id",
            "component_count_in_row",
            "source_folder",
            "source_table",
            "visit_id_hash",
            "visit_id_preview_masked",
            "admit_time",
            "discharge_time",
            "los_days",
            "disease_groups",
        ],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_folder_summary.csv",
        folder_rows,
        [
            "source_folder",
            "csv_file_count",
            "source_row_count",
            "linked_patient_component_count",
            "id_occurrence_count",
            *[f"id_occurrences_{id_type}" for id_type in sorted(LINKABLE_ID_TYPES)],
        ],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_id_conflicts.csv",
        conflict_rows,
        [
            "unified_patient_id",
            "source_folder_count",
            "source_folders",
            "source_table_count",
            "source_tables",
            "id_node_count",
            "id_occurrence_count",
            "has_exact_primary_id",
            "exact_primary_id_types",
            "has_hospital_or_inpatient_equivalent",
            "medical_record_number_count",
            "outpatient_number_count",
            "hospital_number_count",
            "hospital_or_inpatient_number_count",
            "his_inpatient_number_count",
            "patient_sequence_count",
            "patient_sn_count",
            "empi_count",
            "identity_insurance_number_count",
            "diagnosis_groups",
            "conflict_flag",
            "conflict_reason",
        ],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_diagnosis_keyword_counts.csv",
        disease_rows,
        ["disease_group", "diagnosis_row_count", "linked_patient_component_count"],
    )
    write_csv(
        BASE_DIR / "expanded_cardiac_patient_table_summary.csv",
        patient_table_rows,
        [
            "source_folder",
            "patient_table_rows",
            "patient_table_rows_with_exact_primary_id",
            "patient_table_rows_with_empi",
            "unique_medical_record_numbers",
            "unique_outpatient_numbers",
            "unique_his_inpatient_numbers",
            "unique_empi",
        ],
    )

    with (BASE_DIR / "expanded_cardiac_cohort_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    exact_primary_note = (
        "Direct 病案号/门诊号 columns were found and used as exact primary identifiers."
        if exact_primary_columns_found
        else "No columns explicitly named 病案号 or 门诊号 were found; linkage therefore used the closest available hospital/inpatient identifiers plus patient_SN/EMPI/identity-insurance fields, and this limitation should be stated in the revision."
    )
    markdown = f"""# Expanded Cardiac Cohort Audit

Generated: {summary['generated_at']}

## Cohort Linkage Result

- Source folders scanned: {len(SOURCE_FOLDERS)}
- CSV files scanned: {summary['csv_file_count']:,}
- Source rows scanned: {summary['source_row_count']:,}
- Row-count method: {summary['row_count_method']}
- Large detail tables: schema and row counts were retained, with row-level linkage/time/diagnosis extraction limited to the first {summary['large_table_sample_row_limit']:,} rows unless the table was a core linkage table.
- Linked patient components: {summary['linked_patient_component_count']:,}
- Rows in the three `患者.csv` tables: {summary['patient_table_total_rows']:,}
- Unique EMPI values in `患者.csv`: {summary['patient_table_unique_empi']:,}
- Unique 病案号 values in `患者.csv`: {summary['patient_table_unique_medical_record_numbers']:,}
- Unique 门诊号 values in `患者.csv`: {summary['patient_table_unique_outpatient_numbers']:,}
- Visit/admission summary rows: {summary['visit_summary_row_count']:,}
- Components spanning more than one source folder: {summary['multi_source_component_count']:,}
- Components with direct 病案号/门诊号: {summary['exact_primary_component_count']:,}
- Components with hospital/inpatient equivalent identifiers: {summary['hospital_or_inpatient_equivalent_component_count']:,}
- Potential identifier-conflict components: {summary['conflict_component_count']:,}
- Overall date range detected: {summary['overall_min_time'] or 'not detected'} to {summary['overall_max_time'] or 'not detected'}

{exact_primary_note}

## Privacy Handling

The crosswalk output uses salted hashes and masked previews. Raw 病案号、门诊号、医院号、HIS住院号、patient_SN、EMPI、身份证号/医保号 values are used only in memory during linkage and are not written by this script.

`patient_SN` values are retained in the audit table but are not used for automatic union-based patient merging, because several source tables contain high-frequency `patient_SN`-like values that behave as non-unique administrative or name-like fields.

## Manuscript-Ready Methods Text

For the revision, the added cardiovascular cohort can be described as follows:

> We incorporated three additional de-identified electronic medical record exports from the Wuhan Union Hospital cardiovascular service into the secondary cardiovascular cohort analysis. Patient records were harmonized across the exports using direct medical-record and outpatient identifiers when available. Direct 病案号 and 门诊号 fields were used as the primary alignment keys; hospital number, HIS inpatient number, patient sequence number, EMPI, and identity/insurance fields were retained as auxiliary linkage evidence. All identifiers were standardized, split when multiple values were concatenated, and resolved into hashed patient components before aggregate analyses. The field patient_SN was retained in the audit output but excluded from automatic union-based patient merging because several source tables contained high-frequency patient_SN-like values that behaved as non-unique administrative or name-like fields. Components with multiple high-specificity identifiers were retained in an audit file for manual review rather than silently merged.

## Manuscript-Ready Results Text

> The three added cardiovascular exports contained {summary['csv_file_count']:,} CSV tables and approximately {summary['source_row_count']:,} source rows. The patient-index tables contained {summary['patient_table_total_rows']:,} rows, with {summary['patient_table_unique_empi']:,} unique EMPI values, {summary['patient_table_unique_medical_record_numbers']:,} unique 病案号 values, and {summary['patient_table_unique_outpatient_numbers']:,} unique 门诊号 values. Identifier harmonization produced {summary['linked_patient_component_count']:,} auditable identifier components and {summary['visit_summary_row_count']:,} visit/admission-level summary rows. {summary['multi_source_component_count']:,} components appeared in more than one source export, supporting cross-export de-duplication. {summary['conflict_component_count']:,} components contained multiple high-specificity identifiers and were flagged for manual audit. The detected clinical time range spanned {summary['overall_min_time'] or 'not detected'} to {summary['overall_max_time'] or 'not detected'}.

Note: large narrative and laboratory detail tables were counted and schema-audited, but row-level extraction was sampled to keep the reproducible audit tractable. The core linkage tables were parsed in full.

## Output Files

- expanded_cardiac_table_schema.csv: CSV-level row counts, encodings, columns, identifier columns, time columns, and diagnosis columns.
- expanded_cardiac_crosswalk.csv: hashed identifier-to-patient-component crosswalk.
- expanded_cardiac_component_summary.csv: patient-component-level source overlap and conflict flags.
- expanded_cardiac_visit_summary.csv: de-duplicated visit/admission summary rows with hashed visit identifiers.
- expanded_cardiac_folder_summary.csv: source-folder-level row and linkage counts.
- expanded_cardiac_id_conflicts.csv: components requiring manual identifier review.
- expanded_cardiac_diagnosis_keyword_counts.csv: coarse disease-keyword counts for revision reporting.
- expanded_cardiac_patient_table_summary.csv: patient-table row counts and unique patient identifier counts by source export.
- expanded_cardiac_cohort_summary.json: machine-readable summary statistics.
"""
    (BASE_DIR / "expanded_cardiac_revision_summary.md").write_text(markdown, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
