from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import build_expanded_cardiac_cohort as audit


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_CARDIAC_CSV = Path(
    BASE_DIR.parent / "data" / "private" / "original_cardiac" / "final_preprocessed_data_new.csv"
)
OUTPUT_PREFIX = "cardiac_original_expanded_dedup"
HASH_SALT = "NC_revision_original_expanded_dedup_2026_05"


def hash_identifier(id_type: str, value: str) -> str:
    payload = f"{HASH_SALT}|{id_type}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def open_csv(path: Path) -> tuple[str, str, list[str], csv.reader, object]:
    encoding, delimiter, _ = audit.decode_sample(path)
    handle = path.open("r", encoding=encoding, errors="replace", newline="")
    reader = csv.reader(handle, delimiter=delimiter)
    try:
        header = audit.clean_header(next(reader))
    except StopIteration:
        header = []
    return encoding, delimiter, header, reader, handle


def collect_original_mrn() -> tuple[Counter[str], dict[str, object]]:
    encoding, delimiter, header, reader, handle = open_csv(ORIGINAL_CARDIAC_CSV)
    try:
        if "病案号" not in header:
            raise ValueError("Original cardiac CSV does not contain a 病案号 column")
        mrn_index = header.index("病案号")
        row_count = 0
        missing_count = 0
        values: Counter[str] = Counter()
        for row in reader:
            row_count += 1
            value = audit.normalize_value(row[mrn_index] if mrn_index < len(row) else "")
            tokens = audit.split_identifier_values(value)
            if not tokens:
                missing_count += 1
                continue
            for token in tokens:
                values[token] += 1
        meta = {
            "path": str(ORIGINAL_CARDIAC_CSV),
            "encoding": encoding,
            "delimiter": "TAB" if delimiter == "\t" else delimiter,
            "row_count": row_count,
            "missing_mrn_rows": missing_count,
            "unique_mrn": len(values),
            "column_count": len(header),
        }
        return values, meta
    finally:
        handle.close()


def collect_expanded_mrn() -> tuple[Counter[str], dict[str, Counter[str]], list[dict[str, object]]]:
    values: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    table_rows: list[dict[str, object]] = []
    target_tables = {
        "患者",
        "索引",
        "病案首页",
        "病案首页出院诊断",
        "病案首页手术记录",
        "病案首页病理诊断",
    }

    for source_folder in audit.SOURCE_FOLDERS:
        source_root = BASE_DIR / source_folder
        for path in sorted(source_root.rglob("*.csv")):
            table_name = path.stem
            if table_name not in target_tables:
                continue
            encoding, delimiter, header, reader, handle = open_csv(path)
            try:
                mrn_columns = [
                    column
                    for column in header
                    if audit.classify_identifier_column(column) == "medical_record_number"
                ]
                row_count = 0
                rows_with_mrn = 0
                table_values: Counter[str] = Counter()
                if mrn_columns:
                    mrn_indexes = [header.index(column) for column in mrn_columns]
                    for row in reader:
                        row_count += 1
                        row_tokens: set[str] = set()
                        for index in mrn_indexes:
                            raw = row[index] if index < len(row) else ""
                            row_tokens.update(audit.split_identifier_values(raw))
                        if row_tokens:
                            rows_with_mrn += 1
                        for token in row_tokens:
                            values[token] += 1
                            by_source[source_folder][token] += 1
                            table_values[token] += 1
                else:
                    for _ in reader:
                        row_count += 1
                table_rows.append(
                    {
                        "source_folder": source_folder,
                        "table_name": table_name,
                        "relative_path": path.relative_to(BASE_DIR).as_posix(),
                        "encoding": encoding,
                        "row_count": row_count,
                        "mrn_columns": ";".join(mrn_columns),
                        "rows_with_mrn": rows_with_mrn,
                        "unique_mrn": len(table_values),
                    }
                )
            finally:
                handle.close()
    return values, by_source, table_rows


def main() -> int:
    if not ORIGINAL_CARDIAC_CSV.exists():
        raise FileNotFoundError(ORIGINAL_CARDIAC_CSV)

    original_mrn, original_meta = collect_original_mrn()
    expanded_mrn, expanded_by_source, expanded_table_rows = collect_expanded_mrn()

    original_set = set(original_mrn)
    expanded_set = set(expanded_mrn)
    overlap = original_set & expanded_set
    original_only = original_set - expanded_set
    expanded_only = expanded_set - original_set
    combined = original_set | expanded_set

    overlap_rows = []
    for value in sorted(overlap, key=lambda item: hash_identifier("medical_record_number", item)):
        source_folders = [source for source, counter in expanded_by_source.items() if value in counter]
        overlap_rows.append(
            {
                "mrn_hash": hash_identifier("medical_record_number", value),
                "original_row_occurrences": original_mrn[value],
                "expanded_occurrences": expanded_mrn[value],
                "expanded_source_folders": ";".join(source_folders),
                "expanded_source_folder_count": len(source_folders),
            }
        )

    source_rows = []
    for source_folder in audit.SOURCE_FOLDERS:
        source_set = set(expanded_by_source[source_folder])
        source_rows.append(
            {
                "source_folder": source_folder,
                "expanded_unique_mrn": len(source_set),
                "overlap_with_original_mrn": len(source_set & original_set),
                "new_only_mrn_vs_original": len(source_set - original_set),
                "overlap_pct_of_source": round(100 * len(source_set & original_set) / len(source_set), 2)
                if source_set
                else 0,
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "original_cardiac_csv": original_meta,
        "expanded_tables_scanned_for_dedup": len(expanded_table_rows),
        "expanded_unique_mrn_in_core_tables": len(expanded_set),
        "overlap_unique_mrn": len(overlap),
        "original_only_unique_mrn": len(original_only),
        "expanded_only_unique_mrn": len(expanded_only),
        "combined_unique_mrn_after_dedup": len(combined),
        "incremental_unique_mrn_added_by_expanded_exports": len(expanded_only),
        "incremental_unique_mrn_pct_vs_original": round(100 * len(expanded_only) / len(original_set), 2)
        if original_set
        else 0,
        "overlap_pct_of_original": round(100 * len(overlap) / len(original_set), 2) if original_set else 0,
        "overlap_pct_of_expanded": round(100 * len(overlap) / len(expanded_set), 2) if expanded_set else 0,
        "privacy_note": "Raw 病案号 values were used in memory only. Outputs contain aggregate counts and salted hashes only.",
    }

    validation_wide_note = ""
    wide_summary_path = BASE_DIR / "expanded_cardiac_wide_table_summary.json"
    if wide_summary_path.exists():
        with wide_summary_path.open("r", encoding="utf-8") as handle:
            wide_summary = json.load(handle)
        expanded_wide = wide_summary.get("expanded", {})
        validation_wide_note = (
            f"\nThe {summary['combined_unique_mrn_after_dedup']:,} count is the old-versus-expanded linkage-audit denominator across core 病案号 tables. "
            f"The analysis-ready expanded validation wide table contains {expanded_wide.get('expanded_unique_patients', 0):,} unique 病案号 "
            f"across {expanded_wide.get('expanded_admissions', 0):,} admissions; this is the denominator used in the revised main cardiac validation.\n"
        )

    with (BASE_DIR / f"{OUTPUT_PREFIX}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_csv(
        BASE_DIR / f"{OUTPUT_PREFIX}_overlap_hashes.csv",
        overlap_rows,
        [
            "mrn_hash",
            "original_row_occurrences",
            "expanded_occurrences",
            "expanded_source_folders",
            "expanded_source_folder_count",
        ],
    )
    write_csv(
        BASE_DIR / f"{OUTPUT_PREFIX}_source_overlap.csv",
        source_rows,
        [
            "source_folder",
            "expanded_unique_mrn",
            "overlap_with_original_mrn",
            "new_only_mrn_vs_original",
            "overlap_pct_of_source",
        ],
    )
    write_csv(
        BASE_DIR / f"{OUTPUT_PREFIX}_table_audit.csv",
        expanded_table_rows,
        [
            "source_folder",
            "table_name",
            "relative_path",
            "encoding",
            "row_count",
            "mrn_columns",
            "rows_with_mrn",
            "unique_mrn",
        ],
    )

    markdown = f"""# Original-vs-Expanded Cardiac Deduplication Audit

Generated: {summary['generated_at']}

## Purpose

This audit compares the previously used `final_preprocessed_data_new.csv` cardiac cohort with the three expanded cardiac exports under `NC_revision`, using 病案号 as the shared deduplication key. The original file exposes 病案号 but not 门诊号, so this overlap analysis is intentionally 病案号-based.

## Key Results

- Original cardiac rows: {original_meta['row_count']:,}
- Original unique 病案号: {original_meta['unique_mrn']:,}
- Expanded unique 病案号 in core linkage tables: {summary['expanded_unique_mrn_in_core_tables']:,}
- Overlapping unique 病案号: {summary['overlap_unique_mrn']:,}
- Original-only unique 病案号: {summary['original_only_unique_mrn']:,}
- Expanded-only unique 病案号: {summary['expanded_only_unique_mrn']:,}
- Combined unique 病案号 after deduplication: {summary['combined_unique_mrn_after_dedup']:,}
- Incremental unique 病案号 added by expanded exports: {summary['incremental_unique_mrn_added_by_expanded_exports']:,} ({summary['incremental_unique_mrn_pct_vs_original']}% of the original unique 病案号 count)
- Overlap as percentage of original unique 病案号: {summary['overlap_pct_of_original']}%
- Overlap as percentage of expanded unique 病案号: {summary['overlap_pct_of_expanded']}%

## Interpretation for Revision

The newly located original cardiac file removes the previous uncertainty about whether overlap with the existing 20K cardiac cohort can be audited. Using 病案号, the added cardiac exports contribute {summary['incremental_unique_mrn_added_by_expanded_exports']:,} additional unique 病案号 beyond the original file, increasing the deduplicated 病案号 denominator from {original_meta['unique_mrn']:,} to {summary['combined_unique_mrn_after_dedup']:,}. This directly strengthens the secondary cardiac cohort by documenting both overlap and incremental volume.

This analysis still should not be described as fully independent external validation, because the original and expanded cardiac data remain within the same health-system environment. The appropriate wording is expanded same-health-system supportive replication with explicit 病案号-based de-duplication.
{validation_wide_note}

## Privacy Handling

Raw 病案号 values were used only in memory. The overlap file contains salted hashes and occurrence counts, not raw identifiers.
"""
    (BASE_DIR / f"{OUTPUT_PREFIX}_summary.md").write_text(markdown, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
