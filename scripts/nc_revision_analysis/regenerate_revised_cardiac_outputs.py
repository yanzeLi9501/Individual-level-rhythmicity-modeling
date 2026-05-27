from __future__ import annotations

import csv
import importlib.util
import json
import re
import runpy
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"data/readmission_output/figures_v2/merge/Individual-level-rhythmicity-modeling")
HEALTHLINE_ROOT = Path(r"data")
READMISSION_ROOT = Path(r"data/readmission_output")

ORIGINAL_CONFIG = PROJECT_ROOT / "src" / "figures" / "fig_config.py"
ORIGINAL_PROSPECTIVE_SCRIPT = PROJECT_ROOT / "src" / "supplementary" / "gen_supp_prospective_validation.py"
ORIGINAL_FIG4_SCRIPT = PROJECT_ROOT / "src" / "figures" / "gen_fig_new4.py"

EXPANDED_SUMMARY = BASE_DIR / "expanded_cardiac_cohort_summary.json"
DEDUP_SUMMARY = BASE_DIR / "cardiac_original_expanded_dedup_summary.json"
SOURCE_OVERLAP = BASE_DIR / "cardiac_original_expanded_dedup_source_overlap.csv"
DISEASE_COUNTS = BASE_DIR / "expanded_cardiac_diagnosis_keyword_counts.csv"
FOLDER_SUMMARY = BASE_DIR / "expanded_cardiac_folder_summary.csv"
EXTERNAL_POSITIVE_CONTROL_SUMMARY = BASE_DIR / "external_positive_control_results" / "external_positive_control_summary.json"
MIMIC_EXTERNAL_CORRELATIONS = BASE_DIR / "external_positive_control_results" / "mimic_influenza_profile_correlations.csv"
NWICU_EXTERNAL_CORRELATIONS = BASE_DIR / "external_positive_control_results" / "nwicu_covid_profile_correlations.csv"
EICU_EXTERNAL_CORRELATIONS = BASE_DIR / "external_positive_control_results" / "eicu_viral_pneumonia_profile_correlations.csv"
EXTERNAL_COVID_DATASET_AUDIT = BASE_DIR / "external_positive_control_results" / "external_covid_dataset_audit.csv"
WHO_FLUNET_ANNUAL = BASE_DIR / "external_positive_control_results" / "who_flunet_annual_china_vs_rest_world.csv"
LGDI_WHU_INFLUENZA_SUMMARY = BASE_DIR / "lgdi_results" / "lgdi_whu_influenza_summary.json"
LGDI_CARDIAC42K_SUMMARY = BASE_DIR / "lgdi_results" / "lgdi_cardiac42k_influenza_summary.json"
LGDI_WHU_LAG_BOOTSTRAP = BASE_DIR / "lgdi_results" / "lgdi_whu_lag_spearman_bootstrap.csv"
LGDI_CARDIAC42K_LAG = BASE_DIR / "lgdi_results" / "lgdi_cardiac42k_influenza_lag.csv"
EXPANDED_RAW_VALIDATION_RESULTS = BASE_DIR / "prospective_validation_results_expanded_raw.json"
EXPANDED_RAW_COMPARISON = BASE_DIR / "expanded_raw_validation_comparison.md"
WHU_PRIMARY_ADMISSIONS_TABLE = READMISSION_ROOT / "all_admissions.csv"
EXPANDED_VALIDATION_WIDE_TABLE = BASE_DIR / "expanded_cardiac_wide_table.csv"
VALIDATION_WIDE_MRN_OVERLAP_JSON = BASE_DIR / "whu_primary_expanded_mrn_overlap_summary.json"
VALIDATION_WIDE_MRN_OVERLAP_MD = BASE_DIR / "whu_primary_expanded_mrn_overlap_summary.md"

REVISED_RESULTS = BASE_DIR / "revised_cardiac_validation_results.json"
RECOMPUTED_SENTINEL_RESULTS = BASE_DIR / "prospective_validation_results_recomputed.json"
REPORT_PATH = BASE_DIR / "revised_cardiac_analysis_change_report.md"
MANIFEST_PATH = BASE_DIR / "revised_cardiac_regeneration_manifest.json"


def load_original_config() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("original_fig_config", ORIGINAL_CONFIG)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {ORIGINAL_CONFIG}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT_DIR = str(BASE_DIR)
    module.FIG_DIR = str(BASE_DIR)
    Path(module.FIG_DIR).mkdir(parents=True, exist_ok=True)
    return module


def install_revision_fig_config() -> types.ModuleType:
    original = load_original_config()
    shim = types.ModuleType("fig_config")
    for name, value in vars(original).items():
        if not name.startswith("__"):
            setattr(shim, name, value)
    shim.OUTPUT_DIR = str(BASE_DIR)
    shim.FIG_DIR = str(BASE_DIR)
    sys.modules["fig_config"] = shim
    return shim


def copy_original_named_outputs() -> None:
    return None


def run_original_validation_and_figures() -> None:
    install_revision_fig_config()
    if not RECOMPUTED_SENTINEL_RESULTS.exists():
        runpy.run_path(str(ORIGINAL_PROSPECTIVE_SCRIPT), run_name="__main__")
        shutil.copyfile(BASE_DIR / "prospective_validation_results.json", RECOMPUTED_SENTINEL_RESULTS)
    else:
        shutil.copyfile(RECOMPUTED_SENTINEL_RESULTS, BASE_DIR / "prospective_validation_results.json")
    pv_dir = BASE_DIR / "prospective_validation"
    pv_dir.mkdir(exist_ok=True)
    shutil.copyfile(RECOMPUTED_SENTINEL_RESULTS, pv_dir / "prospective_validation_results.json")
    copy_original_named_outputs()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


MISSING_ID_VALUES = {
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


def normalize_mrn(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in MISSING_ID_VALUES:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def read_mrn_set(path: Path) -> tuple[set[str], int, int]:
    values: set[str] = set()
    rows = 0
    missing = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "病案号" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain a 病案号 column")
        for row in reader:
            rows += 1
            value = normalize_mrn(row.get("病案号"))
            if value:
                values.add(value)
            else:
                missing += 1
    return values, rows, missing


def build_validation_wide_mrn_overlap() -> dict[str, object]:
    original_mrn, original_rows, original_missing = read_mrn_set(WHU_PRIMARY_ADMISSIONS_TABLE)
    expanded_mrn, expanded_rows, expanded_missing = read_mrn_set(EXPANDED_VALIDATION_WIDE_TABLE)
    overlap = original_mrn & expanded_mrn
    original_only = original_mrn - expanded_mrn
    expanded_only = expanded_mrn - original_mrn
    union = original_mrn | expanded_mrn
    summary = {
        "scope": "WHU primary original cohort and expanded validation wide table by 病案号",
        "original_label": "WHU primary original cohort",
        "expanded_label": "expanded cardiac validation wide table",
        "original_path": str(WHU_PRIMARY_ADMISSIONS_TABLE),
        "expanded_path": str(EXPANDED_VALIDATION_WIDE_TABLE),
        "original_admissions": original_rows,
        "expanded_admissions": expanded_rows,
        "original_unique_mrn": len(original_mrn),
        "expanded_unique_mrn": len(expanded_mrn),
        "overlap_unique_mrn": len(overlap),
        "original_only_unique_mrn": len(original_only),
        "expanded_only_unique_mrn": len(expanded_only),
        "union_unique_mrn": len(union),
        "original_missing_mrn_rows": original_missing,
        "expanded_missing_mrn_rows": expanded_missing,
        "overlap_pct_of_original": round(100 * len(overlap) / len(original_mrn), 4) if original_mrn else 0.0,
        "overlap_pct_of_expanded": round(100 * len(overlap) / len(expanded_mrn), 4) if expanded_mrn else 0.0,
    }
    VALIDATION_WIDE_MRN_OVERLAP_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# WHU Primary-vs-Expanded MRN Overlap Audit",
        "",
        "This audit compares the WHU primary original cohort and the analysis-ready expanded cardiac validation wide table by normalized 病案号. It does not use the WHU/original cardiac wide table.",
        "",
        "| Scope | Count |",
        "| --- | ---: |",
        f"| WHU primary original cohort admissions | {summary['original_admissions']:,} |",
        f"| WHU primary original cohort unique 病案号 | {summary['original_unique_mrn']:,} |",
        f"| Expanded validation wide table unique 病案号 | {summary['expanded_unique_mrn']:,} |",
        f"| Overlapping unique 病案号 | {summary['overlap_unique_mrn']:,} |",
        f"| WHU-primary-only unique 病案号 | {summary['original_only_unique_mrn']:,} |",
        f"| Expanded-only unique 病案号 | {summary['expanded_only_unique_mrn']:,} |",
        f"| Union unique 病案号 | {summary['union_unique_mrn']:,} |",
        "",
        f"The overlap is {summary['overlap_unique_mrn']:,}/{summary['original_unique_mrn']:,} ({summary['overlap_pct_of_original']:.3f}%) of the WHU primary original cohort and {summary['overlap_unique_mrn']:,}/{summary['expanded_unique_mrn']:,} ({summary['overlap_pct_of_expanded']:.3f}%) of the expanded validation wide table.",
    ]
    VALIDATION_WIDE_MRN_OVERLAP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def pct(numerator: float, denominator: float) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def save_revision_figure(fig: plt.Figure, stem: str) -> None:
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(BASE_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(BASE_DIR / f"{stem}.tif", dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left", color="black")


def draw_mrn_overlap_venn(ax: plt.Axes, overlap_summary: dict[str, object], colors: dict[str, str]) -> None:
    original_total = int(overlap_summary.get("original_unique_mrn", 0))
    expanded_total = int(overlap_summary.get("expanded_unique_mrn", 0))
    overlap = int(overlap_summary.get("overlap_unique_mrn", 0))
    original_only = int(overlap_summary.get("original_only_unique_mrn", 0))
    expanded_only = int(overlap_summary.get("expanded_only_unique_mrn", 0))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    expanded_circle = mpatches.Circle((0.60, 0.50), 0.34, facecolor=colors["secondary"], edgecolor=colors["secondary"], alpha=0.28, linewidth=1.5)
    original_circle = mpatches.Circle((0.43, 0.50), 0.30, facecolor=colors["primary"], edgecolor=colors["primary"], alpha=0.38, linewidth=1.5)
    ax.add_patch(expanded_circle)
    ax.add_patch(original_circle)
    ax.text(0.51, 0.50, f"Overlap\n{overlap:,}", ha="center", va="center", fontsize=9.2, fontweight="bold", color=colors["text"])
    ax.text(0.24, 0.50, f"WHU-only\n{original_only:,}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=colors["text"])
    ax.text(0.78, 0.50, f"Expanded-only\n{expanded_only:,}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=colors["text"])
    ax.text(0.29, 0.84, f"WHU primary\n{original_total:,} MRNs", ha="center", va="center", fontsize=7.5, color=colors["text"])
    ax.text(0.73, 0.84, f"Expanded validation\n{expanded_total:,} MRNs", ha="center", va="center", fontsize=7.5, color=colors["text"])
    ax.text(0.50, 0.14, "Normalized MRN overlap", ha="center", va="center", fontsize=7.5, color=colors["text"])
    ax.set_title("Validation-cohort MRN overlap")


def get_colors(config: types.ModuleType) -> tuple[dict[str, str], list[str], object]:
    colors = config.COLORS
    palette = config.PAL
    cmap = config.SPRING_CMAP
    return colors, palette, cmap


def safe_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_revised_main_figure(results: dict, dedup: dict, source_rows: list[dict[str, str]], config: types.ModuleType) -> None:
    colors, _, cmap = get_colors(config)
    monthly = results.get("monthly_timeline", [])
    if not monthly:
        build_revised_main_figure_from_recomputed_results(results, dedup, config)
        return
    wave = [
        row for row in monthly
        if ((int(row.get("year", 0)) == 2022 and int(row.get("month", 0)) >= 5)
            or (int(row.get("year", 0)) == 2023 and int(row.get("month", 0)) <= 9))
    ]
    neg_ctrl = results.get("negative_control_q4_2022", {})
    wave_info = results.get("covid_wave_dec2022", {})
    top_comor = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
    short = {
        "Cardiovascular": "Cardio",
        "Hypertension": "Hypert",
        "Diabetes": "Diabetes",
        "Cerebrovascular": "Cerebro",
        "Renal": "Renal",
        "Respiratory": "Resp",
    }
    comor_colors = {
        "Respiratory": colors["secondary"],
        "Cardiovascular": colors["primary"],
        "Hypertension": colors["accent1"],
        "Diabetes": colors["accent2"],
        "Cerebrovascular": colors["accent3"],
        "Renal": colors["accent5"],
    }

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = gridspec.GridSpec(2, 2, wspace=0.32, hspace=0.48, left=0.08, right=0.97, top=0.90, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    if wave:
        x_idx = np.arange(len(wave))
        labels = [f"{int(r['month']):02d}" if int(r["month"]) != 1 else f"{int(r['year'])}\n{int(r['month']):02d}" for r in wave]
        resp_pct = np.array([safe_float(r.get("resp_pct")) for r in wave])
        bars = ax_a.bar(x_idx, resp_pct, color=colors["secondary"], alpha=0.35, width=0.75, label="Respiratory %")
        for idx, row in enumerate(wave):
            if int(row.get("year", 0)) == 2022 and int(row.get("month", 0)) == 12:
                bars[idx].set_alpha(0.7)
                bars[idx].set_edgecolor(colors["secondary"])
                bars[idx].set_linewidth(1.5)
        ax_a.set_ylabel("Respiratory admission %", color=colors["secondary"])
        ax_a.set_ylim(0, np.nanmax(resp_pct) * 1.25 if np.isfinite(resp_pct).any() else 1)
        baseline_pct = safe_float(wave_info.get("baseline_resp_pct"))
        if np.isfinite(baseline_pct):
            ax_a.axhline(baseline_pct, color=colors["secondary"], linewidth=0.9, linestyle=":", alpha=0.7, label=f"Baseline ({baseline_pct:.1f}%)")
        ax_a2 = ax_a.twinx()
        resp_sim = np.array([safe_float(r.get("resp_sim")) for r in wave])
        nonresp_sim = np.array([safe_float(r.get("nonresp_sim")) for r in wave])
        ax_a2.plot(x_idx, resp_sim, "-o", color=colors["secondary"], linewidth=2.0, markersize=4.5, label="Respiratory sim.")
        ax_a2.plot(x_idx, nonresp_sim, "-s", color=colors["primary"], linewidth=1.4, markersize=3.5, alpha=0.75, label="Non-respiratory sim.")
        ax_a2.set_ylabel("Similarity to COVID+ profile")
        ax_a2.set_ylim(-0.5, 1.05)
        ax_a.set_xticks(x_idx)
        ax_a.set_xticklabels(labels, fontsize=7)
        ax_a.set_xlabel("Month (2022-2023)")
        for idx, row in enumerate(wave):
            if int(row.get("year", 0)) == 2022 and int(row.get("month", 0)) == 12:
                ax_a2.annotate(
                    f"Dec 2022\nResp%={resp_pct[idx]:.1f}%\nSim={resp_sim[idx]:.3f}",
                    xy=(idx, resp_sim[idx]), xytext=(idx + 2.0, min(resp_sim[idx] + 0.18, 0.95)),
                    fontsize=7.5, fontweight="bold", color=colors["secondary"],
                    arrowprops={"arrowstyle": "->", "color": colors["secondary"], "lw": 1.2},
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": colors["secondary"], "alpha": 0.95},
                )
        lines1, labels1 = ax_a.get_legend_handles_labels()
        lines2, labels2 = ax_a2.get_legend_handles_labels()
        ax_a.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", framealpha=0.9)
    ax_a.set_title("Original 20K cohort: post-lockdown COVID wave")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    ref_names = ["COVID+ (2022-12)", "Heart Disease", "Diabetes", "General Pop."]
    ref_short = ["COVID+", "Heart dis.", "Diabetes", "General"]
    matrix = np.full((len(ref_names), len(top_comor)), np.nan)
    for i, ref in enumerate(ref_names):
        for j, comor in enumerate(top_comor):
            matrix[i, j] = safe_float(neg_ctrl.get(ref, {}).get(comor, {}).get("sim"))
    im = ax_b.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=1.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if value > 0.75 or value < -0.2 else colors["text"]
                ax_b.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color=text_color, fontweight="bold")
    expected = {"COVID+ (2022-12)": "Respiratory", "Heart Disease": "Cardiovascular", "Diabetes": "Diabetes"}
    for i, ref in enumerate(ref_names):
        if ref in expected:
            j = top_comor.index(expected[ref])
            ax_b.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#333333", linewidth=2.0))
    ax_b.set_xticks(range(len(top_comor)))
    ax_b.set_xticklabels([short[c] for c in top_comor], fontsize=8)
    ax_b.set_yticks(range(len(ref_names)))
    ax_b.set_yticklabels(ref_short, fontsize=8.5)
    ax_b.set_xlabel("Target comorbidity group")
    ax_b.set_ylabel("Reference vector")
    ax_b.set_title("Original 20K cohort: organ-system specificity")
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.035, pad=0.03, shrink=0.82)
    cbar.set_label("Similarity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    original_unique = int(dedup["original_cardiac_csv"]["unique_mrn"])
    overlap = int(dedup["overlap_unique_mrn"])
    expanded_only = int(dedup["expanded_only_unique_mrn"])
    original_only = int(dedup["original_only_unique_mrn"])
    combined = int(dedup["combined_unique_mrn_after_dedup"])
    ax_c.bar([0], [overlap], color=colors["primary"], width=0.58, label="Original retained")
    ax_c.bar([0], [original_only], bottom=[overlap], color=colors["grid"], edgecolor="#888888", width=0.58, label="Original only")
    ax_c.bar([1], [overlap], color=colors["primary"], width=0.58)
    ax_c.bar([1], [expanded_only], bottom=[overlap], color=colors["accent1"], width=0.58, label="Expanded-only")
    ax_c.plot([0, 1], [original_unique, combined], color=colors["secondary"], marker="o", linewidth=2.0, label="Deduplicated denominator")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Original\ncardiac", "Expanded\ncardiac"])
    ax_c.set_ylabel("Unique medical-record numbers")
    ax_c.set_title("Expanded cardiac cohort after de-duplication")
    ax_c.text(0, original_unique * 1.03, f"{original_unique:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_c.text(1, combined * 1.03, f"{combined:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_c.text(1.0, overlap + expanded_only / 2, f"+{expanded_only:,}\n(+{dedup['incremental_unique_mrn_pct_vs_original']:.2f}%)", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax_c.set_ylim(0, combined * 1.22)
    ax_c.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_c.text(0.03, 0.04, f"Overlap: {overlap:,} of {original_unique:,} original ({dedup['overlap_pct_of_original']:.2f}%)", transform=ax_c.transAxes, fontsize=7.5, color=colors["text"])

    save_revision_figure(fig, "Figure4_postpandemic_validation")


def build_revised_main_figure_from_recomputed_results(results: dict, dedup: dict, config: types.ModuleType) -> None:
    colors, palette, _ = get_colors(config)
    sentinel = results.get("sentinel_analysis", {})
    rdi_rows = results.get("rdi_quarterly", [])
    expanded_validation = load_json(EXPANDED_RAW_VALIDATION_RESULTS) if EXPANDED_RAW_VALIDATION_RESULTS.exists() else {}
    expanded_sentinel = expanded_validation.get("sentinel_analysis", {})
    external_summary = load_json(EXTERNAL_POSITIVE_CONTROL_SUMMARY) if EXTERNAL_POSITIVE_CONTROL_SUMMARY.exists() else {}
    mimic_rows = read_csv_rows(MIMIC_EXTERNAL_CORRELATIONS) if MIMIC_EXTERNAL_CORRELATIONS.exists() else []
    eicu_rows = read_csv_rows(EICU_EXTERNAL_CORRELATIONS) if EICU_EXTERNAL_CORRELATIONS.exists() else []
    mimic_perm = external_summary.get("mimic_permutation", {})
    eicu_perm = external_summary.get("eicu_permutation", {})
    periods = [
        ("H1 2019\npre-pandemic", "H1_2019_prepandemic"),
        ("H2 2020\npost-lockdown", "H2_2020_postlockdown"),
        ("Q4 2022\nreopening", "Q4_2022_reopening"),
        ("2023\npost-pandemic", "Post_2023"),
    ]
    comorbidities = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
    short = ["Cardio", "Hypert", "Diabetes", "Cerebro", "Renal", "Resp"]

    fig = plt.figure(figsize=(18.2, 10.6))
    gs = gridspec.GridSpec(2, 3, wspace=0.34, hspace=0.48, left=0.06, right=0.98, top=0.91, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    x = np.arange(len(comorbidities))
    width = 0.18
    for idx, (label, key) in enumerate(periods):
        sims = sentinel.get(key, {}).get("similarities", {})
        values = [safe_float(sims.get(comor)) for comor in comorbidities]
        ax_a.bar(x + (idx - 1.5) * width, values, width, label=label, color=palette[idx], edgecolor="white", linewidth=0.3)
    ax_a.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short, rotation=25, ha="right")
    ax_a.set_ylabel("Similarity to COVID+ profile")
    ax_a.set_title("Original 20K cohort: cross-period profile similarity")
    ax_a.legend(fontsize=7, framealpha=0.9, loc="upper right")
    q4_rank = sentinel.get("Q4_2022_reopening", {}).get("respiratory_rank")
    q4_resp = sentinel.get("Q4_2022_reopening", {}).get("similarities", {}).get("Respiratory")
    if q4_resp is not None:
        ax_a.text(0.03, 0.04, f"Q4 2022 Respiratory: {q4_resp:.3f}, rank #{q4_rank}", transform=ax_a.transAxes, fontsize=8, color=colors["secondary"], fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    expanded_periods = [
        ("H1 2019", "H1_2019_prepandemic"),
        ("Q4 2022", "Q4_2022_reopening"),
        ("2023", "Post_2023"),
    ]
    bx = np.arange(len(expanded_periods))
    original_resp = []
    expanded_resp = []
    for _, key in expanded_periods:
        original_resp.append(safe_float(sentinel.get(key, {}).get("similarities", {}).get("Respiratory")))
        expanded_resp.append(safe_float(expanded_sentinel.get(key, {}).get("similarities", {}).get("Respiratory")))
    ax_b.bar(bx - 0.18, original_resp, 0.36, color=colors["primary"], label="Original 20K", edgecolor="white", linewidth=0.4)
    ax_b.bar(bx + 0.18, expanded_resp, 0.36, color=colors["secondary"], label="Expanded CSV", edgecolor="white", linewidth=0.4)
    ax_b.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels([label for label, _ in expanded_periods])
    ax_b.set_ylabel("Respiratory similarity")
    ax_b.set_ylim(min(-0.15, np.nanmin(np.nan_to_num(original_resp + expanded_resp, nan=0.0)) - 0.08), 1.08)
    ax_b.set_title("Expanded cardiac cohort: same-source replication")
    ax_b.legend(fontsize=7, framealpha=0.9, loc="upper left")
    expanded_cohort = expanded_validation.get("cohort_summary", {})
    ax_b.text(
        0.03,
        0.04,
        f"CSV-derived: {fmt(expanded_cohort.get('total_admissions'))} admissions\nCOVID+ {fmt(expanded_cohort.get('covid_positive_admissions'))} admissions / {fmt(expanded_cohort.get('covid_positive_patients'))} patients",
        transform=ax_b.transAxes,
        fontsize=7.5,
        color=colors["text"],
    )

    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    external_datasets = [
        ("MIMIC-IV\ninfluenza", mimic_rows, mimic_perm, colors["accent1"]),
        ("eICU\nviral pneumonia", eicu_rows, eicu_perm, colors["accent3"]),
    ]
    ex_x = np.arange(len(external_datasets))
    ex_means, ex_low, ex_high = [], [], []
    for _, rows, _, _ in external_datasets:
        respiratory = next((row for row in rows if row.get("group") == "chronic_respiratory"), {})
        mean = safe_float(respiratory.get("pearson_profile_correlation"))
        low = safe_float(respiratory.get("bootstrap_ci_low"))
        high = safe_float(respiratory.get("bootstrap_ci_high"))
        ex_means.append(mean)
        ex_low.append(low)
        ex_high.append(high)
    ex_means_arr = np.array(ex_means, dtype=float)
    ex_err = [np.maximum(ex_means_arr - np.array(ex_low, dtype=float), 0), np.maximum(np.array(ex_high, dtype=float) - ex_means_arr, 0)]
    ax_c.bar(ex_x, ex_means_arr, yerr=ex_err, capsize=4, color=[item[3] for item in external_datasets], edgecolor="white", linewidth=0.5)
    ax_c.set_xticks(ex_x)
    ax_c.set_xticklabels([item[0] for item in external_datasets])
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("Respiratory profile correlation")
    ax_c.set_title("External positive controls")
    for idx, (_, _, perm, _) in enumerate(external_datasets):
        p_value = safe_float(perm.get("p_two_sided"))
        diff = safe_float(perm.get("observed_difference"))
        n_ref = perm.get("n_reference")
        ax_c.text(idx, ex_means_arr[idx] + ex_err[1][idx] + 0.045, f"r={ex_means_arr[idx]:.3f}\ndiff={diff:.3f}, p={p_value:.3f}\nref n={fmt(n_ref)}", ha="center", va="bottom", fontsize=7.2)

    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "D")
    group_order = ["chronic_respiratory", "cardiovascular", "hypertension", "diabetes", "cerebrovascular", "kidney"]
    group_short = ["Resp", "Cardio", "Hypert", "Diabetes", "Cerebro", "Renal"]
    dx = np.arange(len(group_order))
    width_d = 0.36
    def rows_to_values(rows: list[dict[str, str]]) -> list[float]:
        by_group = {row.get("group"): row for row in rows}
        return [safe_float(by_group.get(group, {}).get("pearson_profile_correlation")) for group in group_order]
    mimic_vals = rows_to_values(mimic_rows)
    eicu_vals = rows_to_values(eicu_rows)
    ax_d.bar(dx - width_d / 2, mimic_vals, width_d, label="MIMIC-IV", color=colors["accent1"], edgecolor="white", linewidth=0.3)
    ax_d.bar(dx + width_d / 2, eicu_vals, width_d, label="eICU", color=colors["accent3"], edgecolor="white", linewidth=0.3)
    ax_d.set_xticks(dx)
    ax_d.set_xticklabels(group_short, rotation=25, ha="right")
    ax_d.set_ylabel("Profile correlation")
    ax_d.set_ylim(0, 1.05)
    ax_d.set_title("External datasets: chronic-group specificity")
    ax_d.legend(fontsize=7, framealpha=0.9, loc="upper left")
    ax_d.text(0.02, 0.04, "Respiratory group elevated in both external analyses;\nnot uniquely highest in every dataset.", transform=ax_d.transAxes, fontsize=7.2, color=colors["text"])

    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "E")
    labels = [row.get("label") for row in rdi_rows if row.get("rdi") is not None]
    rdi_values = [safe_float(row.get("rdi")) for row in rdi_rows if row.get("rdi") is not None]
    rx = np.arange(len(rdi_values))
    ax_e.bar(rx, rdi_values, color=[colors["secondary"] if value > 0.2 else colors["primary"] for value in rdi_values], edgecolor="white", linewidth=0.3)
    tick_step = max(1, len(labels) // 12)
    ax_e.set_xticks(rx[::tick_step])
    ax_e.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)
    ax_e.axhline(0, color="black", linewidth=0.6)
    ax_e.set_ylabel("Respiratory Dominance Index")
    ax_e.set_title("Original 20K cohort: quarterly RDI")
    if rdi_values:
        lower = min(0.0, float(np.nanmin(rdi_values)) - 0.05)
        upper = max(0.65, float(np.nanmax(rdi_values)) + 0.08)
        ax_e.set_ylim(lower, upper)
    if rdi_values:
        peak_idx = int(np.nanargmax(rdi_values))
        ax_e.annotate(
            f"Peak {labels[peak_idx]}\nRDI={rdi_values[peak_idx]:.3f}",
            xy=(peak_idx, rdi_values[peak_idx]),
            xytext=(max(0, peak_idx - 6), min(rdi_values[peak_idx] + 0.12, ax_e.get_ylim()[1] - 0.04)),
            fontsize=7.5,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": colors["secondary"], "lw": 1.0},
        )

    ax_f = fig.add_subplot(gs[1, 2])
    panel_label(ax_f, "F")
    original_unique = int(dedup["original_cardiac_csv"]["unique_mrn"])
    overlap = int(dedup["overlap_unique_mrn"])
    expanded_only = int(dedup["expanded_only_unique_mrn"])
    original_only = int(dedup["original_only_unique_mrn"])
    combined = int(dedup["combined_unique_mrn_after_dedup"])
    ax_f.bar([0], [overlap], color=colors["primary"], width=0.58, label="Original retained")
    ax_f.bar([0], [original_only], bottom=[overlap], color=colors["grid"], edgecolor="#888888", width=0.58, label="Original only")
    ax_f.bar([1], [overlap], color=colors["primary"], width=0.58)
    ax_f.bar([1], [expanded_only], bottom=[overlap], color=colors["accent1"], width=0.58, label="Expanded-only")
    ax_f.plot([0, 1], [original_unique, combined], color=colors["secondary"], marker="o", linewidth=2.0, label="Deduplicated denominator")
    ax_f.set_xticks([0, 1])
    ax_f.set_xticklabels(["Original\ncardiac", "Expanded\ncardiac"])
    ax_f.set_ylabel("Unique medical-record numbers")
    ax_f.set_title("Expanded cardiac exports: denominator gain")
    ax_f.text(0, original_unique * 1.03, f"{original_unique:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_f.text(1, combined * 1.03, f"{combined:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_f.text(1.0, overlap + expanded_only / 2, f"+{expanded_only:,}\n(+{dedup['incremental_unique_mrn_pct_vs_original']:.2f}%)", ha="center", va="center", fontsize=8.2, color="white", fontweight="bold")
    ax_f.set_ylim(0, combined * 1.22)
    ax_f.legend(fontsize=6.8, loc="upper left", framealpha=0.9)
    ax_f.text(0.03, 0.04, f"Overlap: {overlap:,}/{original_unique:,} original ({dedup['overlap_pct_of_original']:.2f}%)", transform=ax_f.transAxes, fontsize=7.2, color=colors["text"])

    fig.suptitle("Validation Evidence: Secondary Cardiac Cohort and External Positive Controls", fontsize=13, fontweight="bold")
    save_revision_figure(fig, "Figure4_postpandemic_validation")


def build_revised_supplement_figure(results: dict, dedup: dict, source_rows: list[dict[str, str]], disease_rows: list[dict[str, str]], config: types.ModuleType) -> None:
    colors, palette, _ = get_colors(config)
    top_comor = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
    boot_ci = results.get("bootstrap_ci", {})
    rdi_rows = results.get("rdi_quarterly", [])
    fig = plt.figure(figsize=(15.5, 10.5))
    gs = gridspec.GridSpec(2, 2, wspace=0.32, hspace=0.42, left=0.07, right=0.96, top=0.92, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    means, lo, hi = [], [], []
    for comor in top_comor:
        ci = boot_ci.get(comor, {})
        means.append(safe_float(ci.get("mean")))
        lo.append(safe_float(ci.get("ci_lo", ci.get("ci_low"))))
        hi.append(safe_float(ci.get("ci_hi", ci.get("ci_high"))))
    means = np.nan_to_num(np.array(means), nan=0.0)
    lo = np.nan_to_num(np.array(lo), nan=0.0)
    hi = np.nan_to_num(np.array(hi), nan=0.0)
    y = np.arange(len(top_comor))
    err = [np.maximum(means - lo, 0), np.maximum(hi - means, 0)]
    bar_colors = [colors["secondary"] if c == "Respiratory" else colors["primary"] for c in top_comor]
    ax_a.barh(y, means, xerr=err, color=bar_colors, alpha=0.78, capsize=3, edgecolor="white", linewidth=0.5)
    for idx, (mean, low, high) in enumerate(zip(means, lo, hi)):
        ax_a.text(max(high + 0.03, mean + 0.04), idx, f"{mean:.2f}\n[{low:.2f}, {high:.2f}]", va="center", fontsize=7)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(top_comor)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Similarity to COVID+ profile")
    ax_a.set_title("Original 20K cohort: H1 2019 bootstrap CIs")
    ax_a.axvline(0, color="grey", linestyle=":", linewidth=0.8)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    labels = [row.get("label") for row in rdi_rows if row.get("rdi") is not None]
    rdi_vals = [safe_float(row.get("rdi")) for row in rdi_rows if row.get("rdi") is not None]
    x = np.arange(len(rdi_vals))
    ax_b.bar(x, rdi_vals, color=[colors["secondary"] if value > 0.2 else colors["primary"] for value in rdi_vals], edgecolor="white", linewidth=0.3)
    tick_step = max(1, len(labels) // 12)
    ax_b.set_xticks(x[::tick_step])
    ax_b.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_ylabel("RDI")
    ax_b.set_title("Original 20K cohort: quarterly RDI")
    if rdi_vals:
        peak_idx = int(np.nanargmax(rdi_vals))
        ax_b.annotate(f"Peak {labels[peak_idx]}\nRDI={rdi_vals[peak_idx]:.2f}", xy=(peak_idx, rdi_vals[peak_idx]), xytext=(max(0, peak_idx - 7), rdi_vals[peak_idx] + 0.18), fontsize=7.5, arrowprops={"arrowstyle": "->", "color": colors["secondary"]})

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    source_labels = [row["source_folder"].replace("电子病历信息", "Export ").replace("副本", "") for row in source_rows]
    source_labels = [label[:18] for label in source_labels]
    overlap = np.array([int(row["overlap_with_original_mrn"]) for row in source_rows])
    new_only = np.array([int(row["new_only_mrn_vs_original"]) for row in source_rows])
    sx = np.arange(len(source_rows))
    ax_c.bar(sx, overlap, color=colors["primary"], label="Overlap with original")
    ax_c.bar(sx, new_only, bottom=overlap, color=colors["accent1"], label="Expanded-only")
    ax_c.set_xticks(sx)
    ax_c.set_xticklabels(source_labels, rotation=25, ha="right", fontsize=7)
    ax_c.set_ylabel("Unique MRNs")
    ax_c.set_title("Expanded exports: overlap and added denominator")
    for idx, total in enumerate(overlap + new_only):
        ax_c.text(idx, total * 1.02, f"{total:,}", ha="center", va="bottom", fontsize=8)
    ax_c.legend(fontsize=7, framealpha=0.9)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "D")
    disease_order = ["cardiovascular", "respiratory", "renal", "diabetes", "cerebrovascular", "covid_related"]
    disease_map = {row["disease_group"]: row for row in disease_rows}
    counts = [int(disease_map[d]["linked_patient_component_count"]) for d in disease_order if d in disease_map]
    labels_d = [d.replace("_", " ").title() for d in disease_order if d in disease_map]
    bar_cols = [palette[i % len(palette)] for i in range(len(counts))]
    ax_d.barh(np.arange(len(counts)), counts, color=bar_cols, edgecolor="white", linewidth=0.5)
    ax_d.set_yticks(np.arange(len(counts)))
    ax_d.set_yticklabels(labels_d)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Linked patient components")
    ax_d.set_title("Expanded exports: diagnosis burden audit")
    for idx, value in enumerate(counts):
        ax_d.text(value * 1.01, idx, f"{value:,}", va="center", fontsize=8)

    fig.suptitle("Secondary Cardiac Cohort Evidence", fontsize=13, fontweight="bold")
    save_revision_figure(fig, "FigureS6_postpandemic_extended")


def build_results_and_report(config: types.ModuleType) -> None:
    sentinel_results = load_json(RECOMPUTED_SENTINEL_RESULTS)
    expanded_validation = load_json(EXPANDED_RAW_VALIDATION_RESULTS) if EXPANDED_RAW_VALIDATION_RESULTS.exists() else {}
    external_summary = load_json(EXTERNAL_POSITIVE_CONTROL_SUMMARY) if EXTERNAL_POSITIVE_CONTROL_SUMMARY.exists() else {}
    mimic_external = read_csv_rows(MIMIC_EXTERNAL_CORRELATIONS) if MIMIC_EXTERNAL_CORRELATIONS.exists() else []
    eicu_external = read_csv_rows(EICU_EXTERNAL_CORRELATIONS) if EICU_EXTERNAL_CORRELATIONS.exists() else []
    expanded = load_json(EXPANDED_SUMMARY)
    dedup = load_json(DEDUP_SUMMARY)
    source_rows = read_csv_rows(SOURCE_OVERLAP)
    disease_rows = read_csv_rows(DISEASE_COUNTS)
    folder_rows = read_csv_rows(FOLDER_SUMMARY)

    build_revised_main_figure(sentinel_results, dedup, source_rows, config)
    build_revised_supplement_figure(sentinel_results, dedup, source_rows, disease_rows, config)

    original_unique = int(dedup["original_cardiac_csv"]["unique_mrn"])
    combined = int(dedup["combined_unique_mrn_after_dedup"])
    expanded_only = int(dedup["expanded_only_unique_mrn"])
    overlap = int(dedup["overlap_unique_mrn"])
    original_only = int(dedup["original_only_unique_mrn"])
    expansion_factor = round(combined / original_unique, 3) if original_unique else 0
    expanded_share = pct(expanded_only, combined)

    revised = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_scripts": {
            "original_prospective_validation": str(ORIGINAL_PROSPECTIVE_SCRIPT),
            "original_figure4": str(ORIGINAL_FIG4_SCRIPT),
            "revision_driver": str(Path(__file__).resolve()),
        },
        "original_20k_prospective_validation_recomputed": sentinel_results,
        "expanded_csv_wide_validation": expanded_validation,
        "external_positive_control_validation": {
            "summary": external_summary,
            "mimic_profile_correlations": mimic_external,
            "eicu_profile_correlations": eicu_external,
        },
        "expanded_cardiac_cohort_audit": expanded,
        "original_vs_expanded_deduplication": dedup,
        "source_overlap": source_rows,
        "disease_keyword_counts": disease_rows,
        "folder_summary": folder_rows,
        "revision_interpretation": {
            "deduplicated_unique_mrn_after_expansion": combined,
            "expanded_only_unique_mrn": expanded_only,
            "expansion_factor_vs_original_unique_mrn": expansion_factor,
            "expanded_only_share_of_combined_unique_mrn_pct": expanded_share,
            "what_changed": "The original 20K profile/RDI estimates were regenerated, the expanded cardiac CSV exports were reconstructed into the admission-level wide validation schema and rerun, and public external positive-control datasets (MIMIC-IV influenza-coded admissions and eICU viral-pneumonia ICU stays) were added to the main validation evidence.",
            "claim_strength": "The revised evidence strengthens the sample-size, overlap-audit, same-source replication, and external positive-control response to the reviewers. The cardiac cohort remains a secondary same-health-system replication, while MIMIC-IV/eICU provide independent public-dataset support for respiratory-burden sensitivity.",
        },
    }
    with REVISED_RESULTS.open("w", encoding="utf-8") as handle:
        json.dump(revised, handle, ensure_ascii=False, indent=2)

    cohort = sentinel_results.get("cohort_summary", {})
    sentinel = sentinel_results.get("sentinel_analysis", {})
    q4 = sentinel.get("Q4_2022_reopening", {})
    h1 = sentinel.get("H1_2019_prepandemic", {})
    perm = sentinel_results.get("permutation_test", {})
    expanded_cohort = expanded_validation.get("cohort_summary", {})
    expanded_sentinel = expanded_validation.get("sentinel_analysis", {})
    expanded_h1 = expanded_sentinel.get("H1_2019_prepandemic", {})
    expanded_q4 = expanded_sentinel.get("Q4_2022_reopening", {})
    expanded_perm = expanded_validation.get("permutation_test", {})
    expanded_rdi_rows = [row for row in expanded_validation.get("rdi_quarterly", []) if row.get("rdi") is not None]
    expanded_rdi_peak = max(expanded_rdi_rows, key=lambda row: row.get("rdi", float("-inf"))) if expanded_rdi_rows else {}
    mimic_resp = next((row for row in mimic_external if row.get("group") == "chronic_respiratory"), {})
    eicu_resp = next((row for row in eicu_external if row.get("group") == "chronic_respiratory"), {})
    mimic_perm = external_summary.get("mimic_permutation", {})
    eicu_perm = external_summary.get("eicu_permutation", {})
    covid_positive_admissions = cohort.get("covid_positive_admissions")
    covid_positive_patients = cohort.get("covid_positive_patients")
    q4_resp = q4.get("similarities", {}).get("Respiratory")
    h1_resp = h1.get("similarities", {}).get("Respiratory")
    q4_rank = q4.get("respiratory_rank")
    h1_rank = h1.get("respiratory_rank")

    lines = [
        "# Revised Cardiac Analysis Change Report",
        "",
        f"Generated: {revised['generated_at']}",
        "",
        "## Regeneration Scope",
        "",
        "The original GitHub cardiac validation workflow was rerun into `NC_revision`; the expanded cardiac CSV exports were reconstructed into the same admission-level validation schema; and external public positive-control datasets were promoted into the main validation figure.",
        "",
        "## Original 20K Validation Results Recomputed",
        "",
        f"- Admissions in original cardiac wide table: {fmt(cohort.get('total_admissions'))}",
        f"- Unique original cardiac 病案号: {fmt(cohort.get('unique_patients'))}",
        f"- COVID-positive admissions/patients: {fmt(covid_positive_admissions)} / {fmt(covid_positive_patients)}",
        f"- H1 2019 respiratory similarity/rank: {h1_resp:.4f} / rank {h1_rank}" if h1_resp is not None else "- H1 2019 respiratory similarity/rank: NA",
        f"- Q4 2022 respiratory similarity/rank: {q4_resp:.4f} / rank {q4_rank}" if q4_resp is not None else "- Q4 2022 respiratory similarity/rank: NA",
        f"- H1 2019 vs H1 2018 permutation p-value: {perm.get('p_value'):.4f}" if perm.get("p_value") is not None else "- H1 2019 vs H1 2018 permutation p-value: NA",
        "",
        "## New Expanded-Cohort Evidence",
        "",
        f"- Expanded exports scanned: {fmt(expanded.get('csv_file_count'))} CSV files, approximately {fmt(expanded.get('source_row_count'))} source rows",
        f"- Expanded patient-index rows: {fmt(expanded.get('patient_table_total_rows'))}",
        f"- Expanded unique 病案号 in core linkage tables: {fmt(dedup.get('expanded_unique_mrn_in_core_tables'))}",
        f"- Original-vs-expanded overlap by 病案号: {overlap:,} overlapping, {original_only:,} original-only, {expanded_only:,} expanded-only",
        f"- Combined unique 病案号 after de-duplication: {combined:,}",
        f"- Increment vs original: +{expanded_only:,} unique 病案号 (+{dedup.get('incremental_unique_mrn_pct_vs_original'):.2f}%), expansion factor {expansion_factor}x",
        "",
        "## Reconstructed Wide-Table Cardiac Replication",
        "",
        "Because the prior 20K cardiac cohort is contained within the expanded exports, the revision uses the CSV-derived expanded wide table as the primary same-source cardiac replication rather than treating missing legacy preprocessing code as a blocker.",
        "",
        f"- CSV-derived expanded wide table: {fmt(expanded_cohort.get('total_admissions'))} admissions, {fmt(expanded_cohort.get('unique_patients'))} unique 病案号",
        f"- COVID-positive admissions/patients: {fmt(expanded_cohort.get('covid_positive_admissions'))} / {fmt(expanded_cohort.get('covid_positive_patients'))}",
        f"- H1 2019 respiratory similarity/rank: {expanded_h1.get('similarities', {}).get('Respiratory'):.4f} / rank {expanded_h1.get('respiratory_rank')}" if expanded_h1.get("similarities", {}).get("Respiratory") is not None else "- H1 2019 respiratory similarity/rank: NA",
        f"- Q4 2022 respiratory similarity/rank: {expanded_q4.get('similarities', {}).get('Respiratory'):.4f} / rank {expanded_q4.get('respiratory_rank')}" if expanded_q4.get("similarities", {}).get("Respiratory") is not None else "- Q4 2022 respiratory similarity/rank: NA",
        f"- H1 2019 vs H1 2018 permutation p-value: {expanded_perm.get('p_value'):.4f}" if expanded_perm.get("p_value") is not None else "- H1 2019 vs H1 2018 permutation p-value: NA",
        f"- RDI peak: {expanded_rdi_peak.get('label')} (RDI {expanded_rdi_peak.get('rdi'):.3f})" if expanded_rdi_peak else "- RDI peak: NA",
        "",
        "## External Public-Dataset Positive Controls Added to Main Figure",
        "",
        f"- MIMIC-IV v2.2 influenza-coded admissions: reference n={fmt(mimic_perm.get('n_reference'))}; chronic respiratory profile correlation {safe_float(mimic_resp.get('pearson_profile_correlation')):.3f} (bootstrap 95% CI {safe_float(mimic_resp.get('bootstrap_ci_low')):.3f}-{safe_float(mimic_resp.get('bootstrap_ci_high')):.3f}); respiratory-vs-nonrespiratory chronic difference {safe_float(mimic_perm.get('observed_difference')):.3f}, permutation p={safe_float(mimic_perm.get('p_two_sided')):.3f}",
        f"- eICU-CRD v2.0 viral-pneumonia ICU stays: reference n={fmt(eicu_perm.get('n_reference'))}; chronic respiratory profile correlation {safe_float(eicu_resp.get('pearson_profile_correlation')):.3f} (bootstrap 95% CI {safe_float(eicu_resp.get('bootstrap_ci_low')):.3f}-{safe_float(eicu_resp.get('bootstrap_ci_high')):.3f}); respiratory-vs-nonrespiratory chronic difference {safe_float(eicu_perm.get('observed_difference')):.3f}, permutation p={safe_float(eicu_perm.get('p_two_sided')):.3f}",
        "- These external analyses are framed as public-dataset positive controls for respiratory-burden sensitivity, not as proof that respiratory patients are uniquely highest in every dataset.",
        "",
        "## Interpretation Change",
        "",
        "The expanded data materially improve the cardiac-cohort denominator and directly answer the reviewer concern that overlap with the prior 20K cardiac cohort was undocumented. The revised evidence shows that the expanded exports contain almost all of the old cardiac cohort and add 24,902 new unique 病案号 after de-duplication.",
        "",
        "The main figure now combines three reviewer-facing evidence layers: original cardiac rerun, expanded same-source cardiac replication, and independent public-dataset positive controls. This directly addresses the criticism that the prior cardiac validation alone was too small and insufficiently independent.",
        "",
        "Bottom line: the larger cardiac data strengthen same-health-system replication and overlap transparency, while MIMIC-IV/eICU add external public-dataset support for the respiratory-burden-sensitive pattern requested by the reviewers. The language should remain measured: these results support plausibility and robustness, not a deployed prospective surveillance claim.",
        "",
        "## Regenerated Files",
        "",
        "- Active package main Figure 4: Figure4_v3.png/pdf/svg (assembled from revised_main_panels_v3)",
        "- FigureS6_postpandemic_extended.png/pdf/tif/svg",
        "- Figure5_event_driven_sentinel.png/pdf/tif",
        "- prospective_validation_results.json",
        "- prospective_validation_results_recomputed.json",
        "- prospective_validation/prospective_validation_results.json",
        "- revised_cardiac_validation_results.json",
        "- prospective_validation_results_expanded_raw.json",
        "- external_positive_control_results/external_positive_control_summary.json",
        "- external_positive_control_results/mimic_influenza_profile_correlations.csv/png",
        "- external_positive_control_results/eicu_viral_pneumonia_profile_correlations.csv/png",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": revised["generated_at"],
        "output_directory": str(BASE_DIR),
        "source_scripts": revised["source_scripts"],
        "outputs": [
            str(path.relative_to(BASE_DIR))
            for path in sorted(BASE_DIR.glob("Figure4_postpandemic_validation*.*"))
            + sorted(BASE_DIR.glob("FigureS6_postpandemic_extended*.*"))
            + sorted(BASE_DIR.glob("Figure5_event_driven_sentinel*.*"))
        ] + [
            "prospective_validation_results.json",
            "prospective_validation/prospective_validation_results.json",
            "prospective_validation_results_expanded_raw.json",
            "expanded_raw_validation_comparison.md",
            "external_positive_control_results/external_positive_control_summary.json",
            "external_positive_control_results/mimic_influenza_profile_correlations.csv",
            "external_positive_control_results/mimic_influenza_profile_correlations.png",
            "external_positive_control_results/eicu_viral_pneumonia_profile_correlations.csv",
            "external_positive_control_results/eicu_viral_pneumonia_profile_correlations.png",
            REVISED_RESULTS.name,
            REPORT_PATH.name,
        ],
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def group_value_map(rows: list[dict[str, str]], value_col: str = "pearson_profile_correlation") -> dict[str, float]:
    return {row.get("group", ""): safe_float(row.get(value_col)) for row in rows}


def respiratory_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return next((row for row in rows if row.get("group") == "chronic_respiratory"), {})


def display_token(value: object) -> str:
    return str(value or "NA").replace("_", " ").replace("ICD10", "ICD-10")


def remove_stale_original_outputs() -> None:
    for pattern in [
        "Figure5_event_driven_sentinel*.*",
        "FigureS6_postpandemic_extended*.*",
    ]:
        for path in BASE_DIR.glob(pattern):
            path.unlink(missing_ok=True)


def build_current_main_figure(
    expanded_validation: dict,
    dedup: dict,
    validation_overlap: dict,
    external_summary: dict,
    mimic_rows: list[dict[str, str]],
    nwicu_rows: list[dict[str, str]],
    eicu_rows: list[dict[str, str]],
    config: types.ModuleType,
) -> None:
    colors, palette, _ = get_colors(config)
    expanded_sentinel = expanded_validation.get("sentinel_analysis", {})
    expanded_rdi = expanded_validation.get("rdi_quarterly", [])
    expanded_ci = expanded_validation.get("bootstrap_ci", {})
    expanded_cohort = expanded_validation.get("cohort_summary", {})
    expanded_unique_mrn_label = fmt(expanded_cohort.get("unique_patients"))
    expanded_admissions_label = fmt(expanded_cohort.get("total_admissions"))
    mimic_perm = external_summary.get("mimic_permutation", {})
    nwicu_perm = external_summary.get("nwicu_permutation", {})
    eicu_perm = external_summary.get("eicu_permutation", {})

    comorbidities = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
    short = ["Cardio", "Hypert", "Diabetes", "Cerebro", "Renal", "Resp"]
    periods = [
        ("H1 2019", "H1_2019_prepandemic"),
        ("H2 2020", "H2_2020_postlockdown"),
        ("Q4 2022", "Q4_2022_reopening"),
        ("2023", "Post_2023"),
    ]

    fig = plt.figure(figsize=(17.2, 10.8))
    gs = gridspec.GridSpec(2, 3, wspace=0.36, hspace=0.52, left=0.06, right=0.98, top=0.91, bottom=0.10)

    ax_venn = fig.add_subplot(gs[0, 0])
    panel_label(ax_venn, "A")
    draw_mrn_overlap_venn(ax_venn, validation_overlap, colors)

    ax_a = fig.add_subplot(gs[0, 1])
    panel_label(ax_a, "B")
    x = np.arange(len(comorbidities))
    width = 0.18
    for idx, (label, key) in enumerate(periods):
        values = [safe_float(expanded_sentinel.get(key, {}).get("similarities", {}).get(comor)) for comor in comorbidities]
        ax_a.bar(x + (idx - 1.5) * width, values, width, label=label, color=palette[idx], edgecolor="white", linewidth=0.3)
    ax_a.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short, rotation=25, ha="right")
    ax_a.set_ylim(-0.2, 1.08)
    ax_a.set_ylabel("Similarity to COVID+ profile")
    ax_a.set_title(f"Expanded cardiac cohort ({expanded_unique_mrn_label} MRNs): profile similarity")
    ax_a.legend(fontsize=7, framealpha=0.9, loc="upper left")
    q4_resp = expanded_sentinel.get("Q4_2022_reopening", {}).get("similarities", {}).get("Respiratory")
    q4_rank = expanded_sentinel.get("Q4_2022_reopening", {}).get("respiratory_rank")
    if q4_resp is not None:
        ax_a.text(0.03, 0.04, f"Q4 2022 respiratory: {q4_resp:.3f}, rank #{q4_rank}", transform=ax_a.transAxes, fontsize=8, color=colors["secondary"], fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 2])
    panel_label(ax_b, "C")
    external_specs = [
        ("MIMIC-IV\ninfluenza", mimic_rows, mimic_perm, colors["accent1"]),
        ("NWICU\nCOVID", nwicu_rows, nwicu_perm, colors["secondary"]),
        ("eICU\nviral pneumonia", eicu_rows, eicu_perm, colors["accent3"]),
    ]
    ex_x = np.arange(len(external_specs))
    means, lows, highs = [], [], []
    for _, rows, _, _ in external_specs:
        row = respiratory_row(rows)
        means.append(safe_float(row.get("pearson_profile_correlation")))
        lows.append(safe_float(row.get("bootstrap_ci_low")))
        highs.append(safe_float(row.get("bootstrap_ci_high")))
    means_arr = np.array(means, dtype=float)
    err = [np.maximum(means_arr - np.array(lows, dtype=float), 0), np.maximum(np.array(highs, dtype=float) - means_arr, 0)]
    ax_b.bar(ex_x, means_arr, yerr=err, capsize=4, color=[spec[3] for spec in external_specs], edgecolor="white", linewidth=0.5)
    ax_b.axhline(0, color="black", linewidth=0.7)
    ax_b.set_xticks(ex_x)
    ax_b.set_xticklabels([spec[0] for spec in external_specs])
    ax_b.set_ylim(-0.42, 1.08)
    ax_b.set_ylabel("Chronic respiratory profile correlation")
    ax_b.set_title("Public external endpoints")
    for idx, (_, _, perm, _) in enumerate(external_specs):
        p_value = safe_float(perm.get("p_two_sided"))
        diff = safe_float(perm.get("observed_difference"))
        n_ref = perm.get("n_reference")
        y_pos = means_arr[idx] + err[1][idx] + 0.045 if means_arr[idx] >= 0 else -0.35
        va = "bottom" if means_arr[idx] >= 0 else "bottom"
        ax_b.text(idx, y_pos, f"r={means_arr[idx]:.3f}\ndiff={diff:.3f}, p={p_value:.3f}\nref n={fmt(n_ref)}", ha="center", va=va, fontsize=6.8)

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "D")
    group_order = ["chronic_respiratory", "cardiovascular", "hypertension", "diabetes", "cerebrovascular", "kidney"]
    group_short = ["Resp", "Cardio", "Hypert", "Diabetes", "Cerebro", "Renal"]
    matrix = np.array([
        [group_value_map(mimic_rows).get(group, np.nan) for group in group_order],
        [group_value_map(nwicu_rows).get(group, np.nan) for group in group_order],
        [group_value_map(eicu_rows).get(group, np.nan) for group in group_order],
    ])
    im = ax_c.imshow(matrix, cmap=plt.get_cmap("RdBu_r"), vmin=-0.35, vmax=1.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if value > 0.65 or value < -0.18 else colors["text"]
                ax_c.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
    ax_c.set_xticks(range(len(group_short)))
    ax_c.set_xticklabels(group_short, rotation=25, ha="right")
    ax_c.set_yticks(range(3))
    ax_c.set_yticklabels(["MIMIC-IV flu", "NWICU COVID", "eICU viral pneumonia"], fontsize=8)
    ax_c.set_title("External datasets: chronic-group specificity")
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.035, pad=0.03, shrink=0.82)
    cbar.set_label("Profile correlation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "E")
    labels = [row.get("label") for row in expanded_rdi if row.get("rdi") is not None]
    rdi_values = [safe_float(row.get("rdi")) for row in expanded_rdi if row.get("rdi") is not None]
    rx = np.arange(len(rdi_values))
    ax_d.bar(rx, rdi_values, color=[colors["secondary"] if value > 0.2 else colors["primary"] for value in rdi_values], edgecolor="white", linewidth=0.3)
    tick_step = max(1, len(labels) // 12)
    ax_d.set_xticks(rx[::tick_step])
    ax_d.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)
    ax_d.axhline(0, color="black", linewidth=0.6)
    ax_d.set_ylabel("Respiratory Dominance Index")
    ax_d.set_title(f"Expanded cardiac cohort ({expanded_unique_mrn_label} MRNs): quarterly RDI")
    if rdi_values:
        lower = max(0.0, float(np.nanmin(rdi_values)) - 0.05)
        upper = max(0.65, float(np.nanmax(rdi_values)) + 0.08)
        ax_d.set_ylim(lower, upper)
        peak_idx = int(np.nanargmax(rdi_values))
        ax_d.annotate(
            f"Peak {labels[peak_idx]}\nRDI={rdi_values[peak_idx]:.3f}",
            xy=(peak_idx, rdi_values[peak_idx]),
            xytext=(max(0, peak_idx - 6), min(rdi_values[peak_idx] + 0.12, upper - 0.04)),
            fontsize=7.5,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": colors["secondary"], "lw": 1.0},
        )

    ax_e = fig.add_subplot(gs[1, 2])
    panel_label(ax_e, "F")
    means_ci, lo_ci, hi_ci = [], [], []
    for comor in comorbidities:
        ci = expanded_ci.get(comor, {})
        means_ci.append(safe_float(ci.get("mean")))
        lo_ci.append(safe_float(ci.get("ci_lo", ci.get("ci_low"))))
        hi_ci.append(safe_float(ci.get("ci_hi", ci.get("ci_high"))))
    means_ci = np.array(means_ci, dtype=float)
    lo_ci = np.array(lo_ci, dtype=float)
    hi_ci = np.array(hi_ci, dtype=float)
    y = np.arange(len(comorbidities))
    ax_e.barh(y, means_ci, xerr=[np.maximum(means_ci - lo_ci, 0), np.maximum(hi_ci - means_ci, 0)], color=[colors["secondary"] if c == "Respiratory" else colors["primary"] for c in comorbidities], alpha=0.78, capsize=3, edgecolor="white", linewidth=0.5)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(short)
    ax_e.invert_yaxis()
    ax_e.axvline(0, color="grey", linestyle=":", linewidth=0.8)
    ax_e.set_xlim(-0.05, 0.9)
    ax_e.set_xlabel("Similarity to COVID+ profile")
    ax_e.set_title(f"Expanded cardiac cohort ({expanded_unique_mrn_label} MRNs): bootstrap CIs")

    fig.suptitle("Validation Evidence: Expanded Cardiac Cohort and Public External Datasets", fontsize=13, fontweight="bold")
    save_revision_figure(fig, "Figure4_postpandemic_validation")


def build_external_covid_figure(
    external_summary: dict,
    mimic_rows: list[dict[str, str]],
    nwicu_rows: list[dict[str, str]],
    eicu_rows: list[dict[str, str]],
    who_rows: list[dict[str, str]],
    lgdi_flu_summary: dict,
    cardiac42k_summary: dict,
    config: types.ModuleType,
) -> None:
    colors, _, _ = get_colors(config)
    group_order = ["chronic_respiratory", "cardiovascular", "hypertension", "diabetes", "cerebrovascular", "kidney"]
    label_map = {
        "chronic_respiratory": "Chronic respiratory",
        "cardiovascular": "Cardiovascular",
        "hypertension": "Hypertension",
        "diabetes": "Diabetes",
        "cerebrovascular": "Cerebrovascular",
        "kidney": "Kidney",
    }
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, wspace=0.38, hspace=0.52, left=0.07, right=0.98, top=0.91, bottom=0.10)

    # --- Panel A: NWICU COVID stress test (public EHR foreign-validation) ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    nw_by_group = {row.get("group"): row for row in nwicu_rows}
    values = np.array([safe_float(nw_by_group.get(group, {}).get("pearson_profile_correlation")) for group in group_order])
    lows = np.array([safe_float(nw_by_group.get(group, {}).get("bootstrap_ci_low")) for group in group_order])
    highs = np.array([safe_float(nw_by_group.get(group, {}).get("bootstrap_ci_high")) for group in group_order])
    order = np.argsort(values)
    y = np.arange(len(group_order))
    bar_colors = [colors["secondary"] if group_order[i] == "chronic_respiratory" else colors["primary"] for i in order]
    ax_a.barh(y, values[order], xerr=[np.maximum(values[order] - lows[order], 0), np.maximum(highs[order] - values[order], 0)], color=bar_colors, capsize=3, edgecolor="white", linewidth=0.4)
    ax_a.axvline(0, color="black", linewidth=0.7)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([label_map[group_order[i]] for i in order], fontsize=7.5)
    ax_a.set_xlabel("Profile correlation")
    ax_a.set_title("NWICU COVID stress test\n(utilization-profile correlation)", fontsize=9)

    # --- Panel B: Respiratory profile correlations across public EHR endpoints ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    specs = [("MIMIC-IV flu", mimic_rows, external_summary.get("mimic_permutation", {})), ("NWICU COVID", nwicu_rows, external_summary.get("nwicu_permutation", {})), ("eICU viral pneumonia", eicu_rows, external_summary.get("eicu_permutation", {}))]
    x = np.arange(len(specs))
    resp_means, resp_lows, resp_highs = [], [], []
    for _, rows, _ in specs:
        row = respiratory_row(rows)
        resp_means.append(safe_float(row.get("pearson_profile_correlation")))
        resp_lows.append(safe_float(row.get("bootstrap_ci_low")))
        resp_highs.append(safe_float(row.get("bootstrap_ci_high")))
    resp_means = np.array(resp_means, dtype=float)
    resp_lows = np.array(resp_lows, dtype=float)
    resp_highs = np.array(resp_highs, dtype=float)
    ax_b.bar(x, resp_means, yerr=[np.maximum(resp_means - resp_lows, 0), np.maximum(resp_highs - resp_means, 0)], color=[colors["accent1"], colors["secondary"], colors["accent3"]], capsize=4, edgecolor="white")
    ax_b.axhline(0, color="black", linewidth=0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([spec[0] for spec in specs], rotation=15, ha="right", fontsize=7.5)
    ax_b.set_ylim(-0.35, 1.05)
    ax_b.set_ylabel("Chronic respiratory correlation")
    ax_b.set_title("Respiratory utilization profile\nacross public EHR endpoints", fontsize=9)

    # --- Panel C: ROC-AUC comparison: WHU-32k vs Cardiac-42k ---
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "C")
    whu_auc_resp = safe_float(lgdi_flu_summary.get("roc_auc_resp_score"))
    whu_auc_lgdi = safe_float(lgdi_flu_summary.get("roc_auc_lgdi"))
    c42k_auc_resp = safe_float(cardiac42k_summary.get("roc_auc_resp_sim"))
    c42k_auc_rdi = safe_float(cardiac42k_summary.get("roc_auc_rdi"))
    # Grouped: [Pearson/resp group, LGDI/RDI group]
    group_x = np.array([0.0, 1.0])
    bw = 0.35
    e_whu_vals = [whu_auc_resp, whu_auc_lgdi]
    e_c42k_vals = [c42k_auc_resp, c42k_auc_rdi]
    ax_d.bar(group_x - bw / 2, e_whu_vals, bw, label="WHU-32k", color=colors["secondary"], edgecolor="white", alpha=0.90)
    ax_d.bar(group_x + bw / 2, e_c42k_vals, bw, label="Cardiac-42k", color=colors["accent3"], edgecolor="white", alpha=0.90)
    ax_d.axhline(0.5, color="black", linestyle="--", linewidth=0.9, label="Chance (0.5)")
    for xi, v in zip(group_x - bw / 2, e_whu_vals):
        if np.isfinite(v):
            ax_d.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=colors["secondary"])
    for xi, v in zip(group_x + bw / 2, e_c42k_vals):
        if np.isfinite(v):
            ax_d.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=colors["accent3"])
    ax_d.set_xticks(group_x)
    ax_d.set_xticklabels(["Pearson profile\ncorr. (resp_sim)", "Scalar index\n(LGDI / RDI)"], fontsize=8)
    ax_d.set_ylim(0.3, 0.75)
    ax_d.set_ylabel("ROC-AUC (FluNet event detection)")
    ax_d.set_title("FluNet ROC-AUC\nWHU-32k vs Cardiac-42k", fontsize=9)
    ax_d.legend(frameon=False, fontsize=7)
    ax_d.grid(axis="y", alpha=0.25)

    # --- Panel D: Lag-Spearman: WHU-32k vs Cardiac-42k (lags 0-4 weeks) ---
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "D")
    # WHU-32k: bootstrap point estimates and CIs
    whu_boot_path = LGDI_WHU_LAG_BOOTSTRAP
    c42k_lag_path = LGDI_CARDIAC42K_LAG
    lag_x_pos = np.arange(5)
    bwf = 0.35
    whu_rho_pts = [np.nan] * 5
    whu_ci_lo = [np.nan] * 5
    whu_ci_hi = [np.nan] * 5
    if whu_boot_path.exists():
        import pandas as _pd
        whu_boot = _pd.read_csv(whu_boot_path)
        for _, row in whu_boot.iterrows():
            lag_idx = int(row["lag_weeks"])
            if 0 <= lag_idx <= 4:
                whu_rho_pts[lag_idx] = float(row["rho_point"])
                whu_ci_lo[lag_idx] = float(row["rho_low_2_5"])
                whu_ci_hi[lag_idx] = float(row["rho_high_97_5"])
    whu_rho_pts = np.array(whu_rho_pts)
    whu_ci_lo = np.array(whu_ci_lo)
    whu_ci_hi = np.array(whu_ci_hi)
    whu_yerr_lo = np.maximum(whu_rho_pts - whu_ci_lo, 0)
    whu_yerr_hi = np.maximum(whu_ci_hi - whu_rho_pts, 0)
    ax_e.bar(lag_x_pos - bwf / 2, whu_rho_pts,
             bwf, yerr=[whu_yerr_lo, whu_yerr_hi], capsize=3,
             label="WHU-32k", color=colors["secondary"], edgecolor="white", alpha=0.90)
    c42k_rho_pts = [np.nan] * 5
    c42k_pvals = [np.nan] * 5
    if c42k_lag_path.exists():
        import pandas as _pd2
        c42k_lag = _pd2.read_csv(c42k_lag_path)
        for _, row in c42k_lag.iterrows():
            lag_idx = int(row["lgdi_leads_flunet_weeks"])
            if 0 <= lag_idx <= 4:
                c42k_rho_pts[lag_idx] = float(row["spearman_rho"])
                c42k_pvals[lag_idx] = float(row["p_value"])
    c42k_rho_pts = np.array(c42k_rho_pts)
    ax_e.bar(lag_x_pos + bwf / 2, c42k_rho_pts,
             bwf, label="Cardiac-42k", color=colors["accent3"], edgecolor="white", alpha=0.90)
    # Significance markers for WHU-32k (based on bootstrap CI excluding 0)
    for i, (rpt, lo, hi) in enumerate(zip(whu_rho_pts, whu_ci_lo, whu_ci_hi)):
        if np.isfinite(rpt) and np.isfinite(lo) and lo > 0:
            ax_e.text(lag_x_pos[i] - bwf / 2, rpt + max(whu_ci_hi[i] - rpt, 0) + 0.008,
                      "*", ha="center", va="bottom", fontsize=9, color=colors["secondary"])
    # Significance markers for Cardiac-42k
    for i, (rpt, pval) in enumerate(zip(c42k_rho_pts, c42k_pvals)):
        if np.isfinite(rpt) and np.isfinite(pval) and pval < 0.05:
            ax_e.text(lag_x_pos[i] + bwf / 2, rpt + 0.008,
                      "*", ha="center", va="bottom", fontsize=9, color=colors["accent3"])
    ax_e.axhline(0, color="black", linewidth=0.6)
    ax_e.set_xticks(lag_x_pos)
    ax_e.set_xticklabels([f"Lag {i}" for i in range(5)], fontsize=8)
    ax_e.set_xlabel("EHR index leads FluNet by (weeks)")
    ax_e.set_ylabel("Spearman \u03c1")
    ax_e.set_title("Lag-Spearman: EHR index vs FluNet\nWHU-32k vs Cardiac-42k", fontsize=9)
    ax_e.legend(frameon=False, fontsize=7)
    ax_e.grid(axis="y", alpha=0.25)

    fig.suptitle("Figure 5. External Validation: Public EHR Respiratory Profiles and Cross-Cohort FluNet Sentinel Surveillance", fontsize=12, fontweight="bold")
    save_revision_figure(fig, "Figure5_external_covid_validation")


def build_results_and_report(config: types.ModuleType) -> None:
    expanded_validation = load_json(EXPANDED_RAW_VALIDATION_RESULTS)
    external_summary = load_json(EXTERNAL_POSITIVE_CONTROL_SUMMARY)
    mimic_external = read_csv_rows(MIMIC_EXTERNAL_CORRELATIONS)
    nwicu_external = read_csv_rows(NWICU_EXTERNAL_CORRELATIONS)
    eicu_external = read_csv_rows(EICU_EXTERNAL_CORRELATIONS)
    audit_rows = read_csv_rows(EXTERNAL_COVID_DATASET_AUDIT)
    who_rows = read_csv_rows(WHO_FLUNET_ANNUAL)
    expanded = load_json(EXPANDED_SUMMARY)
    dedup = load_json(DEDUP_SUMMARY)
    validation_overlap = build_validation_wide_mrn_overlap()
    source_rows = read_csv_rows(SOURCE_OVERLAP)
    disease_rows = read_csv_rows(DISEASE_COUNTS)
    folder_rows = read_csv_rows(FOLDER_SUMMARY)

    remove_stale_original_outputs()
    build_current_main_figure(expanded_validation, dedup, validation_overlap, external_summary, mimic_external, nwicu_external, eicu_external, config)
    lgdi_flu_summary = load_json(LGDI_WHU_INFLUENZA_SUMMARY) if LGDI_WHU_INFLUENZA_SUMMARY.exists() else {}
    lgdi_cardiac42k_summary = load_json(LGDI_CARDIAC42K_SUMMARY) if LGDI_CARDIAC42K_SUMMARY.exists() else {}
    build_external_covid_figure(external_summary, mimic_external, nwicu_external, eicu_external, who_rows, lgdi_flu_summary, lgdi_cardiac42k_summary, config)

    combined = int(dedup["combined_unique_mrn_after_dedup"])
    expanded_only = int(dedup["expanded_only_unique_mrn"])
    original_unique = int(dedup["original_cardiac_csv"]["unique_mrn"])
    overlap = int(dedup["overlap_unique_mrn"])
    expansion_factor = round(combined / original_unique, 3) if original_unique else 0
    expanded_share = pct(expanded_only, combined)
    expanded_cohort = expanded_validation.get("cohort_summary", {})
    expanded_sentinel = expanded_validation.get("sentinel_analysis", {})
    expanded_h1 = expanded_sentinel.get("H1_2019_prepandemic", {})
    expanded_q4 = expanded_sentinel.get("Q4_2022_reopening", {})
    expanded_perm = expanded_validation.get("permutation_test", {})
    expanded_rdi_rows = [row for row in expanded_validation.get("rdi_quarterly", []) if row.get("rdi") is not None]
    expanded_rdi_peak = max(expanded_rdi_rows, key=lambda row: row.get("rdi", float("-inf"))) if expanded_rdi_rows else {}
    mimic_resp = respiratory_row(mimic_external)
    nwicu_resp = respiratory_row(nwicu_external)
    eicu_resp = respiratory_row(eicu_external)
    mimic_perm = external_summary.get("mimic_permutation", {})
    nwicu_perm = external_summary.get("nwicu_permutation", {})
    eicu_perm = external_summary.get("eicu_permutation", {})

    revised = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_scripts": {
            "expanded_wide_validation": str(BASE_DIR / "run_expanded_wide_validation.py"),
            "external_positive_control_analysis": str(BASE_DIR / "external_positive_control_analysis.py"),
            "revision_driver": str(Path(__file__).resolve()),
        },
        "expanded_csv_wide_validation": expanded_validation,
        "external_public_validation": {
            "summary": external_summary,
            "mimic_profile_correlations": mimic_external,
            "nwicu_covid_profile_correlations": nwicu_external,
            "eicu_profile_correlations": eicu_external,
            "covid_dataset_audit": audit_rows,
        },
        "expanded_cardiac_cohort_audit": expanded,
        "old_20k_subset_audit": {
            "interpretation": "Legacy cardiac-file overlap metadata are retained only as a separate de-duplication record and are not used for the WHU-primary-vs-expanded Figure4 Venn.",
            "original_unique_mrn": original_unique,
            "overlap_with_expanded_mrn": overlap,
            "combined_unique_mrn_after_dedup": combined,
            "expanded_only_unique_mrn": expanded_only,
        },
        "validation_wide_mrn_overlap": validation_overlap,
        "source_overlap": source_rows,
        "disease_keyword_counts": disease_rows,
        "folder_summary": folder_rows,
        "revision_interpretation": {
            "deduplicated_unique_mrn_after_expansion": combined,
            "expanded_only_unique_mrn": expanded_only,
            "expansion_factor_vs_old_20k_subset": expansion_factor,
            "expanded_only_share_of_combined_unique_mrn_pct": expanded_share,
            "what_changed": f"Original 20K cardiac plots and legacy WHU/Figure5 outputs were removed from the active main-figure path. The revised figures use the reconstructed expanded cardiac validation wide table ({fmt(expanded_cohort.get('unique_patients'))} unique MRNs) plus public external datasets: MIMIC-IV influenza, NWICU COVID, and eICU viral pneumonia.",
            "claim_strength": "The expanded cardiac cohort supports same-system replication. MIMIC/eICU support respiratory-infection positive controls. NWICU provides a US COVID stress test that does not reproduce respiratory dominance, so the manuscript should avoid universal respiratory-sentinel claims.",
        },
    }
    with REVISED_RESULTS.open("w", encoding="utf-8") as handle:
        json.dump(revised, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Revised Main-Figure and External Validation Report",
        "",
        f"Generated: {revised['generated_at']}",
        "",
        "## Regeneration Scope",
        "",
        "The main cardiac analysis uses the analysis-ready expanded validation wide table (42,795 unique 病案号). The Figure4 overlap audit now compares that table with the true WHU primary original cohort (32,056 unique 病案号 from readmission_output/all_admissions.csv), not the WHU/original cardiac wide table. Legacy original-20K grouping panels and the old WHU/Figure5 event-driven outputs were removed from the active figure manifest.",
        "",
        "## Expanded Cardiac Validation Wide Table",
        "",
        f"- CSV-derived expanded wide table: {fmt(expanded_cohort.get('total_admissions'))} admissions, {fmt(expanded_cohort.get('unique_patients'))} unique 病案号",
        f"- WHU primary original-vs-expanded overlap by 病案号: {validation_overlap['overlap_unique_mrn']:,}/{validation_overlap['original_unique_mrn']:,} WHU-primary unique 病案号 overlapped; {validation_overlap['original_only_unique_mrn']:,} WHU-primary-only and {validation_overlap['expanded_only_unique_mrn']:,} expanded-only; union {validation_overlap['union_unique_mrn']:,}. This audit uses readmission_output/all_admissions.csv, not the WHU/original cardiac wide table.",
        f"- COVID-positive admissions/patients: {fmt(expanded_cohort.get('covid_positive_admissions'))} / {fmt(expanded_cohort.get('covid_positive_patients'))}",
        f"- H1 2019 respiratory similarity/rank: {expanded_h1.get('similarities', {}).get('Respiratory'):.4f} / rank {expanded_h1.get('respiratory_rank')}" if expanded_h1.get("similarities", {}).get("Respiratory") is not None else "- H1 2019 respiratory similarity/rank: NA",
        f"- Q4 2022 respiratory similarity/rank: {expanded_q4.get('similarities', {}).get('Respiratory'):.4f} / rank {expanded_q4.get('respiratory_rank')}" if expanded_q4.get("similarities", {}).get("Respiratory") is not None else "- Q4 2022 respiratory similarity/rank: NA",
        f"- H1 2019 vs H1 2018 permutation p-value: {expanded_perm.get('p_value'):.4f}" if expanded_perm.get("p_value") is not None else "- H1 2019 vs H1 2018 permutation p-value: NA",
        f"- RDI peak: {expanded_rdi_peak.get('label')} (RDI {expanded_rdi_peak.get('rdi'):.3f})" if expanded_rdi_peak else "- RDI peak: NA",
        "",
        "## Public External Dataset Audit",
        "",
        "- NWICU v0.1.0 is the downloaded non-China public EHR dataset with a sizable COVID-coded endpoint and was added as a main external COVID stress test.",
        "- MIMIC-IV v2.2 and MIMIC-IV-ED v2.2 are pre-COVID releases; they remain useful for the seasonal influenza positive control requested by Reviewer #1.",
        "- eICU-CRD v2.0 is pre-COVID and supports a viral-pneumonia ICU positive-control replication, not COVID validation.",
        "- MIMIC-IV v3.1 is locally incomplete and is not used; WHO FluNet CSV files were analyzed separately as aggregate China-versus-world-excluding-China surveillance context.",
        "",
        "## External Validation Results Now in Main Figures",
        "",
        f"- MIMIC-IV influenza-coded admissions: reference n={fmt(mimic_perm.get('n_reference'))}; chronic respiratory r={safe_float(mimic_resp.get('pearson_profile_correlation')):.3f} (95% CI {safe_float(mimic_resp.get('bootstrap_ci_low')):.3f}-{safe_float(mimic_resp.get('bootstrap_ci_high')):.3f}); respiratory-vs-nonrespiratory chronic difference {safe_float(mimic_perm.get('observed_difference')):.3f}, permutation p={safe_float(mimic_perm.get('p_two_sided')):.3f}.",
        f"- NWICU COVID-coded admissions: reference n={fmt(nwicu_perm.get('n_reference'))}; chronic respiratory r={safe_float(nwicu_resp.get('pearson_profile_correlation')):.3f} (95% CI {safe_float(nwicu_resp.get('bootstrap_ci_low')):.3f}-{safe_float(nwicu_resp.get('bootstrap_ci_high')):.3f}); respiratory-vs-nonrespiratory chronic difference {safe_float(nwicu_perm.get('observed_difference')):.3f}, permutation p={safe_float(nwicu_perm.get('p_two_sided')):.3f}. Diabetes and kidney groups were highest in this US COVID cohort.",
        f"- eICU viral-pneumonia ICU stays: reference n={fmt(eicu_perm.get('n_reference'))}; chronic respiratory r={safe_float(eicu_resp.get('pearson_profile_correlation')):.3f} (95% CI {safe_float(eicu_resp.get('bootstrap_ci_low')):.3f}-{safe_float(eicu_resp.get('bootstrap_ci_high')):.3f}); respiratory-vs-nonrespiratory chronic difference {safe_float(eicu_perm.get('observed_difference')):.3f}, permutation p={safe_float(eicu_perm.get('p_two_sided')):.3f}.",
        "",
        "## Interpretation Change",
        "",
        "The main paper should now present the expanded cardiac validation wide table as the cardiac replication dataset and the WHU primary original cohort as the Figure4 overlap comparator. The WHU/original cardiac wide table is not used for the main overlap Venn.",
        "",
        "The external public evidence is mixed and therefore useful for reviewer response: MIMIC-IV and eICU support respiratory-infection sensitivity in larger public datasets, while NWICU provides an explicit US COVID stress test that does not reproduce respiratory dominance. This directly addresses the reviewers' independence/generalizability critique and requires cautious wording rather than overclaiming.",
        "",
        "## Regenerated Main-Figure Files",
        "",
        "- Active package main Figure 4: Figure4_v3.png/pdf/svg (assembled from revised_main_panels_v3)",
        "- Figure5_external_covid_validation.png/pdf/tif/svg",
        "- revised_cardiac_validation_results.json",
        "- whu_primary_expanded_mrn_overlap_summary.json/md",
        "- external_positive_control_results/nwicu_covid_profile_correlations.csv/png",
        "- external_positive_control_results/external_covid_dataset_audit.csv",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_outputs = [
        str(path.relative_to(BASE_DIR))
        for path in sorted(BASE_DIR.glob("Figure4_postpandemic_validation*." + "*"))
        + sorted(BASE_DIR.glob("Figure5_external_covid_validation*." + "*"))
    ] + [
        "prospective_validation_results_expanded_raw.json",
        "expanded_raw_validation_comparison.md",
        VALIDATION_WIDE_MRN_OVERLAP_JSON.name,
        VALIDATION_WIDE_MRN_OVERLAP_MD.name,
        "external_positive_control_results/external_positive_control_summary.json",
        "external_positive_control_results/external_covid_dataset_audit.csv",
        "external_positive_control_results/mimic_influenza_profile_correlations.csv",
        "external_positive_control_results/mimic_influenza_profile_correlations.png",
        "external_positive_control_results/nwicu_covid_profile_correlations.csv",
        "external_positive_control_results/nwicu_covid_profile_correlations.png",
        "external_positive_control_results/eicu_viral_pneumonia_profile_correlations.csv",
        "external_positive_control_results/eicu_viral_pneumonia_profile_correlations.png",
        REVISED_RESULTS.name,
        REPORT_PATH.name,
    ]
    manifest = {
        "generated_at": revised["generated_at"],
        "output_directory": str(BASE_DIR),
        "source_scripts": revised["source_scripts"],
        "outputs": manifest_outputs,
        "removed_legacy_outputs": [
            "Figure5_event_driven_sentinel.*",
            "Figure5_event_driven_sentinel_expanded_raw.*",
            "FigureS6_postpandemic_extended.*",
        ],
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> int:
    for required in [
        ORIGINAL_CONFIG,
        EXPANDED_SUMMARY,
        DEDUP_SUMMARY,
        SOURCE_OVERLAP,
        DISEASE_COUNTS,
        FOLDER_SUMMARY,
        EXPANDED_RAW_VALIDATION_RESULTS,
        EXTERNAL_POSITIVE_CONTROL_SUMMARY,
        MIMIC_EXTERNAL_CORRELATIONS,
        NWICU_EXTERNAL_CORRELATIONS,
        EICU_EXTERNAL_CORRELATIONS,
        EXTERNAL_COVID_DATASET_AUDIT,
        WHU_PRIMARY_ADMISSIONS_TABLE,
        EXPANDED_VALIDATION_WIDE_TABLE,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)
    config = install_revision_fig_config()
    build_results_and_report(config)
    print(f"Revised cardiac outputs regenerated in {BASE_DIR}")
    print(f"Report: {REPORT_PATH}")
    print(f"Results: {REVISED_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())