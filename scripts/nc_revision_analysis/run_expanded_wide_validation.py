from __future__ import annotations

import importlib.util
import json
import runpy
import shutil
import sys
import types
import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"data/readmission_output/figures_v2/merge/Individual-level-rhythmicity-modeling")
ORIGINAL_CONFIG = PROJECT_ROOT / "src" / "figures" / "fig_config.py"
ORIGINAL_PROSPECTIVE_SCRIPT = PROJECT_ROOT / "src" / "supplementary" / "gen_supp_prospective_validation.py"

COMBINED_WIDE_TABLE = BASE_DIR / "combined_cardiac_wide_table.csv"
EXPANDED_WIDE_TABLE = BASE_DIR / "expanded_cardiac_wide_table.csv"
ORIGINAL_RESULTS = BASE_DIR / "prospective_validation_results_recomputed.json"

RUNS = {
    "combined": {
        "input_table": COMBINED_WIDE_TABLE,
        "validation_dir": BASE_DIR / "expanded_wide_validation",
        "results": BASE_DIR / "prospective_validation_results_expanded_wide.json",
        "comparison_report": BASE_DIR / "expanded_wide_validation_comparison.md",
        "figure_suffix": "expanded_wide",
        "description": "combined expanded-wide table",
    },
    "expanded": {
        "input_table": EXPANDED_WIDE_TABLE,
        "validation_dir": BASE_DIR / "expanded_raw_validation",
        "results": BASE_DIR / "prospective_validation_results_expanded_raw.json",
        "comparison_report": BASE_DIR / "expanded_raw_validation_comparison.md",
        "figure_suffix": "expanded_raw",
        "description": "CSV-derived expanded table only",
    },
}


def load_original_config() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("original_fig_config", ORIGINAL_CONFIG)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {ORIGINAL_CONFIG}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_validation_config(validation_dir: Path) -> None:
    original = load_original_config()
    shim = types.ModuleType("fig_config")
    for name, value in vars(original).items():
        if not name.startswith("__"):
            setattr(shim, name, value)
    shim.OUTPUT_DIR = str(validation_dir)
    shim.FIG_DIR = str(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, name, close=True):
        png_path = validation_dir / f"{name}.png"
        pdf_path = validation_dir / f"{name}.pdf"
        tif_path = validation_dir / f"{name}.tif"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(tif_path, dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
        print(f"  Saved: {png_path.name} / .pdf / .tif")
        if close:
            original.plt.close(fig)

    shim.save_figure = save_figure
    sys.modules["fig_config"] = shim


def metric(results: dict, period: str, comorbidity: str) -> float | None:
    value = (
        results.get("sentinel_analysis", {})
        .get(period, {})
        .get("similarities", {})
        .get(comorbidity)
    )
    return float(value) if value is not None else None


def rank(results: dict, period: str) -> int | None:
    value = results.get("sentinel_analysis", {}).get(period, {}).get("respiratory_rank")
    return int(value) if value is not None else None


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def write_comparison_report(original: dict, expanded: dict, comparison_report: Path, description: str) -> None:
    original_summary = original.get("cohort_summary", {})
    expanded_summary = expanded.get("cohort_summary", {})
    rows = [
        ("Admissions", original_summary.get("total_admissions"), expanded_summary.get("total_admissions")),
        ("Unique 病案号 in validation wide table", original_summary.get("unique_patients"), expanded_summary.get("unique_patients")),
        ("COVID+ admissions", original_summary.get("covid_positive_admissions"), expanded_summary.get("covid_positive_admissions")),
        ("COVID+ patients", original_summary.get("covid_positive_patients"), expanded_summary.get("covid_positive_patients")),
        ("H1 2019 respiratory similarity", metric(original, "H1_2019_prepandemic", "Respiratory"), metric(expanded, "H1_2019_prepandemic", "Respiratory")),
        ("H1 2019 respiratory rank", rank(original, "H1_2019_prepandemic"), rank(expanded, "H1_2019_prepandemic")),
        ("Q4 2022 respiratory similarity", metric(original, "Q4_2022_reopening", "Respiratory"), metric(expanded, "Q4_2022_reopening", "Respiratory")),
        ("Q4 2022 respiratory rank", rank(original, "Q4_2022_reopening"), rank(expanded, "Q4_2022_reopening")),
        ("Permutation p", original.get("permutation_test", {}).get("p_value"), expanded.get("permutation_test", {}).get("p_value")),
    ]

    lines = [
        "# Reconstructed cardiac validation comparison",
        "",
        f"This validation reruns the original prospective validation script on the {description}.",
        "",
        "| Metric | Original 20K run | Reconstructed run |",
        "| --- | ---: | ---: |",
    ]
    for label, original_value, expanded_value in rows:
        if isinstance(original_value, int) or isinstance(expanded_value, int):
            left = f"{int(original_value):,}" if original_value is not None else "NA"
            right = f"{int(expanded_value):,}" if expanded_value is not None else "NA"
        else:
            left = fmt(original_value, 4)
            right = fmt(expanded_value, 4)
        lines.append(f"| {label} | {left} | {right} |")

    rdi_rows = expanded.get("rdi_quarterly", [])
    valid_rdi = [row for row in rdi_rows if row.get("rdi") is not None]
    if valid_rdi:
        peak = max(valid_rdi, key=lambda row: row.get("rdi", float("-inf")))
        lines.extend(
            [
                "",
                f"Expanded-wide RDI peak: {peak.get('label')} (RDI={fmt(peak.get('rdi'), 3)}).",
            ]
        )
    comparison_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_figures(validation_dir: Path, figure_suffix: str) -> None:
    for extension in ["png", "pdf", "tif"]:
        source = validation_dir / f"Figure5_event_driven_sentinel.{extension}"
        if source.exists():
            shutil.copyfile(source, BASE_DIR / f"Figure5_event_driven_sentinel_{figure_suffix}.{extension}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prospective validation on a reconstructed cardiac wide table.")
    parser.add_argument("--mode", choices=sorted(RUNS), default="combined")
    args = parser.parse_args()
    run = RUNS[args.mode]
    input_table = run["input_table"]
    validation_dir = run["validation_dir"]
    validation_input = validation_dir / "final_preprocessed_data_new.csv"
    results_path_out = run["results"]
    comparison_report = run["comparison_report"]

    if not input_table.exists():
        raise FileNotFoundError(f"Run build_expanded_cardiac_wide_table.py first: {input_table}")
    validation_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_table, validation_input)
    install_validation_config(validation_dir)
    runpy.run_path(str(ORIGINAL_PROSPECTIVE_SCRIPT), run_name="__main__")
    result_path = validation_dir / "prospective_validation_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Validation did not produce expected result: {result_path}")
    shutil.copyfile(result_path, results_path_out)
    copy_figures(validation_dir, str(run["figure_suffix"]))
    if ORIGINAL_RESULTS.exists():
        original = json.loads(ORIGINAL_RESULTS.read_text(encoding="utf-8"))
        expanded = json.loads(results_path_out.read_text(encoding="utf-8"))
        write_comparison_report(original, expanded, comparison_report, str(run["description"]))
    print(f"Saved validation results: {results_path_out}")
    print(f"Saved comparison report: {comparison_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())