from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_DIR = SCRIPT_DIR.parent
NC_DIR = REBUILD_DIR.parent
SUBMIT_DIR = NC_DIR.parent

FIGURES_DIR = REBUILD_DIR / "figures"
SUPP_FIGURES_DIR = REBUILD_DIR / "supp_figures"
TABLES_DIR = REBUILD_DIR / "tables"
OUTPUTS_DIR = REBUILD_DIR / "outputs"
LOGS_DIR = REBUILD_DIR / "logs"
MERGED_FIGURES_DIR = OUTPUTS_DIR / "figures" / "merged"

LGDI_DIR = NC_DIR / "lgdi_results"
XGB_VISIT_DIR = NC_DIR / "xgboost_visit_order_results"
GPU_XGB_DIR = OUTPUTS_DIR / "gpu_native_xgb"
ANALYSIS_OUTPUTS_DIR = OUTPUTS_DIR / "analysis_outputs"

FORMATS = ("png", "pdf", "svg", "tif")

# Spring/summer palette (§1.5 of merge_figures.md)
# Old keys are preserved so downstream scripts need no edits.
COLORS = {
    # Primary spring/summer roles (replaces old deep-color scheme)
    "blue":        "#8ecae6",   # spring blue  — main lines, bars, flowchart frames
    "orange":      "#f6bd60",   # amber        — secondary bars, season windows
    "green":       "#90be6d",   # mint         — validation / positive-control
    "red":         "#f28482",   # coral        — alert / warning / respiratory
    "purple":      "#b39ddb",   # lavender     — external datasets, third group
    "teal":        "#76c7c0",   # sea teal     — LGDI scalar, auxiliary lines
    "gray":        "#8b95a7",   # warm gray    — baseline, reference, grid helpers
    "light_gray":  "#e8eaec",
    "dark":        "#111827",   # kept for text / labels
    "gold":        "#f6bd60",   # alias → amber
    # Semantic aliases for new code
    "spring_blue": "#8ecae6",
    "coral":       "#f28482",
    "amber":       "#f6bd60",
    "mint":        "#90be6d",
    "lavender":    "#b39ddb",
    "sea_teal":    "#76c7c0",
    "warm_gray":   "#8b95a7",
    "cream":       "#fdecc8",   # season / disruption window fill
}

# (fill_color, alpha) tuples for axvspan / fill_between shading
SHADING = {
    "cream":   ("#fdecc8", 0.22),  # season window default
    "blue":    ("#8ecae6", 0.20),  # calibration phase
    "orange":  ("#f6bd60", 0.20),  # validation phase
    "red":     ("#f28482", 0.18),  # test / alert phase
    "mint":    ("#90be6d", 0.16),  # FluNet event weeks
    "lavender":("#b39ddb", 0.18),  # external-dataset windows
}


def ensure_dirs() -> None:
    for directory in (FIGURES_DIR, SUPP_FIGURES_DIR, TABLES_DIR, OUTPUTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("")
    for fmt in FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)


def save_merged(fig: plt.Figure, name: str) -> None:
    """Save a merged/combined figure to the merged output directory."""
    save_figure(fig, MERGED_FIGURES_DIR / name)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["dark"],
    )


def short_label(text: str) -> str:
    mapping = {
        "REFERENCE_resp_mean_plus_1_5sd": "S1 single respiratory",
        "consensus_2groups_mean_plus_1_5sd": "S2 consensus >=2",
        "consensus_2groups_season_oct_apr": "S3 season consensus",
        "season_sustained_consensus2grp_nov_mar": "S4 sustained Nov-Mar",
        "REFERENCE_resp_mean_plus_2sd": "Resp +2SD",
        "LGDI_mean_plus_1_5sd": "LGDI scalar",
        "REFERENCE_resp_mean_plus_1sd_relaxed": "Resp +1SD",
        "mimic_iv_native_features": "MIMIC-IV",
        "eicu_native_features": "eICU",
        "nwicu_native_features": "NWICU",
        "resp_score": "Respiratory residual score",
        "lgdi": "LGDI scalar",
    }
    return mapping.get(text, text.replace("_", " "))


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):.1f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.astype(object).values.tolist()]
    widths = [len(str(col)) for col in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def line(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    output = [line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    output.extend(line(row) for row in rows)
    return "\n".join(output) + "\n"


def write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown_table(frame), encoding="utf-8")


def add_note(ax: plt.Axes, text: str) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.92,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["dark"],
        linespacing=1.35,
    )


def parse_dates(frame: pd.DataFrame, column: str = "week_start") -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column])
    return frame


def safe_auc(curve: pd.DataFrame) -> float:
    ordered = curve.sort_values("fpr")
    return float(np.trapezoid(ordered["tpr"], ordered["fpr"]))