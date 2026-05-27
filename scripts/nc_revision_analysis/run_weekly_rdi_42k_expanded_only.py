from __future__ import annotations

import json
import math
import csv
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_weekly_rdi_42k import (
    BASE_DIR,
    COMORBIDITIES,
    EVENT_END,
    EVENT_START,
    INPUT_TABLE,
    LEAD_START,
    METRICS,
    OUT_DIR as FULL_OUT_DIR,
    PACKAGE_ANALYSIS,
    PACKAGE_FIGURES,
    SOURCE_GAP_END,
    SOURCE_GAP_START,
    WINDOW_SPECS,
    add_policy_loss_band,
    build_gap_audit,
    build_reference_stats,
    load_and_prepare,
    normalize_mrn,
    performance_for,
    plot_with_policy_break,
    profile_vector,
    summarize_performance,
    write_sample_size_table,
)

OUT_DIR = BASE_DIR / "weekly_rdi_42k_expanded_only_results"
WHU_PRIMARY_TABLE = BASE_DIR.parents[3] / "all_admissions.csv"
EXPANDED_ONLY_TABLE = BASE_DIR / "expanded_cardiac_wide_table_expanded_only.csv"
MISSING_ID_VALUES = {"", "nan", "none", "null", "na", "n/a", "-", "--"}


def normalize_mrn_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in MISSING_ID_VALUES:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def load_whu_primary_mrns() -> set[str]:
    ids: set[str] = set()
    with WHU_PRIMARY_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = normalize_mrn_value(row.get("病案号"))
            if value:
                ids.add(value)
    return ids


def export_expanded_only_wide(primary_mrns: set[str]) -> int:
    wide = pd.read_csv(INPUT_TABLE, low_memory=False, dtype={"病案号": str})
    wide["病案号_norm"] = normalize_mrn(wide["病案号"])
    subset = wide[~wide["病案号_norm"].isin(primary_mrns)].copy()
    subset.drop(columns=["病案号_norm"]).to_csv(EXPANDED_ONLY_TABLE, index=False, encoding="utf-8-sig")
    return int(subset["病案号_norm"].replace("", np.nan).nunique())


def make_windows(data: pd.DataFrame, spec, reference: np.ndarray, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    from run_weekly_rdi_42k import make_windows as _make_windows

    return _make_windows(data, spec, reference, stats)


def plot_expanded_only_figure(frames: dict[str, pd.DataFrame], performance: pd.DataFrame, thresholds_by_window: dict[str, dict[str, float]]) -> None:
    rolling_all = frames["rolling4_weekly"].copy().sort_values("window_start_dt")
    rolling_plot = rolling_all.copy()
    rolling_plot.loc[~rolling_plot["valid"].eq(True), ["rdi", "resp_sim", "mean_other"]] = np.nan
    rolling = rolling_all[rolling_all["valid"].eq(True)].copy()
    primary_threshold = thresholds_by_window["rolling4_weekly"]["rdi_mean_plus_1_5sd"]
    rolling["alert_primary"] = rolling["rdi"] >= primary_threshold

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    ax = axes[0]
    plot_with_policy_break(ax, rolling_plot["window_start_dt"], rolling_plot["rdi"], color="#8e44ad", linewidth=1.2, label="Expanded-only 4-week RDI")
    add_policy_loss_band(ax)
    ax.axhline(primary_threshold, color="#c0392b", linestyle="--", linewidth=1.0, label="Baseline mean + 1.5 SD")
    ax.axvspan(EVENT_START, EVENT_END, color="#DA7C30", alpha=0.16, label="Dec 2022-Jan 2023 event window")
    ax.scatter(rolling.loc[rolling["alert_primary"], "window_start_dt"], rolling.loc[rolling["alert_primary"], "rdi"], s=18, color="#c0392b", zorder=3, label="Alert weeks")
    ax.set_ylabel("RDI")
    ax.set_title("A. Cardio-expanded-only validation subset (WHU-primary records excluded)", fontweight="bold")
    ax.legend(fontsize=8, ncol=3, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)

    ax = axes[1]
    zoom = rolling[rolling["window_start_dt"].between(pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31"))]
    ax.plot(zoom["window_start_dt"], zoom["resp_sim"], color="#DA7C30", linewidth=1.3, label="Respiratory similarity")
    ax.plot(zoom["window_start_dt"], zoom["mean_other"], color="#396AB1", linewidth=1.2, label="Mean non-respiratory similarity")
    ax.axvspan(EVENT_START, EVENT_END, color="#DA7C30", alpha=0.16)
    ax.set_ylabel("Pearson profile correlation")
    ax.set_xlabel("Window start date")
    ax.set_title("B. Post-2021 component trend", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cardio-expanded-only weekly RDI validation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigureS_cardio_expanded_only_weekly_rdi.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sync_to_package() -> None:
    PACKAGE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    PACKAGE_FIGURES.mkdir(parents=True, exist_ok=True)
    for filename in [
        "weekly_rdi_42k_expanded_only_rolling4_weekly.csv",
        "weekly_rdi_42k_expanded_only_raw_weekly.csv",
        "weekly_rdi_42k_expanded_only_monthly.csv",
        "weekly_rdi_42k_expanded_only_performance.csv",
        "weekly_rdi_42k_expanded_only_sample_sizes.csv",
        "weekly_rdi_42k_expanded_only_gap_audit.csv",
        "weekly_rdi_42k_expanded_only_gap_audit.json",
        "weekly_rdi_42k_expanded_only_summary.json",
    ]:
        source = OUT_DIR / filename
        if source.exists():
            shutil.copyfile(source, PACKAGE_ANALYSIS / filename)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_mrns = load_whu_primary_mrns()
    expanded_only_unique = export_expanded_only_wide(primary_mrns)

    data = load_and_prepare()
    full_unique = int(data["病案号_norm"].replace("", np.nan).nunique())
    data = data[~data["病案号_norm"].isin(primary_mrns)].copy()
    stats = build_reference_stats(data)
    covid_reference = data[data["is_covid_positive"].eq(1)]
    if len(covid_reference) < 10:
        raise RuntimeError("Expanded-only COVID reference is too small for weekly RDI analysis")
    reference = profile_vector(covid_reference, stats)

    frames = {spec.name: make_windows(data, spec, reference, stats) for spec in WINDOW_SPECS}
    for name, frame in frames.items():
        export = frame.drop(columns=[col for col in ["window_start_dt", "window_end_dt"] if col in frame.columns])
        export.to_csv(OUT_DIR / f"weekly_rdi_42k_expanded_only_{name}.csv", index=False, encoding="utf-8-sig")

    performance, thresholds_by_window = summarize_performance(frames)
    sample_sizes = write_sample_size_table(frames)
    gap_audit, gap_summary = build_gap_audit(frames)
    performance.to_csv(OUT_DIR / "weekly_rdi_42k_expanded_only_performance.csv", index=False, encoding="utf-8-sig")
    sample_sizes.to_csv(OUT_DIR / "weekly_rdi_42k_expanded_only_sample_sizes.csv", index=False, encoding="utf-8-sig")
    gap_audit.to_csv(OUT_DIR / "weekly_rdi_42k_expanded_only_gap_audit.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "weekly_rdi_42k_expanded_only_gap_audit.json").write_text(json.dumps(gap_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_expanded_only_figure(frames, performance, thresholds_by_window)

    primary = performance[
        performance["window_type"].eq("rolling4_weekly")
        & performance["threshold_rule"].eq("rdi_mean_plus_1_5sd")
    ].iloc[0].to_dict()
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "input_table": str(INPUT_TABLE),
        "whu_primary_table": str(WHU_PRIMARY_TABLE),
        "selection": "Expanded cardiac patient record numbers not present in the WHU primary original cohort all_admissions.csv 病案号 set.",
        "cohort": {
            "full_expanded_unique_patient_record_numbers": full_unique,
            "whu_primary_unique_patient_record_numbers": int(len(primary_mrns)),
            "expanded_only_unique_patient_record_numbers": expanded_only_unique,
            "expanded_only_admissions_with_dates": int(len(data)),
            "date_range": f"{data['admit_dt'].min().date()} to {data['admit_dt'].max().date()}",
            "covid_reference_admissions": int(len(covid_reference)),
            "covid_reference_patients": int(covid_reference["病案号_norm"].replace("", np.nan).nunique()),
            "comorbidity_counts": {group: int(data[group].sum()) for group in COMORBIDITIES},
        },
        "feature_set": {
            "feature_count": len(METRICS),
            "features": METRICS,
            "note": "Same 9-feature cardiac wide-table RDI subset as the full 42k validation; no WHU-primary patient record numbers are included.",
        },
        "baseline": {"start": "2016-01-01", "end": "2018-12-31", "thresholds": thresholds_by_window},
        "event_definition": {"event_start": EVENT_START.date().isoformat(), "event_end": EVENT_END.date().isoformat(), "lead_start": LEAD_START.date().isoformat()},
        "primary_operating_characteristics": primary,
        "source_gap_audit": gap_summary,
        "outputs": {
            "expanded_only_wide_table": str(EXPANDED_ONLY_TABLE),
            "rolling_weekly_csv": str(OUT_DIR / "weekly_rdi_42k_expanded_only_rolling4_weekly.csv"),
            "raw_weekly_csv": str(OUT_DIR / "weekly_rdi_42k_expanded_only_raw_weekly.csv"),
            "monthly_csv": str(OUT_DIR / "weekly_rdi_42k_expanded_only_monthly.csv"),
            "performance_csv": str(OUT_DIR / "weekly_rdi_42k_expanded_only_performance.csv"),
            "figure": str(OUT_DIR / "FigureS_cardio_expanded_only_weekly_rdi.png"),
        },
    }
    (OUT_DIR / "weekly_rdi_42k_expanded_only_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sync_to_package()

    print(json.dumps(summary["cohort"], ensure_ascii=False, indent=2))
    print("Primary expanded-only weekly RDI operating characteristics:")
    print(json.dumps(primary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
