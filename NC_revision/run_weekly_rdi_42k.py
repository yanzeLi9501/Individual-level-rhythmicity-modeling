from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _panel_label(ax, label):
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left", color="black",
            zorder=1000,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      boxstyle="square,pad=0.15"))


BASE_DIR = Path(__file__).resolve().parent
INPUT_TABLE = BASE_DIR / "expanded_cardiac_wide_table.csv"
OUT_DIR = BASE_DIR / "weekly_rdi_42k_results"
PACKAGE_DIR = BASE_DIR / "resubmission_package_20260512"
PACKAGE_FIGURES = PACKAGE_DIR / "figures"
PACKAGE_ANALYSIS = PACKAGE_DIR / "analysis_outputs"
COMBINED_TABLE = BASE_DIR / "combined_cardiac_wide_table.csv"
DEDUP_SUMMARY = BASE_DIR / "cardiac_original_expanded_dedup_summary.json"
COVID_CARDIAC_SUMMARY = BASE_DIR / "external_positive_control_results" / "whu_covid_cardiac_summary.json"
CARMEN_CARDIAC_SUMMARY = BASE_DIR / "external_positive_control_results" / "carmen_i_cardiac_summary.json"

COMORBIDITIES = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
COMORBIDITY_PATTERNS = {
    "Cardiovascular": r"冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病",
    "Hypertension": r"高血压",
    "Diabetes": r"糖尿病|血糖",
    "Cerebrovascular": r"脑梗|脑出血|脑血管|脑卒中|中风|腔隙性",
    "Renal": r"肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏",
    "Respiratory": r"肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染",
}

LAB_COLS_MAP = {
    "白细胞": "lab_WBC",
    "超敏C反应蛋白": "lab_CRP",
    "血红蛋白": "lab_HGB",
    "白蛋白": "lab_ALB",
    "肌酐": "lab_CREA",
    "空腹血糖": "lab_GLU",
    "钾": "lab_K",
    "钠": "lab_Na",
}

METRICS = ["los_days", "lab_WBC", "lab_CRP", "lab_HGB", "lab_ALB", "lab_CREA", "lab_GLU", "lab_K", "lab_Na"]

EVENT_START = pd.Timestamp("2022-12-01")
EVENT_END = pd.Timestamp("2023-01-31")
LEAD_START = EVENT_START - pd.Timedelta(days=28)
MONITOR_START = pd.Timestamp("2019-01-01")
MONITOR_END = pd.Timestamp("2024-12-31")
SOURCE_GAP_START = pd.Timestamp("2019-07-01")
SOURCE_GAP_END = pd.Timestamp("2020-04-26")
POLICY_LOSS_LABEL = "Policy-induced data loss"

COLORS = {
    "primary": "#396AB1",
    "secondary": "#DA7C30",
    "accent": "#3E9651",
    "warn": "#CC2529",
    "text": "#222222",
    "grid": "#DDDDDD",
}


def add_policy_loss_band(ax: plt.Axes, *, label: bool = True) -> None:
    ax.axvspan(
        SOURCE_GAP_START,
        SOURCE_GAP_END,
        color="#7f7f7f",
        alpha=0.18,
        linewidth=0,
        label=POLICY_LOSS_LABEL if label else None,
    )
    midpoint = SOURCE_GAP_START + (SOURCE_GAP_END - SOURCE_GAP_START) / 2
    ax.text(
        midpoint,
        0.52,
        POLICY_LOSS_LABEL,
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=8,
        color="#4d4d4d",
        alpha=0.85,
    )


def plot_with_policy_break(ax: plt.Axes, dates: pd.Series, values: pd.Series, **kwargs) -> None:
    dates = pd.to_datetime(dates)
    values = pd.to_numeric(values, errors="coerce")
    segments = [dates < SOURCE_GAP_START, dates > SOURCE_GAP_END]
    for idx, segment in enumerate(segments):
        plot_kwargs = dict(kwargs)
        if idx > 0:
            plot_kwargs["label"] = None
        ax.plot(dates[segment], values[segment], **plot_kwargs)


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def plot_covid_cardiac_mechanism(ax: plt.Axes) -> None:
    whu = load_json(COVID_CARDIAC_SUMMARY)
    carmen = load_json(CARMEN_CARDIAC_SUMMARY)
    if not whu or not carmen:
        ax.text(0.5, 0.5, "COVID cardiac mechanism\nsummary unavailable", ha="center", va="center")
        ax.axis("off")
        return

    labels = ["De novo", "Exacerbated\npre-existing", "Pre-existing\nonly", "No cardiac"]
    whu_values = [
        float(whu.get("de_novo_covid_cardiac_pct", np.nan)),
        float(whu.get("exacerbation_existing_pct", np.nan)),
        float(whu.get("preexisting_only_pct", np.nan)),
        float(whu.get("no_cardiac_pct", np.nan)),
    ]
    carmen_values = [
        float(carmen.get("de_novo_covid_cardiac_pct", np.nan)),
        float(carmen.get("covid_exacerbation_of_preexisting_pct", np.nan)),
        float(carmen.get("preexisting_only_no_acute_pct", np.nan)),
        float(carmen.get("no_cardiac_pct", np.nan)),
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, whu_values, width, color=COLORS["primary"], label="WHU cardiac COVID patients")
    ax.bar(x + width / 2, carmen_values, width, color=COLORS["secondary"], label="CARMEN-I benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(80, np.nanmax(whu_values + carmen_values) + 10))
    ax.set_ylabel("Percent of COVID patients")
    ax.text(
        0.10,
        0.93,
        f"WHU de novo: {whu_values[0]:.1f}% ({int(whu.get('de_novo_covid_cardiac_n', 0))}/{int(whu.get('n_unique_covid_patients', 0))})",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["text"],
    )
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


@dataclass(frozen=True)
class WindowSpec:
    name: str
    label: str
    unit: str
    span_days: int
    min_total_n: int
    min_group_n: int


WINDOW_SPECS = [
    WindowSpec("raw_weekly", "Raw weekly", "week", 7, 20, 3),
    WindowSpec("rolling4_weekly", "4-week rolling weekly", "week", 28, 50, 10),
    WindowSpec("monthly", "Calendar monthly", "month", 31, 50, 10),
]


def normalize_mrn(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.mask(text.str.lower().isin({"", "nan", "none", "null", "na", "n/a"}), "")


def load_and_prepare() -> pd.DataFrame:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_TABLE}")
    data = pd.read_csv(INPUT_TABLE, low_memory=False, dtype={"病案号": str})
    data["病案号_norm"] = normalize_mrn(data["病案号"])
    data["admit_dt"] = pd.to_datetime(data["入院时间"], errors="coerce")
    data["discharge_dt"] = pd.to_datetime(data["出院时间"], errors="coerce")
    data = data.dropna(subset=["admit_dt"]).copy()
    data["los_days"] = (data["discharge_dt"] - data["admit_dt"]).dt.days
    data.loc[(data["los_days"] < 0) | (data["los_days"] > 180), "los_days"] = np.nan
    data["year"] = data["admit_dt"].dt.year
    data["month"] = data["admit_dt"].dt.month
    data["week_start"] = (data["admit_dt"] - pd.to_timedelta(data["admit_dt"].dt.weekday, unit="D")).dt.normalize()

    diagnosis = data["主要诊断"].fillna("").astype(str)
    previous_diagnosis = data.get("上次诊断", pd.Series("", index=data.index)).fillna("").astype(str)
    all_diagnosis = diagnosis + " " + previous_diagnosis
    for group, pattern in COMORBIDITY_PATTERNS.items():
        data[group] = diagnosis.str.contains(pattern, case=False, regex=True, na=False).astype(int)

    covid_pattern = r"新型冠状病毒|冠状病毒感染|冠状病毒肺炎"
    data["is_covid_positive"] = all_diagnosis.str.contains(covid_pattern, case=False, regex=True, na=False).astype(int)

    for source_col, target_col in LAB_COLS_MAP.items():
        data[target_col] = pd.to_numeric(data[source_col], errors="coerce") if source_col in data.columns else np.nan
    return data


def coalesced_datetime(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    values = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for col in candidates:
        if col in frame.columns:
            values = values.fillna(pd.to_datetime(frame[col], errors="coerce"))
    return values


def audit_gap_table(label: str, path: Path) -> dict[str, object]:
    row: dict[str, object] = {"source": label, "path": str(path), "available": path.exists()}
    if not path.exists():
        return row
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    columns = [col for col in ["病案号", "入院时间", "入院日期", "出院时间", "出院日期", "主要诊断", "上次诊断"] if col in header.columns]
    dtype = {"病案号": str} if "病案号" in columns else None
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig", low_memory=False, dtype=dtype)
    if "病案号" in frame.columns:
        frame["病案号_norm"] = normalize_mrn(frame["病案号"])
    else:
        frame["病案号_norm"] = ""
    frame["admit_dt"] = coalesced_datetime(frame, ["入院时间", "入院日期"])
    dated = frame.dropna(subset=["admit_dt"]).copy()
    gap = dated[dated["admit_dt"].between(SOURCE_GAP_START, SOURCE_GAP_END)]
    row.update(
        {
            "rows_with_admit_date": int(len(dated)),
            "unique_patient_record_numbers": int(dated["病案号_norm"].replace("", np.nan).nunique()),
            "date_min": None if dated.empty else dated["admit_dt"].min().date().isoformat(),
            "date_max": None if dated.empty else dated["admit_dt"].max().date().isoformat(),
            "rows_in_2019_07_01_to_2020_04_26_gap": int(len(gap)),
            "unique_patient_record_numbers_in_gap": int(gap["病案号_norm"].replace("", np.nan).nunique()),
        }
    )
    return row


def zero_admission_runs(frame: pd.DataFrame, min_weeks: int = 4) -> list[dict[str, object]]:
    ordered = frame.sort_values("window_start_dt").copy()
    runs: list[dict[str, object]] = []
    current: list[pd.Series] = []
    for _, row in ordered.iterrows():
        if int(row.get("n_admissions", 0)) == 0:
            current.append(row)
            continue
        if len(current) >= min_weeks:
            runs.append(
                {
                    "first_label": str(current[0]["label"]),
                    "last_label": str(current[-1]["label"]),
                    "window_start": pd.Timestamp(current[0]["window_start_dt"]).date().isoformat(),
                    "window_end": pd.Timestamp(current[-1]["window_end_dt"]).date().isoformat(),
                    "weeks": int(len(current)),
                }
            )
        current = []
    if len(current) >= min_weeks:
        runs.append(
            {
                "first_label": str(current[0]["label"]),
                "last_label": str(current[-1]["label"]),
                "window_start": pd.Timestamp(current[0]["window_start_dt"]).date().isoformat(),
                "window_end": pd.Timestamp(current[-1]["window_end_dt"]).date().isoformat(),
                "weeks": int(len(current)),
            }
        )
    return runs


def build_gap_audit(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    comparison_paths: dict[str, Path] = {
        "current_expanded_wide_table": INPUT_TABLE,
        "combined_old20k_plus_expanded_wide_table": COMBINED_TABLE,
    }
    if DEDUP_SUMMARY.exists():
        try:
            summary = json.loads(DEDUP_SUMMARY.read_text(encoding="utf-8"))
            original_path = Path(str(summary.get("original_cardiac_csv", {}).get("path", "")))
            if original_path.exists():
                comparison_paths["original_20k_cardiac_file"] = original_path
        except Exception:
            pass
    rows = [audit_gap_table(label, path) for label, path in comparison_paths.items()]
    audit_df = pd.DataFrame(rows)
    rolling = frames["rolling4_weekly"].copy()
    valid = rolling[rolling["valid"].eq(True)].sort_values("window_start_dt").copy()
    valid_gaps = valid["window_start_dt"].diff().dt.days
    source_gap = {
        "audit_gap_start": SOURCE_GAP_START.date().isoformat(),
        "audit_gap_end": SOURCE_GAP_END.date().isoformat(),
        "rolling_zero_admission_runs_min_4_weeks": zero_admission_runs(rolling, min_weeks=4),
        "longest_gap_between_plotted_valid_points_days": None if valid_gaps.dropna().empty else int(valid_gaps.max()),
        "old_20k_can_fill_audit_gap": bool((audit_df.get("rows_in_2019_07_01_to_2020_04_26_gap", pd.Series(dtype=float)).fillna(0) > 0).any()),
        "interpretation": "The long visual slope in panel A is a plotting artifact caused by connecting valid weekly RDI points across a prolonged no-admission interval in the cardiac source tables. The original 20K cardiac file and the combined old-plus-expanded table do not provide admissions in the audited 2019-07-01 to 2020-04-26 interval when available, so 病案号-based supplementation cannot fill this gap.",
    }
    return audit_df, source_gap


def build_reference_stats(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline = data[(data["year"] >= 2016) & (data["year"] <= 2018)]
    stats: dict[str, dict[str, float]] = {}
    for col in METRICS:
        values = pd.to_numeric(baseline[col], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 1.0
        if not np.isfinite(std) or std <= 1e-9:
            std = 1.0
        stats[col] = {"mean": mean, "std": std, "n": int(len(values))}
    return stats


def profile_vector(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> np.ndarray:
    values: list[float] = []
    for col in METRICS:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(series):
            values.append((float(series.mean()) - stats[col]["mean"]) / stats[col]["std"])
        else:
            values.append(0.0)
    return np.nan_to_num(np.array(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def pearson_profile_correlation(reference: np.ndarray, vector: np.ndarray) -> float:
    if np.linalg.norm(reference) <= 0 or np.linalg.norm(vector) <= 0:
        return math.nan
    return float(1 - cosine(reference, vector))


def compute_group_sims(frame: pd.DataFrame, reference: np.ndarray, stats: dict[str, dict[str, float]], min_group_n: int) -> tuple[dict[str, float], dict[str, int]]:
    sims: dict[str, float] = {}
    counts: dict[str, int] = {}
    for group in COMORBIDITIES:
        subset = frame[frame[group].eq(1)]
        counts[group] = int(len(subset))
        sims[group] = pearson_profile_correlation(reference, profile_vector(subset, stats)) if len(subset) >= min_group_n else math.nan
    return sims, counts


def row_from_window(label: str, start: pd.Timestamp, end: pd.Timestamp, frame: pd.DataFrame, spec: WindowSpec, reference: np.ndarray, stats: dict[str, dict[str, float]]) -> dict[str, object]:
    if len(frame) < spec.min_total_n:
        return {"window_type": spec.name, "label": label, "window_start": start.date().isoformat(), "window_end": end.date().isoformat(), "n_admissions": int(len(frame)), "valid": False}
    monitoring_frame = frame[frame["is_covid_positive"].eq(0)]
    sims, counts = compute_group_sims(monitoring_frame, reference, stats, spec.min_group_n)
    resp_sim = sims.get("Respiratory", math.nan)
    other = [value for group, value in sims.items() if group != "Respiratory" and np.isfinite(value)]
    mean_other = float(np.mean(other)) if other else math.nan
    rdi = resp_sim - mean_other if np.isfinite(resp_sim) and np.isfinite(mean_other) else math.nan
    row: dict[str, object] = {
        "window_type": spec.name,
        "label": label,
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "n_admissions": int(len(frame)),
        "n_admissions_excluding_covid_reference": int(len(monitoring_frame)),
        "valid": bool(np.isfinite(rdi)),
        "resp_sim": resp_sim,
        "mean_other": mean_other,
        "rdi": rdi,
        "event_window": bool(start <= EVENT_END and end >= EVENT_START),
        "lead_or_event_window": bool(start <= EVENT_END and end >= LEAD_START),
    }
    for group in COMORBIDITIES:
        row[f"sim_{group}"] = sims.get(group, math.nan)
        row[f"n_{group}"] = counts.get(group, 0)
    return row


def make_windows(data: pd.DataFrame, spec: WindowSpec, reference: np.ndarray, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if spec.unit == "week":
        first = pd.Timestamp("2016-01-04")
        last = data["week_start"].max()
        for week_start in pd.date_range(first, last, freq="W-MON"):
            start = week_start if spec.span_days == 7 else week_start - pd.Timedelta(days=spec.span_days - 7)
            end = week_start + pd.Timedelta(days=6)
            frame = data[(data["admit_dt"] >= start) & (data["admit_dt"] <= end)]
            rows.append(row_from_window(f"{week_start.isocalendar().year}W{week_start.isocalendar().week:02d}", start, end, frame, spec, reference, stats))
    else:
        for month_start in pd.date_range("2016-01-01", data["admit_dt"].max().replace(day=1), freq="MS"):
            end = month_start + pd.offsets.MonthEnd(1)
            frame = data[(data["admit_dt"] >= month_start) & (data["admit_dt"] <= end)]
            rows.append(row_from_window(month_start.strftime("%Y-%m"), month_start, pd.Timestamp(end), frame, spec, reference, stats))
    out = pd.DataFrame(rows)
    out["window_start_dt"] = pd.to_datetime(out["window_start"])
    out["window_end_dt"] = pd.to_datetime(out["window_end"])
    return out


def thresholds(frame: pd.DataFrame) -> dict[str, float]:
    baseline = frame[
        frame["valid"].eq(True)
        & frame["window_start_dt"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31"))
    ]["rdi"].dropna()
    return {
        "rdi_mean_plus_1_5sd": float(baseline.mean() + 1.5 * baseline.std()),
        "rdi_mean_plus_2sd": float(baseline.mean() + 2.0 * baseline.std()),
        "rdi_p97_5": float(np.percentile(baseline, 97.5)),
    }


def performance_for(frame: pd.DataFrame, threshold_name: str, threshold_value: float) -> dict[str, object]:
    monitor = frame[
        frame["valid"].eq(True)
        & frame["window_start_dt"].between(MONITOR_START, MONITOR_END)
    ].copy()
    monitor["alert"] = monitor["rdi"] >= threshold_value
    event = monitor["event_window"].astype(bool)
    lead_or_event = monitor["lead_or_event_window"].astype(bool)
    alert = monitor["alert"].astype(bool)

    tp_strict = int((alert & event).sum())
    tp_lead = int((alert & lead_or_event).sum())
    fp_strict = int((alert & ~event).sum())
    fp_lead = int((alert & ~lead_or_event).sum())
    fn = int((~alert & event).sum())
    tn_strict = int((~alert & ~event).sum())
    tn_lead = int((~alert & ~lead_or_event).sum())

    def safe_div(num: int, den: int) -> float:
        return float(num / den) if den else math.nan

    event_alerts = monitor[alert & (monitor["window_start_dt"] >= LEAD_START) & (monitor["window_start_dt"] <= EVENT_END)]
    if len(event_alerts):
        first_alert = pd.Timestamp(event_alerts["window_start_dt"].min())
        lead_time_days = int((EVENT_START - first_alert).days)
    else:
        first_alert = pd.NaT
        lead_time_days = None

    return {
        "window_type": str(monitor["window_type"].iloc[0]) if len(monitor) else frame["window_type"].iloc[0],
        "threshold_rule": threshold_name,
        "threshold_value": threshold_value,
        "monitor_windows": int(len(monitor)),
        "event_windows": int(event.sum()),
        "alert_windows": int(alert.sum()),
        "true_positive_event_windows": tp_strict,
        "true_positive_lead_or_event_windows": tp_lead,
        "false_positive_event_strict": fp_strict,
        "false_positive_lead_allowed": fp_lead,
        "false_negative_event_windows": fn,
        "true_negative_event_strict": tn_strict,
        "true_negative_lead_allowed": tn_lead,
        "sensitivity_event_week": safe_div(tp_strict, tp_strict + fn),
        "precision_ppv_event_strict": safe_div(tp_strict, tp_strict + fp_strict),
        "precision_ppv_lead_allowed": safe_div(tp_lead, tp_lead + fp_lead),
        "false_alarm_rate_event_strict": safe_div(fp_strict, fp_strict + tn_strict),
        "false_alarm_rate_lead_allowed": safe_div(fp_lead, fp_lead + tn_lead),
        "first_alert_in_lead_or_event": None if pd.isna(first_alert) else first_alert.date().isoformat(),
        "lead_time_days": lead_time_days,
    }


def summarize_performance(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    perf_rows: list[dict[str, object]] = []
    thresh_by_window: dict[str, dict[str, float]] = {}
    for name, frame in frames.items():
        thresh = thresholds(frame)
        thresh_by_window[name] = thresh
        for rule, value in thresh.items():
            perf_rows.append(performance_for(frame, rule, value))
    return pd.DataFrame(perf_rows), thresh_by_window


def write_sample_size_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in frames.items():
        valid = frame[frame["valid"].eq(True)].copy()
        if valid.empty:
            continue
        row = {
            "window_type": name,
            "valid_windows": int(len(valid)),
            "median_admissions": float(valid["n_admissions"].median()),
            "p10_admissions": float(valid["n_admissions"].quantile(0.10)),
            "p90_admissions": float(valid["n_admissions"].quantile(0.90)),
        }
        for group in COMORBIDITIES:
            row[f"median_n_{group}"] = float(valid[f"n_{group}"].median())
            row[f"p10_n_{group}"] = float(valid[f"n_{group}"].quantile(0.10))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_figure(frames: dict[str, pd.DataFrame], performance: pd.DataFrame, thresholds_by_window: dict[str, dict[str, float]]) -> None:
    rolling_all = frames["rolling4_weekly"].copy().sort_values("window_start_dt")
    rolling_plot = rolling_all.copy()
    rolling_plot.loc[~rolling_plot["valid"].eq(True), ["rdi", "resp_sim", "mean_other"]] = np.nan
    rolling = rolling_all[rolling_all["valid"].eq(True)].copy()
    monthly = frames["monthly"].copy()
    primary_threshold = thresholds_by_window["rolling4_weekly"]["rdi_mean_plus_1_5sd"]
    rolling["alert_primary"] = rolling["rdi"] >= primary_threshold
    zoom = rolling[rolling["window_start_dt"].between(pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31"))]

    fig = plt.figure(figsize=(16, 9.8))
    grid = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34, left=0.07, right=0.98, top=0.97, bottom=0.10)

    ax_a = fig.add_subplot(grid[0, :])
    plot_with_policy_break(ax_a, rolling_plot["window_start_dt"], rolling_plot["rdi"], color=COLORS["primary"], linewidth=1.25, label="4-week rolling weekly RDI")
    add_policy_loss_band(ax_a)
    ax_a.axhline(primary_threshold, color=COLORS["warn"], linestyle="--", linewidth=1.0, label="Baseline mean + 1.5 SD")
    ax_a.axvspan(EVENT_START, EVENT_END, color=COLORS["secondary"], alpha=0.16, label="Dec 2022-Jan 2023 event window")
    ax_a.scatter(rolling.loc[rolling["alert_primary"], "window_start_dt"], rolling.loc[rolling["alert_primary"], "rdi"], s=18, color=COLORS["warn"], zorder=3, label="Alert weeks")
    ax_a.set_ylabel("Respiratory Dominance Index")
    _panel_label(ax_a, "A")
    ax_a.set_title("Weekly RDI time series with COVID alert thresholds (2018–2024)", fontsize=9.5)
    ax_a.legend(fontsize=7, ncol=3, framealpha=0.9, loc="upper right")
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

    ax_b = fig.add_subplot(grid[1, 0])
    ax_b.plot(zoom["window_start_dt"], zoom["resp_sim"], color=COLORS["secondary"], linewidth=1.4, label="Respiratory similarity")
    ax_b.plot(zoom["window_start_dt"], zoom["mean_other"], color=COLORS["primary"], linewidth=1.2, label="Mean non-respiratory similarity")
    ax_b.axvspan(EVENT_START, EVENT_END, color=COLORS["secondary"], alpha=0.16)
    ax_b.set_ylabel("Pearson profile correlation")
    _panel_label(ax_b, "B")
    ax_b.set_title("Respiratory vs non-respiratory Pearson similarity (2021–2024)", fontsize=9.5)
    ax_b.legend(fontsize=7, framealpha=0.9)
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

    ax_c = fig.add_subplot(grid[1, 1])
    perf = performance[performance["window_type"].eq("rolling4_weekly")].copy()
    order = ["rdi_mean_plus_1_5sd", "rdi_mean_plus_2sd", "rdi_p97_5"]
    perf = perf.set_index("threshold_rule").loc[order].reset_index()
    x = np.arange(len(perf))
    width = 0.25
    ax_c.bar(x - width, perf["sensitivity_event_week"], width, color=COLORS["secondary"], label="Sensitivity")
    ax_c.bar(x, perf["precision_ppv_lead_allowed"], width, color=COLORS["primary"], label="PPV (4-week lead allowed)")
    ax_c.bar(x + width, perf["false_alarm_rate_lead_allowed"], width, color=COLORS["warn"], label="False-alarm rate")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Mean+1.5SD", "Mean+2SD", "P97.5"], rotation=20, ha="right")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("Metric value")
    _panel_label(ax_c, "C")
    ax_c.set_title("Detection performance across RDI alert thresholds", fontsize=9.5)
    ax_c.legend(fontsize=7, framealpha=0.9)
    ax_c.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

    ax_d = fig.add_subplot(grid[1, 2])
    plot_covid_cardiac_mechanism(ax_d)
    ax_d.set_title("Post-COVID cardiac admission mechanism overview", fontsize=9.5)

    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigureS14_weekly_rdi_42k.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "FigureS14_weekly_rdi_42k.tif", dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def sync_to_package() -> None:
    PACKAGE_FIGURES.mkdir(parents=True, exist_ok=True)
    PACKAGE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf", "svg", "tif"]:
        shutil.copyfile(OUT_DIR / f"FigureS14_weekly_rdi_42k.{extension}", PACKAGE_FIGURES / f"FigureS14_weekly_rdi_42k.{extension}")
    for filename in [
        "weekly_rdi_42k_rolling4_weekly.csv",
        "weekly_rdi_42k_raw_weekly.csv",
        "weekly_rdi_42k_monthly.csv",
        "weekly_rdi_42k_performance.csv",
        "weekly_rdi_42k_sample_sizes.csv",
        "weekly_rdi_42k_gap_audit.csv",
        "weekly_rdi_42k_gap_audit.json",
        "weekly_rdi_42k_summary.json",
    ]:
        shutil.copyfile(OUT_DIR / filename, PACKAGE_ANALYSIS / filename)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare()
    stats = build_reference_stats(data)
    covid_reference = data[data["is_covid_positive"].eq(1)]
    if len(covid_reference) < 10:
        raise RuntimeError("COVID reference is too small for weekly RDI analysis")
    reference = profile_vector(covid_reference, stats)

    frames = {spec.name: make_windows(data, spec, reference, stats) for spec in WINDOW_SPECS}
    for name, frame in frames.items():
        export = frame.drop(columns=[col for col in ["window_start_dt", "window_end_dt"] if col in frame.columns])
        export.to_csv(OUT_DIR / f"weekly_rdi_42k_{name}.csv", index=False, encoding="utf-8-sig")

    performance, thresholds_by_window = summarize_performance(frames)
    sample_sizes = write_sample_size_table(frames)
    gap_audit, gap_summary = build_gap_audit(frames)
    performance.to_csv(OUT_DIR / "weekly_rdi_42k_performance.csv", index=False, encoding="utf-8-sig")
    sample_sizes.to_csv(OUT_DIR / "weekly_rdi_42k_sample_sizes.csv", index=False, encoding="utf-8-sig")
    gap_audit.to_csv(OUT_DIR / "weekly_rdi_42k_gap_audit.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "weekly_rdi_42k_gap_audit.json").write_text(json.dumps(gap_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_figure(frames, performance, thresholds_by_window)

    primary = performance[
        performance["window_type"].eq("rolling4_weekly")
        & performance["threshold_rule"].eq("rdi_mean_plus_1_5sd")
    ].iloc[0].to_dict()
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "input_table": str(INPUT_TABLE),
        "cohort": {
            "admissions_with_dates": int(len(data)),
            "unique_patient_record_numbers": int(data["病案号_norm"].replace("", np.nan).nunique()),
            "date_range": f"{data['admit_dt'].min().date()} to {data['admit_dt'].max().date()}",
            "covid_reference_admissions": int(len(covid_reference)),
            "covid_reference_patients": int(covid_reference["病案号_norm"].replace("", np.nan).nunique()),
            "comorbidity_counts": {group: int(data[group].sum()) for group in COMORBIDITIES},
        },
        "feature_set": {
            "feature_count": len(METRICS),
            "features": METRICS,
            "note": "Weekly RDI uses the 9-feature cardiac wide-table subset available in the corrected 42k validation table (LOS plus eight laboratory measures).",
        },
        "baseline": {"start": "2016-01-01", "end": "2018-12-31", "thresholds": thresholds_by_window},
        "event_definition": {"event_start": EVENT_START.date().isoformat(), "event_end": EVENT_END.date().isoformat(), "lead_start": LEAD_START.date().isoformat()},
        "primary_operating_characteristics": primary,
        "source_gap_audit": gap_summary,
        "outputs": {
            "rolling_weekly_csv": str(OUT_DIR / "weekly_rdi_42k_rolling4_weekly.csv"),
            "raw_weekly_csv": str(OUT_DIR / "weekly_rdi_42k_raw_weekly.csv"),
            "monthly_csv": str(OUT_DIR / "weekly_rdi_42k_monthly.csv"),
            "performance_csv": str(OUT_DIR / "weekly_rdi_42k_performance.csv"),
            "sample_sizes_csv": str(OUT_DIR / "weekly_rdi_42k_sample_sizes.csv"),
            "gap_audit_csv": str(OUT_DIR / "weekly_rdi_42k_gap_audit.csv"),
            "gap_audit_json": str(OUT_DIR / "weekly_rdi_42k_gap_audit.json"),
            "figure": str(OUT_DIR / "FigureS14_weekly_rdi_42k.png"),
        },
    }
    (OUT_DIR / "weekly_rdi_42k_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sync_to_package()

    print(json.dumps(summary["cohort"], ensure_ascii=False, indent=2))
    print("Primary weekly RDI operating characteristics:")
    print(json.dumps(primary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())