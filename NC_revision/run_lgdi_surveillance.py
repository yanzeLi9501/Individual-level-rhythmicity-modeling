#!/usr/bin/env python3
"""LOS-Gap Deviation Index surveillance analysis for the NC revision."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

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
SUBMIT_DIR = BASE_DIR.parent
INPUT_TABLE = BASE_DIR / "expanded_cardiac_wide_table.csv"
OUT_DIR = BASE_DIR / "lgdi_results"
PACKAGE_DIR = BASE_DIR / "resubmission_package_20260512"
PACKAGE_FIGURES = PACKAGE_DIR / "figures"
PACKAGE_ANALYSIS = PACKAGE_DIR / "analysis_outputs"
NWICU_ROOT = SUBMIT_DIR / "external_data" / "physionet" / "nwicu-northwestern-icu" / "0.1.0"
FLUNET_CHINA = SUBMIT_DIR / "external_data" / "flunet" / "flunet_china_2009_2024.csv"

COMORBIDITIES = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
RESPIRATORY = "Respiratory"
COMORBIDITY_PATTERNS = {
    "Cardiovascular": r"冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病",
    "Hypertension": r"高血压",
    "Diabetes": r"糖尿病|血糖",
    "Cerebrovascular": r"脑梗|脑出血|脑血管|脑卒中|中风|腔隙性",
    "Renal": r"肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏",
    "Respiratory": r"肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染|呼吸道感染",
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
EXPECTED_COLUMNS = ["病案号", "入院时间", "出院时间", "入院日期", "出院日期", "主要诊断", "白细胞", "超敏C反应蛋白"]
OUTCOMES = ["los_days", "gap_days"]

EVENT_START = pd.Timestamp("2022-12-01")
EVENT_END = pd.Timestamp("2023-01-31")
LEAD_START = EVENT_START - pd.Timedelta(days=28)
MONITOR_START = pd.Timestamp("2019-01-01")
MONITOR_END = pd.Timestamp("2024-12-31")
POLICY_GAP_START = pd.Timestamp("2019-07-01")
POLICY_GAP_END = pd.Timestamp("2020-04-26")
POLICY_GAP_LABEL = "Policy-induced data loss"

COLORS = {
    "primary": "#396AB1",
    "secondary": "#DA7C30",
    "accent": "#3E9651",
    "warn": "#CC2529",
    "text": "#222222",
    "grid": "#DDDDDD",
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


def resolve_column(frame: pd.DataFrame, candidates: list[str], position: int | None = None, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if position is not None and 0 <= position < len(frame.columns):
        return str(frame.columns[position])
    if required:
        raise KeyError(f"None of {candidates} found in columns")
    return None


def compute_los_gap(frame: pd.DataFrame, mrn_col: str, admit_col: str, discharge_col: str) -> pd.DataFrame:
    data = frame.copy()
    data["病案号_norm"] = normalize_mrn(data[mrn_col])
    data["admit_dt"] = pd.to_datetime(data[admit_col], errors="coerce")
    data["discharge_dt"] = pd.to_datetime(data[discharge_col], errors="coerce")
    data = data.dropna(subset=["admit_dt"]).copy()
    data["los_days"] = (data["discharge_dt"] - data["admit_dt"]).dt.total_seconds() / 86400
    data.loc[(data["los_days"] < 0) | (data["los_days"] > 180), "los_days"] = np.nan
    data = data.sort_values(["病案号_norm", "admit_dt", "discharge_dt"]).copy()
    data["prev_discharge_dt"] = data.groupby("病案号_norm", dropna=False)["discharge_dt"].shift(1)
    data["gap_days"] = (data["admit_dt"] - data["prev_discharge_dt"]).dt.total_seconds() / 86400
    data.loc[(data["gap_days"] < 0) | (data["gap_days"] > 3650), "gap_days"] = np.nan
    data["year"] = data["admit_dt"].dt.year
    data["week_start"] = (data["admit_dt"] - pd.to_timedelta(data["admit_dt"].dt.weekday, unit="D")).dt.normalize()
    return data


def prepare_cardiac_table(path: Path) -> pd.DataFrame:
    raw = read_csv_best(path)
    mrn_col = resolve_column(raw, ["病案号", "mrn", "patient_id", "病案号_norm"], position=0)
    admit_col = resolve_column(raw, ["入院时间", "入院日期", "admit_dt", "admit_time", "admission_date"], position=1)
    discharge_col = resolve_column(raw, ["出院时间", "出院日期", "discharge_dt", "discharge_time"], position=2)
    diagnosis_col = resolve_column(raw, ["主要诊断", "诊断文本", "diagnosis_text", "EMR_初步诊断", "EMR_出院记录"], position=3, required=False)
    previous_col = resolve_column(raw, ["上次诊断", "previous_diagnosis"], position=6, required=False)

    data = compute_los_gap(raw, str(mrn_col), str(admit_col), str(discharge_col))
    diagnosis = data[diagnosis_col].fillna("").astype(str) if diagnosis_col else pd.Series("", index=data.index)
    previous = data[previous_col].fillna("").astype(str) if previous_col and previous_col in data.columns else pd.Series("", index=data.index)
    all_diagnosis = diagnosis + " " + previous
    data["diagnosis_text"] = all_diagnosis
    for group, pattern in COMORBIDITY_PATTERNS.items():
        data[group] = all_diagnosis.str.contains(pattern, case=False, regex=True, na=False).astype(bool)
    covid_pattern = r"新型冠状病毒|冠状病毒感染|冠状病毒肺炎|COVID|U07"
    data["is_covid_positive"] = all_diagnosis.str.contains(covid_pattern, case=False, regex=True, na=False).astype(bool)
    for source_col, target_col in LAB_COLS_MAP.items():
        if source_col in data.columns:
            data[target_col] = pd.to_numeric(data[source_col], errors="coerce")
        elif target_col in data.columns:
            data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
        else:
            data[target_col] = np.nan
    return data


RANDOM_STATE = 20260513

UTILIZATION_COLUMNS = ["检验项目数", "医嘱数量", "检查数量"]


def _seasonal_features(admit_dt: pd.Series) -> pd.DataFrame:
    month = admit_dt.dt.month
    dow = admit_dt.dt.weekday
    doy = admit_dt.dt.dayofyear
    return pd.DataFrame(
        {
            "year": admit_dt.dt.year.astype("float64"),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=admit_dt.index,
    )


def build_admission_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build leak-free per-admission features and next_los/next_gap targets."""
    work = data.sort_values(["病案号_norm", "admit_dt", "discharge_dt"]).copy()
    grouped = work.groupby("病案号_norm", dropna=False, sort=False)
    work["visit_order"] = grouped.cumcount()
    work["first_admit_dt"] = grouped["admit_dt"].transform("min")
    work["days_since_first_admit"] = (work["admit_dt"] - work["first_admit_dt"]).dt.total_seconds() / 86400
    work["prior_los_mean"] = grouped["los_days"].transform(lambda s: s.shift(1).expanding().mean())
    work["prior_los_std"] = grouped["los_days"].transform(lambda s: s.shift(1).expanding().std())
    work["prior_los_last"] = grouped["los_days"].shift(1)
    work["prior_gap_mean"] = grouped["gap_days"].transform(lambda s: s.shift(1).expanding().mean())
    work["prior_gap_std"] = grouped["gap_days"].transform(lambda s: s.shift(1).expanding().std())
    work["prior_gap_last"] = grouped["gap_days"].shift(1)
    work["next_admit_dt"] = grouped["admit_dt"].shift(-1)
    work["next_los_days"] = grouped["los_days"].shift(-1)
    work["next_gap_days"] = (work["next_admit_dt"] - work["discharge_dt"]).dt.total_seconds() / 86400
    work.loc[(work["next_gap_days"] < 0) | (work["next_gap_days"] > 3650), "next_gap_days"] = np.nan

    seasonal = _seasonal_features(work["admit_dt"])
    for col in seasonal.columns:
        work[col] = seasonal[col].values

    feature_cols: list[str] = [
        "visit_order",
        "days_since_first_admit",
        "prior_los_mean",
        "prior_los_std",
        "prior_los_last",
        "prior_gap_mean",
        "prior_gap_std",
        "prior_gap_last",
        "year",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
    ]
    feature_cols.extend(COMORBIDITIES)
    for col in LAB_COLS_MAP.values():
        if col in work.columns:
            feature_cols.append(col)
    for col in UTILIZATION_COLUMNS:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            feature_cols.append(col)
    for col in feature_cols:
        if col in COMORBIDITIES:
            work[col] = work[col].astype(bool).astype("float64")
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce").astype("float64")
    return work, feature_cols


def _xgb_regressor() -> "xgb.XGBRegressor":
    # GPU-server tuned hyperparameters (gap_tuned in comprehensive_reanalysis.json,
    # 2026-05-18 GPU re-tuning, gap R²=0.860 / MAE=3.33d on n=2478 test set).
    # device='cuda' falls back to CPU automatically when no GPU is present.
    return xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=20,
        reg_alpha=2.0,
        reg_lambda=5.0,
        gamma=0.1,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=4,
        objective="reg:squarederror",
    )


def _train_target(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple["xgb.XGBRegressor", dict[str, float]]:
    subset = train.dropna(subset=[target_col]).copy()
    audit: dict[str, float] = {
        "target": target_col,
        "n_train_total": int(len(subset)),
        "cv_folds": 0,
        "cv_mae": math.nan,
        "cv_r2": math.nan,
        "baseline_naive_mae": math.nan,
    }
    if len(subset) < 200:
        return _xgb_regressor().fit(subset[feature_cols].fillna(np.nan), subset[target_col].astype(float)) if len(subset) else _xgb_regressor().fit(np.zeros((1, len(feature_cols))), np.zeros(1)), audit

    groups = subset["病案号_norm"].fillna("").astype(str).values
    n_groups = len(set(groups))
    n_splits = int(min(5, max(2, n_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    cv_preds: list[float] = []
    cv_truth: list[float] = []
    for train_idx, val_idx in gkf.split(subset[feature_cols], subset[target_col], groups=groups):
        x_tr = subset.iloc[train_idx][feature_cols].astype(float)
        y_tr = subset.iloc[train_idx][target_col].astype(float)
        x_va = subset.iloc[val_idx][feature_cols].astype(float)
        y_va = subset.iloc[val_idx][target_col].astype(float)
        model = _xgb_regressor()
        model.fit(x_tr, y_tr)
        pred = model.predict(x_va)
        cv_preds.extend(pred.tolist())
        cv_truth.extend(y_va.tolist())
    truth_arr = np.asarray(cv_truth, dtype=float)
    pred_arr = np.asarray(cv_preds, dtype=float)
    audit["cv_folds"] = float(n_splits)
    audit["cv_mae"] = float(mean_absolute_error(truth_arr, pred_arr))
    audit["cv_r2"] = float(r2_score(truth_arr, pred_arr)) if truth_arr.var() > 1e-9 else math.nan
    naive_pred = float(np.median(truth_arr))
    audit["baseline_naive_mae"] = float(np.mean(np.abs(truth_arr - naive_pred)))

    final = _xgb_regressor()
    final.fit(subset[feature_cols].astype(float), subset[target_col].astype(float))
    return final, audit


def train_xgb_lgdi_models(
    data: pd.DataFrame, baseline_mask: pd.Series
) -> tuple[dict[str, "xgb.XGBRegressor"], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched, feature_cols = build_admission_features(data)
    baseline = enriched[baseline_mask.reindex(enriched.index, fill_value=False)].copy()

    models: dict[str, xgb.XGBRegressor] = {}
    audit_rows: list[dict[str, float | str]] = []
    importance_rows: list[dict[str, float | str]] = []
    for target in ["next_los_days", "next_gap_days"]:
        model, audit = _train_target(baseline, feature_cols, target)
        models[target] = model
        audit_rows.append(audit)
        try:
            booster_score = model.get_booster().get_score(importance_type="gain")
        except Exception:
            booster_score = {}
        for idx, name in enumerate(feature_cols):
            key = f"f{idx}"
            importance_rows.append(
                {
                    "target": target,
                    "feature": name,
                    "gain": float(booster_score.get(key, 0.0)),
                }
            )

    enriched["pred_next_los_days"] = models["next_los_days"].predict(enriched[feature_cols].astype(float))
    enriched["pred_next_gap_days"] = models["next_gap_days"].predict(enriched[feature_cols].astype(float))
    enriched["resid_next_los_days"] = enriched["next_los_days"] - enriched["pred_next_los_days"]
    enriched["resid_next_gap_days"] = enriched["next_gap_days"] - enriched["pred_next_gap_days"]

    audit_df = pd.DataFrame(audit_rows)
    importance_df = pd.DataFrame(importance_rows)
    return models, feature_cols, enriched, audit_df, importance_df


def baseline_residual_scale(
    enriched: pd.DataFrame, baseline_mask: pd.Series
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per-(group, target) baseline mean absolute residual to use as the MASE denominator."""
    baseline = enriched[baseline_mask.reindex(enriched.index, fill_value=False)].copy()
    stats: dict[str, dict[str, dict[str, float | int]]] = {}
    for group in COMORBIDITIES:
        stats[group] = {}
        subset = baseline[baseline[group].eq(True)]
        for target in ["next_los_days", "next_gap_days"]:
            resid = pd.to_numeric(subset[f"resid_{target}"], errors="coerce").dropna()
            if resid.empty:
                stats[group][target] = {"n": 0, "mae_scale": 1.0, "mean_resid": math.nan}
                continue
            mae_scale = float(np.mean(np.abs(resid)))
            if not np.isfinite(mae_scale) or mae_scale <= 1e-9:
                mae_scale = 1.0
            stats[group][target] = {
                "n": int(len(resid)),
                "mae_scale": mae_scale,
                "mean_resid": float(resid.mean()),
            }
    return stats


def xgb_prediction_metrics(
    truth: pd.Series, pred: pd.Series, scale: dict[str, float | int], direction: float
) -> dict[str, float | int]:
    truth_num = pd.to_numeric(truth, errors="coerce")
    pred_num = pd.to_numeric(pred, errors="coerce")
    valid = truth_num.notna() & pred_num.notna()
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "observed_mean": math.nan, "observed_median": math.nan, "predicted_mean": math.nan, "mae": math.nan, "mase": math.nan, "r2": math.nan, "signed_mase_residual": math.nan}
    truth_arr = truth_num[valid].to_numpy(dtype=float)
    pred_arr = pred_num[valid].to_numpy(dtype=float)
    residuals = truth_arr - pred_arr
    mae = float(np.mean(np.abs(residuals)))
    mae_scale = float(scale.get("mae_scale", 1.0)) if scale else 1.0
    if not np.isfinite(mae_scale) or mae_scale <= 1e-9:
        mae_scale = 1.0
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((truth_arr - truth_arr.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-9 else math.nan
    mean_resid = float(np.mean(residuals))
    signed = direction * mean_resid / mae_scale
    return {
        "n": n,
        "observed_mean": float(truth_arr.mean()),
        "observed_median": float(np.median(truth_arr)),
        "predicted_mean": float(pred_arr.mean()),
        "mae": mae,
        "mase": float(mae / mae_scale),
        "r2": float(r2),
        "signed_mase_residual": float(signed),
    }


def build_baseline_stats(
    data: pd.DataFrame, baseline_mask: pd.Series
) -> tuple[dict[str, dict[str, dict[str, float | int]]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train XGBoost LOS/gap models on the baseline window and return per-group residual scales."""
    _models, _feature_cols, enriched, audit_df, importance_df = train_xgb_lgdi_models(data, baseline_mask)
    stats = baseline_residual_scale(enriched, baseline_mask)
    rows: list[dict[str, object]] = []
    for group in COMORBIDITIES:
        for target in ["next_los_days", "next_gap_days"]:
            row = {"group": group, "outcome": target, **stats[group][target]}
            rows.append(row)
    baseline_df = pd.DataFrame(rows)
    return stats, baseline_df, enriched, audit_df, importance_df


def group_snapshot(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, dict[str, float | int]]],
    min_group_n: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    snapshots: dict[str, dict[str, object]] = {}
    detail_rows: list[dict[str, object]] = []
    for group in COMORBIDITIES:
        subset = frame[frame[group].eq(True)]
        group_row: dict[str, object] = {"group": group, "n_admissions": int(len(subset))}
        outcome_scores: list[float] = []
        for target, direction in [("next_los_days", 1.0), ("next_gap_days", -1.0)]:
            scale = stats[group].get(target, {"mae_scale": 1.0})
            metrics = xgb_prediction_metrics(
                subset[target] if target in subset.columns else pd.Series(dtype=float),
                subset[f"pred_{target}"] if f"pred_{target}" in subset.columns else pd.Series(dtype=float),
                scale,
                direction,
            )
            for key, value in metrics.items():
                group_row[f"{target}_{key}"] = value
            if int(metrics["n"]) >= min_group_n and np.isfinite(float(metrics["signed_mase_residual"])):
                outcome_scores.append(float(metrics["signed_mase_residual"]))
            detail_rows.append({"group": group, "outcome": target, **metrics})
        group_row["group_score"] = float(np.mean(outcome_scores)) if outcome_scores else math.nan
        snapshots[group] = group_row
    return snapshots, detail_rows


def window_row(label: str, anchor: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, frame: pd.DataFrame, stats: dict[str, dict[str, dict[str, float | int]]], min_total_n: int = 50, min_group_n: int = 10) -> tuple[dict[str, object], list[dict[str, object]]]:
    base_row: dict[str, object] = {
        "label": label,
        "window_anchor": anchor.date().isoformat(),
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "n_admissions": int(len(frame)),
        "event_window": bool(start <= EVENT_END and end >= EVENT_START),
        "lead_or_event_window": bool(start <= EVENT_END and end >= LEAD_START),
    }
    if len(frame) < min_total_n:
        base_row.update({"valid": False, "resp_score": math.nan, "mean_other_score": math.nan, "lgdi": math.nan})
        return base_row, []
    monitoring = frame[frame["is_covid_positive"].eq(False)]
    snapshots, details = group_snapshot(monitoring, stats, min_group_n=min_group_n)
    resp_score = float(snapshots[RESPIRATORY]["group_score"])
    other_scores = [float(row["group_score"]) for group, row in snapshots.items() if group != RESPIRATORY and np.isfinite(float(row["group_score"]))]
    mean_other = float(np.mean(other_scores)) if other_scores else math.nan
    lgdi = resp_score - mean_other if np.isfinite(resp_score) and np.isfinite(mean_other) else math.nan
    base_row.update({"n_admissions_excluding_covid_reference": int(len(monitoring)), "valid": bool(np.isfinite(lgdi)), "resp_score": resp_score, "mean_other_score": mean_other, "lgdi": lgdi})
    for group, row in snapshots.items():
        base_row[f"score_{group}"] = row["group_score"]
        base_row[f"n_{group}"] = row["n_admissions"]
    for detail in details:
        detail.update({"label": label, "window_anchor": anchor.date().isoformat(), "window_start": start.date().isoformat(), "window_end": end.date().isoformat()})
    return base_row, details


def make_lgdi_windows(data: pd.DataFrame, stats: dict[str, dict[str, dict[str, float | int]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    first = pd.Timestamp("2016-01-04")
    last = data["week_start"].max()
    for anchor in pd.date_range(first, last, freq="W-MON"):
        start = anchor - pd.Timedelta(days=21)
        end = anchor + pd.Timedelta(days=6)
        frame = data[(data["admit_dt"] >= start) & (data["admit_dt"] <= end)]
        label = f"{anchor.isocalendar().year}W{anchor.isocalendar().week:02d}"
        row, detail = window_row(label, anchor, start, end, frame, stats)
        rows.append(row)
        details.extend(detail)
    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return timeline, pd.DataFrame(details)
    for col in ["window_anchor", "window_start", "window_end"]:
        timeline[f"{col}_dt"] = pd.to_datetime(timeline[col])
    return timeline, pd.DataFrame(details)


def thresholds(timeline: pd.DataFrame) -> pd.DataFrame:
    baseline = timeline[timeline["valid"].eq(True) & timeline["window_anchor_dt"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31"))]["lgdi"].dropna()
    rows: list[dict[str, object]] = []
    if baseline.empty:
        return pd.DataFrame(rows)
    rows.extend([
        {"threshold_rule": "lgdi_mean_plus_1_5sd", "threshold_value": float(baseline.mean() + 1.5 * baseline.std())},
        {"threshold_rule": "lgdi_mean_plus_2sd", "threshold_value": float(baseline.mean() + 2.0 * baseline.std())},
        {"threshold_rule": "lgdi_p97_5", "threshold_value": float(np.percentile(baseline, 97.5))},
    ])
    return pd.DataFrame(rows)


def performance_for(timeline: pd.DataFrame, rule: str, value: float) -> dict[str, object]:
    monitor = timeline[timeline["valid"].eq(True) & timeline["window_anchor_dt"].between(MONITOR_START, MONITOR_END)].copy()
    monitor["alert"] = monitor["lgdi"] >= value
    event = monitor["event_window"].astype(bool)
    lead_or_event = monitor["lead_or_event_window"].astype(bool)
    alert = monitor["alert"].astype(bool)
    tp_event = int((alert & event).sum())
    tp_lead = int((alert & lead_or_event).sum())
    fp_event = int((alert & ~event).sum())
    fp_lead = int((alert & ~lead_or_event).sum())
    fn = int((~alert & event).sum())
    tn_event = int((~alert & ~event).sum())
    tn_lead = int((~alert & ~lead_or_event).sum())
    def safe_div(num: int, den: int) -> float:
        return float(num / den) if den else math.nan
    event_alerts = monitor[alert & monitor["window_anchor_dt"].between(LEAD_START, EVENT_END)]
    first_alert = pd.Timestamp(event_alerts["window_anchor_dt"].min()) if len(event_alerts) else pd.NaT
    return {
        "threshold_rule": rule,
        "threshold_value": value,
        "monitor_windows": int(len(monitor)),
        "event_windows": int(event.sum()),
        "alert_windows": int(alert.sum()),
        "true_positive_event_windows": tp_event,
        "true_positive_lead_or_event_windows": tp_lead,
        "false_positive_event_strict": fp_event,
        "false_positive_lead_allowed": fp_lead,
        "false_negative_event_windows": fn,
        "true_negative_event_strict": tn_event,
        "true_negative_lead_allowed": tn_lead,
        "sensitivity_event_week": safe_div(tp_event, tp_event + fn),
        "precision_ppv_event_strict": safe_div(tp_event, tp_event + fp_event),
        "precision_ppv_lead_allowed": safe_div(tp_lead, tp_lead + fp_lead),
        "false_alarm_rate_event_strict": safe_div(fp_event, fp_event + tn_event),
        "false_alarm_rate_lead_allowed": safe_div(fp_lead, fp_lead + tn_lead),
        "first_alert_in_lead_or_event": None if pd.isna(first_alert) else first_alert.date().isoformat(),
        "lead_time_days": None if pd.isna(first_alert) else int((EVENT_START - first_alert).days),
    }


def summarize_performance(timeline: pd.DataFrame, threshold_table: pd.DataFrame) -> pd.DataFrame:
    rows = [performance_for(timeline, str(row["threshold_rule"]), float(row["threshold_value"])) for _, row in threshold_table.iterrows()]
    return pd.DataFrame(rows)


def normalized_icd(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def numeric_icd3(codes: pd.Series) -> pd.Series:
    return pd.to_numeric(codes.str.extract(r"^(\d{3})")[0], errors="coerce")


def nwicu_group_flags(dx: pd.DataFrame) -> pd.DataFrame:
    codes = normalized_icd(dx["icd_code"])
    code3 = numeric_icd3(codes)
    version = pd.to_numeric(dx["icd_version"], errors="coerce")
    is_icd9 = version.eq(9)
    is_icd10 = version.eq(10)
    flags = pd.DataFrame({"hadm_id": dx["hadm_id"]})
    flags["is_covid_positive"] = is_icd10 & codes.str.startswith("U07")
    flags["Respiratory"] = (is_icd9 & code3.between(490, 496, inclusive="both")) | (is_icd10 & codes.str.startswith(("J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47")))
    flags["Cardiovascular"] = (is_icd9 & code3.between(390, 459, inclusive="both")) | (is_icd10 & codes.str.startswith("I"))
    flags["Hypertension"] = (is_icd9 & code3.between(401, 405, inclusive="both")) | (is_icd10 & codes.str.startswith(("I10", "I11", "I12", "I13", "I15")))
    flags["Diabetes"] = (is_icd9 & code3.eq(250)) | (is_icd10 & codes.str.startswith(("E10", "E11", "E12", "E13", "E14")))
    flags["Cerebrovascular"] = (is_icd9 & code3.between(430, 438, inclusive="both")) | (is_icd10 & codes.str.startswith(("I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69")))
    flags["Renal"] = (is_icd9 & code3.between(580, 589, inclusive="both")) | (is_icd10 & codes.str.startswith(("N17", "N18", "N19")))
    return flags.groupby("hadm_id", as_index=False)[["is_covid_positive", *COMORBIDITIES]].max()


def run_nwicu_validation() -> tuple[pd.DataFrame, dict[str, object]]:
    hosp_root = NWICU_ROOT / "data" / "nw_hosp"
    admissions_path = hosp_root / "admissions.csv.gz"
    dx_path = hosp_root / "diagnoses_icd.csv.gz"
    if not admissions_path.exists() or not dx_path.exists():
        return pd.DataFrame(), {"available": False, "reason": "NWICU admissions or diagnoses file missing"}
    admissions = pd.read_csv(admissions_path, parse_dates=["admittime", "dischtime"])
    dx = pd.read_csv(dx_path, dtype={"icd_code": str})
    data = admissions.merge(nwicu_group_flags(dx), on="hadm_id", how="left")
    data["病案号_norm"] = normalize_mrn(data["subject_id"])
    data["admit_dt"] = data["admittime"]
    data["discharge_dt"] = data["dischtime"]
    data["los_days"] = (data["discharge_dt"] - data["admit_dt"]).dt.total_seconds() / 86400
    data = data.sort_values(["subject_id", "admit_dt", "discharge_dt"]).copy()
    data["prev_discharge_dt"] = data.groupby("subject_id")["discharge_dt"].shift(1)
    data["gap_days"] = (data["admit_dt"] - data["prev_discharge_dt"]).dt.total_seconds() / 86400
    data.loc[(data["los_days"] < 0) | (data["los_days"] > 180), "los_days"] = np.nan
    data.loc[(data["gap_days"] < 0) | (data["gap_days"] > 3650), "gap_days"] = np.nan
    for col in ["is_covid_positive", *COMORBIDITIES]:
        data[col] = data[col].fillna(False).astype(bool)
    stats, _baseline_df, enriched, _audit, _imp = build_baseline_stats(data, data["is_covid_positive"].eq(False))
    enriched = enriched.merge(
        data[["hadm_id", "subject_id", "is_covid_positive"]],
        left_index=True, right_index=True, how="left", suffixes=("", "_orig"),
    ) if "hadm_id" in data.columns else enriched
    event = enriched[enriched["is_covid_positive"].eq(True)]
    snapshots, _ = group_snapshot(event, stats, min_group_n=10)
    rows = []
    for group, row in snapshots.items():
        rows.append({"dataset": "NWICU v0.1.0", **row})
    metrics = pd.DataFrame(rows)
    resp_score = float(metrics.loc[metrics["group"].eq(RESPIRATORY), "group_score"].iloc[0]) if len(metrics) else math.nan
    other = metrics.loc[~metrics["group"].eq(RESPIRATORY), "group_score"].dropna()
    summary = {
        "available": True,
        "interpretation": "NWICU dates are de-identified/shifted; validation is a COVID-coded admission stress test against the dataset's non-COVID baseline using the same XGBoost residual scoring as the primary cohort, not calendar surveillance.",
        "admissions": int(len(data)),
        "covid_reference_admissions": int(event["is_covid_positive"].sum()) if len(event) else 0,
        "covid_reference_patients": int(data.loc[data["is_covid_positive"].eq(True), "subject_id"].nunique()),
        "respiratory_group_score": resp_score,
        "mean_nonrespiratory_group_score": float(other.mean()) if len(other) else math.nan,
        "nwicu_lgdi": resp_score - float(other.mean()) if len(other) and np.isfinite(resp_score) else math.nan,
        "top_group": metrics.sort_values("group_score", ascending=False).head(1).to_dict(orient="records") if len(metrics) else [],
    }
    return metrics, summary


def run_flunet_correlation(timeline: pd.DataFrame) -> pd.DataFrame:
    if not FLUNET_CHINA.exists() or timeline.empty:
        return pd.DataFrame()
    flu = read_csv_best(FLUNET_CHINA)
    flu["week_start"] = pd.to_datetime(flu["ISO_WEEKSTARTDATE"], errors="coerce")
    flu["spec_processed"] = pd.to_numeric(flu.get("SPEC_PROCESSED_NB"), errors="coerce")
    flu["inf_all"] = pd.to_numeric(flu.get("INF_ALL"), errors="coerce")
    flu["flu_positive_rate"] = flu["inf_all"] / flu["spec_processed"].replace(0, np.nan)
    flu_weekly = flu[["week_start", "flu_positive_rate", "spec_processed", "inf_all"]].dropna(subset=["week_start"]).copy()
    series = timeline[timeline["valid"].eq(True)][["window_anchor_dt", "lgdi"]].rename(columns={"window_anchor_dt": "week_start"})
    merged = series.merge(flu_weekly, on="week_start", how="inner").sort_values("week_start")
    rows: list[dict[str, object]] = []
    for lag in range(0, 5):
        test = merged.copy()
        test["flu_future"] = test["flu_positive_rate"].shift(-lag)
        test = test.dropna(subset=["lgdi", "flu_future"])
        if len(test) >= 8:
            rho, p_value = spearmanr(test["lgdi"], test["flu_future"])
        else:
            rho, p_value = math.nan, math.nan
        rows.append({"lgdi_leads_flunet_weeks": lag, "n_weeks": int(len(test)), "spearman_rho": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows)


def _add_policy_gap(ax: plt.Axes, label: bool = True) -> None:
    """Add a grey shaded band and optional label for the 42k policy-induced data-loss period."""
    ax.axvspan(POLICY_GAP_START, POLICY_GAP_END, color="#888888", alpha=0.22,
               label=POLICY_GAP_LABEL if label else None, zorder=0)
    if label:
        mid = POLICY_GAP_START + (POLICY_GAP_END - POLICY_GAP_START) / 2
        ylim = ax.get_ylim()
        ax.text(mid, ylim[1] * 0.97, POLICY_GAP_LABEL, ha="center", va="top",
                fontsize=6.5, color="#555555", style="italic", clip_on=True)


def _plot_policy_break(ax: plt.Axes, frame: pd.DataFrame, date_col: str, value_col: str, **kwargs) -> None:
    """Plot a line series with a visible break across the policy-gap interval."""
    before = frame[frame[date_col] <= POLICY_GAP_START].copy()
    after = frame[frame[date_col] >= POLICY_GAP_END].copy()
    first = True
    for segment in [before, after]:
        if len(segment):
            kw = dict(kwargs)
            if not first:
                kw.pop("label", None)
            ax.plot(segment[date_col], segment[value_col], **kw)
            first = False


def plot_figure(timeline: pd.DataFrame, threshold_table: pd.DataFrame, nwicu_metrics: pd.DataFrame, flunet_corr: pd.DataFrame) -> None:
    valid = timeline[timeline["valid"].eq(True)].copy()
    threshold_value = float(threshold_table.loc[threshold_table["threshold_rule"].eq("lgdi_mean_plus_1_5sd"), "threshold_value"].iloc[0]) if len(threshold_table) else math.nan
    fig = plt.figure(figsize=(14, 9.5))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30, left=0.08, right=0.97, top=0.97, bottom=0.10)
    ax_a = fig.add_subplot(grid[0, :])
    _add_policy_gap(ax_a, label=True)
    _plot_policy_break(ax_a, valid, "window_anchor_dt", "lgdi",
                       color=COLORS["primary"], linewidth=1.25, label="4-week LOS-Gap Deviation Index")
    if np.isfinite(threshold_value):
        ax_a.axhline(threshold_value, color=COLORS["warn"], linestyle="--", linewidth=1.0, label="Baseline mean + 1.5 SD")
    ax_a.axvspan(EVENT_START, EVENT_END, color=COLORS["secondary"], alpha=0.16, label="Dec 2022-Jan 2023 event window")
    ax_a.set_ylabel("LGDI (respiratory minus non-respiratory)")
    _panel_label(ax_a, "A")
    ax_a.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

    ax_b = fig.add_subplot(grid[1, 0])
    if len(nwicu_metrics):
        ordered = nwicu_metrics.sort_values("group_score", ascending=False)
        colors = [COLORS["secondary"] if group == RESPIRATORY else COLORS["primary"] for group in ordered["group"]]
        ax_b.bar(np.arange(len(ordered)), ordered["group_score"], color=colors, edgecolor="white", linewidth=0.3)
        ax_b.set_xticks(np.arange(len(ordered)))
        ax_b.set_xticklabels(ordered["group"], rotation=25, ha="right")
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_ylabel("COVID-coded group score")
    _panel_label(ax_b, "B")
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

    ax_c = fig.add_subplot(grid[1, 1])
    if len(flunet_corr):
        ax_c.bar(flunet_corr["lgdi_leads_flunet_weeks"], flunet_corr["spearman_rho"], color=COLORS["accent"], edgecolor="white", linewidth=0.3)
    ax_c.axhline(0, color="black", linewidth=0.6)
    ax_c.set_xlabel("LGDI lead versus WHO FluNet (weeks)")
    ax_c.set_ylabel("Spearman rho")
    _panel_label(ax_c, "C")
    ax_c.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigureS15_lgdi_surveillance.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "FigureS15_lgdi_surveillance.tif", dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def sync_to_package(prefix: str, include_figure: bool) -> None:
    PACKAGE_FIGURES.mkdir(parents=True, exist_ok=True)
    PACKAGE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    if include_figure:
        for extension in ["png", "pdf", "svg", "tif"]:
            source = OUT_DIR / f"FigureS15_lgdi_surveillance.{extension}"
            if source.exists():
                shutil.copy2(source, PACKAGE_FIGURES / source.name)
    for source in OUT_DIR.glob(f"{prefix}_*"):
        if source.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(source, PACKAGE_ANALYSIS / source.name)


def run_analysis(input_table: Path, prefix: str, include_external: bool, cohort_label: str | None = None) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_cardiac_table(input_table)
    baseline_mask = data["year"].between(2016, 2018)
    stats, baseline_df, enriched, audit_df, importance_df = build_baseline_stats(data, baseline_mask)
    timeline, details = make_lgdi_windows(enriched, stats)
    threshold_table = thresholds(timeline)
    performance = summarize_performance(timeline, threshold_table)

    export_timeline = timeline.drop(columns=[col for col in timeline.columns if col.endswith("_dt")])
    export_timeline.to_csv(OUT_DIR / f"{prefix}_rolling4_weekly.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT_DIR / f"{prefix}_group_window_metrics.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(OUT_DIR / f"{prefix}_performance.csv", index=False, encoding="utf-8-sig")
    threshold_table.to_csv(OUT_DIR / f"{prefix}_thresholds.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(OUT_DIR / f"{prefix}_baseline_stats.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(OUT_DIR / f"{prefix}_model_audit.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(OUT_DIR / f"{prefix}_feature_importance.csv", index=False, encoding="utf-8-sig")

    nwicu_metrics = pd.DataFrame()
    nwicu_summary: dict[str, object] = {"available": False, "reason": "external validation skipped"}
    flunet_corr = pd.DataFrame()
    if include_external:
        nwicu_metrics, nwicu_summary = run_nwicu_validation()
        nwicu_metrics.to_csv(OUT_DIR / f"{prefix}_nwicu_group_metrics.csv", index=False, encoding="utf-8-sig")
        (OUT_DIR / f"{prefix}_nwicu_summary.json").write_text(json.dumps(nwicu_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        flunet_corr = run_flunet_correlation(timeline)
        flunet_corr.to_csv(OUT_DIR / f"{prefix}_flunet_lag_correlation.csv", index=False, encoding="utf-8-sig")
        plot_figure(timeline, threshold_table, nwicu_metrics, flunet_corr)

    primary = performance[performance["threshold_rule"].eq("lgdi_mean_plus_1_5sd")].head(1).to_dict(orient="records")
    audit_records = audit_df.to_dict(orient="records") if not audit_df.empty else []
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "input_table": str(input_table),
        "cohort_label": cohort_label or input_table.stem,
        "algorithm": (
            "XGBoost-based LGDI: per-admission XGBRegressor models (tree_method=hist, "
            "device=cuda, n_estimators=1200, max_depth=5, lr=0.01, subsample=0.7, "
            "colsample_bytree=0.6, min_child_weight=20, reg_alpha=2.0, reg_lambda=5.0, "
            "gamma=0.1; GPU-tuned 2026-05 baseline, GroupKFold by patient over the "
            "2016-2018 baseline) predict next-admission length-of-stay and next-admission "
            "inter-admission gap from leak-free history (visit order, prior LOS/gap stats, "
            "comorbidity flags, available labs, utilization counts when present, calendar "
            "seasonality). Residuals (truth-prediction) are aggregated per group x rolling "
            "4-week window into MAE, MASE (vs baseline residual scale), R2, and a signed "
            "MASE residual (direction = +1 for next_los, -1 for next_gap). LGDI = "
            "respiratory group score - mean(other-group group scores)."
        ),
        "cv_audit": audit_records,
        "cohort": {
            "admissions_with_dates": int(len(data)),
            "unique_patient_record_numbers": int(data["病案号_norm"].replace("", np.nan).nunique()),
            "date_range": f"{data['admit_dt'].min().date()} to {data['admit_dt'].max().date()}",
            "covid_reference_admissions": int(data["is_covid_positive"].sum()),
            "comorbidity_counts": {group: int(data[group].sum()) for group in COMORBIDITIES},
        },
        "baseline": {"start": "2016-01-01", "end": "2018-12-31", "n_admissions": int(baseline_mask.sum())},
        "event_definition": {"event_start": EVENT_START.date().isoformat(), "event_end": EVENT_END.date().isoformat(), "lead_start": LEAD_START.date().isoformat()},
        "primary_operating_characteristics": primary[0] if primary else {},
        "nwicu_validation": nwicu_summary,
        "outputs": {
            "timeline_csv": str(OUT_DIR / f"{prefix}_rolling4_weekly.csv"),
            "group_window_metrics_csv": str(OUT_DIR / f"{prefix}_group_window_metrics.csv"),
            "performance_csv": str(OUT_DIR / f"{prefix}_performance.csv"),
            "baseline_stats_csv": str(OUT_DIR / f"{prefix}_baseline_stats.csv"),
            "model_audit_csv": str(OUT_DIR / f"{prefix}_model_audit.csv"),
            "feature_importance_csv": str(OUT_DIR / f"{prefix}_feature_importance.csv"),
        },
    }
    (OUT_DIR / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sync_to_package(prefix, include_figure=include_external)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LOS-Gap Deviation Index surveillance analysis.")
    parser.add_argument("--input-table", type=Path, default=INPUT_TABLE)
    parser.add_argument("--prefix", default="lgdi")
    parser.add_argument("--cohort-label", default=None)
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(args.input_table, args.prefix, include_external=not args.skip_external, cohort_label=args.cohort_label)
    print(json.dumps(summary["cohort"], ensure_ascii=False, indent=2))
    print("Primary LGDI operating characteristics:")
    print(json.dumps(summary["primary_operating_characteristics"], ensure_ascii=False, indent=2))
    if not args.skip_external:
        print("NWICU LGDI summary:")
        print(json.dumps(summary["nwicu_validation"], ensure_ascii=False, indent=2))
    print(f"Saved outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())