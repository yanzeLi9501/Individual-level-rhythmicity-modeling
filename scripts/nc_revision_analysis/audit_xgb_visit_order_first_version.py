"""Audit why the visit-order XGBoost curve diverges from the first version.

This script does not regenerate the publication figure. It compares the current
history-model feature matrix against the first-version enhanced feature set under
the capped-gap objectives used by the early manuscript numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


BASE_DIR = Path(__file__).resolve().parent
READMISSION_DIR = Path(r"data/readmission_output")
HISTORY_FEATURES_CSV = READMISSION_DIR / "history_model" / "history_features.csv"
TRAIN_CSV = READMISSION_DIR / "train_data.csv"
BROAD_RESULTS_JSON = READMISSION_DIR / "broad_config_results.json"
OUT_DIR = BASE_DIR / "xgboost_visit_order_results"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5


def detect_device() -> str:
    try:
        model = xgb.XGBRegressor(device="cuda", n_estimators=2, verbosity=0)
        random_generator = np.random.default_rng(1)
        feature_sample = random_generator.normal(size=(20, 4)).astype(np.float32)
        target_sample = random_generator.normal(size=20).astype(np.float32)
        model.fit(feature_sample, target_sample)
        return "cuda"
    except Exception:
        return "cpu"


def add_first_version_enhanced_features(features: pd.DataFrame) -> pd.DataFrame:
    enhanced = features.copy()

    incoming_gap = pd.to_numeric(enhanced["incoming_gap"], errors="coerce").to_numpy()
    gap_ema = pd.to_numeric(enhanced["gap_ema"], errors="coerce").to_numpy()
    gap_mean = pd.to_numeric(enhanced["gap_mean_prev"], errors="coerce").to_numpy()
    gap_cv = pd.to_numeric(enhanced["gap_cv_prev"], errors="coerce").to_numpy()
    prev_gap_2 = pd.to_numeric(enhanced["prev_gap_2"], errors="coerce").to_numpy()
    prev_gap_3 = pd.to_numeric(enhanced["prev_gap_3"], errors="coerce").to_numpy()
    los_days = pd.to_numeric(enhanced["los_days"], errors="coerce").to_numpy()
    los_mean = pd.to_numeric(enhanced["los_mean_prev"], errors="coerce").to_numpy()
    los_ema = pd.to_numeric(enhanced["los_ema"], errors="coerce").to_numpy()
    prev_los_1 = pd.to_numeric(enhanced["prev_los_1"], errors="coerce").to_numpy()
    prev_los_2 = pd.to_numeric(enhanced["prev_los_2"], errors="coerce").to_numpy()
    admission_frequency = pd.to_numeric(enhanced["admission_frequency"], errors="coerce").to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        enhanced["gap_regularity"] = 1.0 / (1.0 + np.where(np.isnan(gap_cv), 10, gap_cv))
        enhanced["gap_deviation"] = (incoming_gap - gap_mean) / np.where(gap_mean == 0, np.nan, gap_mean)
        enhanced["gap_shortening"] = (incoming_gap < gap_mean).astype(float)
        enhanced["gap_last_diff"] = incoming_gap - prev_gap_2
        enhanced["gap_last_ratio"] = incoming_gap / np.where(prev_gap_2 == 0, np.nan, prev_gap_2)
        enhanced["gap_accel"] = (incoming_gap - prev_gap_2) - (prev_gap_2 - prev_gap_3)
        enhanced["gap_ema_ratio"] = incoming_gap / np.where(gap_ema == 0, np.nan, gap_ema)
        enhanced["log_incoming_gap_v2"] = np.log1p(np.maximum(incoming_gap, 0))
        enhanced["log_gap_mean"] = np.log1p(np.maximum(gap_mean, 0))
        enhanced["gap_range"] = pd.to_numeric(enhanced["gap_max_prev"], errors="coerce") - pd.to_numeric(enhanced["gap_min_prev"], errors="coerce")
        enhanced["gap_iqr_proxy"] = pd.to_numeric(enhanced["gap_median_prev"], errors="coerce") - pd.to_numeric(enhanced["gap_min_prev"], errors="coerce")
        enhanced["los_deviation"] = (los_days - los_mean) / np.where(los_mean == 0, np.nan, los_mean)
        enhanced["los_ema_ratio"] = los_days / np.where(los_ema == 0, np.nan, los_ema)
        enhanced["log_los_days_v2"] = np.log1p(np.maximum(los_days, 0))
        enhanced["los_days_sq"] = los_days**2
        enhanced["los_days_sqrt"] = np.sqrt(np.maximum(los_days, 0))
        enhanced["los_wavg_2"] = 0.7 * los_days + 0.3 * prev_los_1
        enhanced["los_wavg_3"] = 0.5 * los_days + 0.3 * prev_los_1 + 0.2 * prev_los_2
        enhanced["gap_x_los"] = incoming_gap * los_days
        enhanced["gap_x_freq"] = incoming_gap * admission_frequency
        enhanced["los_x_freq"] = los_days * admission_frequency

    for column in enhanced.columns:
        if enhanced[column].dtype in ["float64", "float32"]:
            enhanced[column] = np.where(np.isinf(enhanced[column].to_numpy()), np.nan, enhanced[column].to_numpy())

    return enhanced


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"visit_id", "patient_id", "target_gap_days", "target_next_los"}
    numeric_dtypes = {"float64", "float32", "int64", "int32"}
    return [
        column
        for column in features.columns
        if column not in excluded and str(features[column].dtype) in numeric_dtypes
    ]


def build_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    feature_matrix = frame[columns].to_numpy(dtype=np.float32)
    column_means = np.nanmean(feature_matrix, axis=0)
    column_means = np.where(np.isnan(column_means), 0, column_means)
    missing_rows, missing_columns = np.where(np.isnan(feature_matrix))
    feature_matrix[missing_rows, missing_columns] = column_means[missing_columns]
    return feature_matrix


def make_params(params: dict[str, object], device: str) -> dict[str, object]:
    model_params = dict(params)
    model_params.update(
        {
            "device": device,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
    )
    return model_params


def fold_indices(patient_ids: np.ndarray, split_mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if split_mode == "row":
        row_indices = np.arange(len(patient_ids))
        return list(kfold.split(row_indices))

    unique_patients = np.unique(patient_ids)
    folds = []
    for train_patient_index, valid_patient_index in kfold.split(unique_patients):
        train_patients = set(unique_patients[train_patient_index])
        valid_patients = set(unique_patients[valid_patient_index])
        train_mask = np.array([patient_id in train_patients for patient_id in patient_ids])
        valid_mask = np.array([patient_id in valid_patients for patient_id in patient_ids])
        folds.append((np.where(train_mask)[0], np.where(valid_mask)[0]))
    return folds


def evaluate_config(
    frame: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, object],
    min_visit_order: int,
    gap_cap: int | None,
    split_mode: str,
) -> dict[str, object]:
    subset = frame[frame["visit_order"] >= min_visit_order].copy()
    if gap_cap is not None:
        subset["target_gap_days"] = subset["target_gap_days"].clip(upper=gap_cap)
    subset = subset[subset["target_gap_days"].notna()].copy()

    feature_matrix = build_matrix(subset, feature_columns)
    target_values = subset["target_gap_days"].to_numpy(dtype=np.float32)
    patient_ids = subset["patient_id"].to_numpy()

    r2_scores: list[float] = []
    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    for train_index, valid_index in fold_indices(patient_ids, split_mode):
        model = xgb.XGBRegressor(**params)
        model.fit(feature_matrix[train_index], target_values[train_index], eval_set=[(feature_matrix[valid_index], target_values[valid_index])], verbose=False)
        predictions = model.predict(feature_matrix[valid_index])
        r2_scores.append(float(r2_score(target_values[valid_index], predictions)))
        mae_scores.append(float(mean_absolute_error(target_values[valid_index], predictions)))
        rmse_scores.append(float(np.sqrt(mean_squared_error(target_values[valid_index], predictions))))

    return {
        "min_visit_order": min_visit_order,
        "gap_cap": gap_cap,
        "split_mode": split_mode,
        "n": int(len(subset)),
        "n_features": int(len(feature_columns)),
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores, ddof=0)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores, ddof=0)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores, ddof=0)),
        "r2_folds": r2_scores,
    }


def main() -> None:
    train_data = pd.read_csv(TRAIN_CSV, low_memory=False)
    json_features = pd.read_csv(HISTORY_FEATURES_CSV, low_memory=False)
    for column in json_features.columns:
        json_features[column] = pd.to_numeric(json_features[column], errors="coerce")
    json_features["target_gap_days"] = pd.to_numeric(train_data["target_gap_days"], errors="coerce")
    json_features["target_next_los"] = pd.to_numeric(train_data["target_next_los"], errors="coerce")
    json_features["patient_id"] = train_data["病案号"].astype(str)

    enhanced_features = add_first_version_enhanced_features(json_features)
    json_feature_columns = [
        column for column in numeric_feature_columns(json_features) if column not in {"target_gap_days", "target_next_los"}
    ]
    enhanced_feature_columns = numeric_feature_columns(enhanced_features)

    device = detect_device()
    broad_results = json.loads(BROAD_RESULTS_JSON.read_text(encoding="utf-8"))
    first_broad_params = make_params(broad_results["optimized_params"], device)
    first_frequent_params = make_params(
        {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
            "min_child_weight": 10,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "gamma": 0,
        },
        device,
    )

    experiments = [
        ("json119_vo5_cap30_first_broad_patient", json_features, json_feature_columns, first_broad_params, 5, 30, "patient"),
        ("enhanced140_vo5_cap30_first_broad_patient", enhanced_features, enhanced_feature_columns, first_broad_params, 5, 30, "patient"),
        ("json119_vo20_cap10_first_frequent_row", json_features, json_feature_columns, first_frequent_params, 20, 10, "row"),
        ("enhanced140_vo20_cap10_first_frequent_row", enhanced_features, enhanced_feature_columns, first_frequent_params, 20, 10, "row"),
        ("enhanced140_vo20_cap10_first_frequent_patient", enhanced_features, enhanced_feature_columns, first_frequent_params, 20, 10, "patient"),
    ]

    rows: list[dict[str, object]] = []
    for label, frame, columns, params, min_visit_order, gap_cap, split_mode in experiments:
        print(f"Running {label}: features={len(columns)} split={split_mode}")
        result = evaluate_config(frame, columns, params, min_visit_order, gap_cap, split_mode)
        rows.append({"experiment": label, **result})
        print(f"  R2={result['r2_mean']:.4f} +/- {result['r2_std']:.4f}; MAE={result['mae_mean']:.3f}")

    summary = pd.DataFrame(rows)
    csv_path = OUT_DIR / "visit_order_first_version_audit.csv"
    json_path = OUT_DIR / "visit_order_first_version_audit.json"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "source_files": {
                    "train_data": str(TRAIN_CSV),
                    "history_features": str(HISTORY_FEATURES_CSV),
                    "broad_results": str(BROAD_RESULTS_JSON),
                },
                "device": device,
                "json_feature_count": len(json_feature_columns),
                "enhanced_feature_count": len(enhanced_feature_columns),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()