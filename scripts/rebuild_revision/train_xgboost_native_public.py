#!/usr/bin/env python3
"""Train dataset-native XGBoost models for public positive-control cohorts.

Outputs are written under RebuildRevision/outputs/gpu_native_xgb. The script
prefers CUDA when available and records an explicit fallback reason otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 20260521
REFERENCE_COLUMNS = {
    "influenza_reference",
    "viral_pneumonia_reference",
    "covid_reference",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    target: str
    id_columns: tuple[str, ...]


def workspace_paths() -> tuple[Path, Path, Path]:
    rebuild_dir = Path(__file__).resolve().parents[1]
    nc_revision_dir = rebuild_dir.parent
    submit_dir = nc_revision_dir.parent
    return rebuild_dir, nc_revision_dir, submit_dir


def default_datasets(nc_revision_dir: Path) -> list[DatasetConfig]:
    external_dir = nc_revision_dir / "external_positive_control_results"
    return [
        DatasetConfig(
            name="mimic_iv_native_features",
            path=external_dir / "mimic_analysis_admission_features.csv",
            target="influenza_reference",
            id_columns=("hadm_id", "subject_id"),
        ),
        DatasetConfig(
            name="eicu_native_features",
            path=external_dir / "eicu_analysis_stay_features.csv",
            target="viral_pneumonia_reference",
            id_columns=("patientunitstayid", "uniquepid", "patienthealthsystemstayid"),
        ),
        DatasetConfig(
            name="nwicu_native_features",
            path=external_dir / "nwicu_analysis_admission_features.csv",
            target="covid_reference",
            id_columns=("hadm_id", "subject_id"),
        ),
    ]


def coerce_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("int8")
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
    if mapped.notna().all():
        return mapped.astype("int8")
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int8")


def build_feature_matrix(frame: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if config.target not in frame.columns:
        raise KeyError(f"{config.name}: missing target column {config.target}")
    work = frame.copy()
    work[config.target] = coerce_target(work[config.target])
    drop_columns = set(config.id_columns) | REFERENCE_COLUMNS
    feature_columns = [column for column in work.columns if column not in drop_columns]
    features = work[feature_columns].copy()
    for column in list(features.columns):
        if features[column].dtype == bool:
            features[column] = features[column].astype("int8")
        elif not pd.api.types.is_numeric_dtype(features[column]):
            unique_count = features[column].nunique(dropna=True)
            if unique_count <= 64:
                encoded = pd.get_dummies(features[column].astype("string"), prefix=column, dummy_na=True)
                features = pd.concat([features.drop(columns=[column]), encoded], axis=1)
            else:
                features = features.drop(columns=[column])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    features = features.astype("float32")
    target = work[config.target].astype("int8")
    return features, target, list(features.columns)


def make_model(device: str, n_jobs: int, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.0,
        reg_alpha=0.0,
        reg_lambda=3.0,
        gamma=0.0,
        random_state=SEED,
        n_jobs=n_jobs,
        tree_method="hist",
        device=device,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )


def probe_cuda(n_jobs: int) -> tuple[str, str]:
    smi = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if smi.returncode != 0:
        detail = (smi.stderr or smi.stdout or "nvidia-smi returned non-zero").strip()
        return "cpu", f"nvidia-smi probe failed: {detail}"
    probe_x = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [1, 1, 0, 0, 1, 0]}, dtype="float32")
    probe_y = pd.Series([0, 0, 0, 1, 1, 1], dtype="int8")
    try:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=4,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            n_jobs=n_jobs,
            random_state=SEED,
            verbosity=0,
        )
        model.fit(probe_x, probe_y)
        return "cuda", "cuda probe fit succeeded"
    except Exception as exc:  # pragma: no cover - environment-specific
        return "cpu", f"cuda probe failed: {type(exc).__name__}: {exc}"


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    youden = true_positive_rate - false_positive_rate
    best_index = int(np.nanargmax(youden))
    threshold = float(thresholds[best_index])
    if not np.isfinite(threshold):
        return 0.5
    return threshold


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype("int8")
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    tn_count, fp_count, fn_count, tp_count = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "ppv_precision": float(precision),
        "sensitivity_recall": float(recall),
        "specificity": float(tn_count / (tn_count + fp_count)) if (tn_count + fp_count) else float("nan"),
        "f1": float(f1_score),
        "tp": int(tp_count),
        "fp": int(fp_count),
        "tn": int(tn_count),
        "fn": int(fn_count),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        metrics["average_precision"] = float(average_precision_score(y_true, probability))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    metrics["brier"] = float(brier_score_loss(y_true, probability))
    return metrics


def train_one_dataset(config: DatasetConfig, output_dir: Path, requested_device: str, n_jobs: int) -> dict[str, object]:
    started = time.time()
    frame = pd.read_csv(config.path, low_memory=False)
    features, target, feature_names = build_feature_matrix(frame, config)
    positive_count = int(target.sum())
    negative_count = int((target == 0).sum())
    if positive_count < 5 or negative_count < 5:
        raise ValueError(
            f"{config.name}: not enough label balance for training "
            f"(positive={positive_count}, negative={negative_count})"
        )

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=target,
    )
    scale_pos_weight = float((train_y == 0).sum() / max(1, int(train_y.sum())))
    model = make_model(requested_device, n_jobs, scale_pos_weight)
    device_used = requested_device
    fallback_reason = ""
    try:
        model.fit(train_x, train_y)
    except Exception as exc:
        if requested_device != "cuda":
            raise
        fallback_reason = f"cuda training failed: {type(exc).__name__}: {exc}"
        device_used = "cpu"
        model = make_model("cpu", n_jobs, scale_pos_weight)
        model.fit(train_x, train_y)

    train_probability = model.predict_proba(train_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    threshold = choose_threshold(train_y.to_numpy(), train_probability)
    metrics = binary_metrics(test_y.to_numpy(), test_probability, threshold)

    dataset_dir = output_dir / config.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "row_index": test_x.index.to_numpy(),
            "target": test_y.to_numpy(),
            "probability": test_probability,
            "prediction": (test_probability >= threshold).astype("int8"),
        }
    )
    predictions.to_csv(dataset_dir / f"{config.name}_predictions.csv", index=False)
    model.save_model(dataset_dir / f"{config.name}_xgboost_model.json")
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(dataset_dir / f"{config.name}_feature_importance.csv", index=False)

    result: dict[str, object] = {
        "dataset": config.name,
        "input_path": str(config.path),
        "target": config.target,
        "n_rows": int(len(frame)),
        "n_features": int(features.shape[1]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "scale_pos_weight": scale_pos_weight,
        "requested_device": requested_device,
        "device_used": device_used,
        "fallback_reason": fallback_reason,
        "fit_seconds": round(time.time() - started, 3),
        "metrics": metrics,
        "top_features": importance.head(20).to_dict(orient="records"),
        "outputs": {
            "predictions": str(dataset_dir / f"{config.name}_predictions.csv"),
            "model": str(dataset_dir / f"{config.name}_xgboost_model.json"),
            "feature_importance": str(dataset_dir / f"{config.name}_feature_importance.csv"),
        },
    }
    with open(dataset_dir / f"{config.name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuild_dir, nc_revision_dir, submit_dir = workspace_paths()
    output_dir = args.output_dir or (rebuild_dir / "outputs" / "gpu_native_xgb")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = default_datasets(nc_revision_dir)
    if args.datasets:
        selected = set(args.datasets)
        datasets = [config for config in datasets if config.name in selected]

    if args.device == "auto":
        requested_device, device_probe = probe_cuda(args.n_jobs)
    else:
        requested_device, device_probe = args.device, f"device forced to {args.device}"

    summary: dict[str, object] = {
        "generated_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "submit_dir": str(submit_dir),
        "nc_revision_dir": str(nc_revision_dir),
        "output_dir": str(output_dir),
        "n_jobs": args.n_jobs,
        "device_probe": device_probe,
        "requested_device_after_probe": requested_device,
        "datasets": [asdict(config) | {"path": str(config.path)} for config in datasets],
        "results": [],
        "errors": [],
    }
    print(json.dumps({key: summary[key] for key in ["host", "xgboost", "n_jobs", "device_probe"]}, indent=2))

    for config in datasets:
        print(f"\n=== {config.name} ===")
        if not config.path.exists():
            message = f"missing input: {config.path}"
            print(message)
            summary["errors"].append({"dataset": config.name, "error": message})
            continue
        try:
            result = train_one_dataset(config, output_dir, requested_device, args.n_jobs)
            summary["results"].append(result)
            metrics = result["metrics"]
            print(
                f"device={result['device_used']} rows={result['n_rows']} features={result['n_features']} "
                f"AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f} "
                f"PPV={metrics['ppv_precision']:.4f} Sens={metrics['sensitivity_recall']:.4f}"
            )
            if result["fallback_reason"]:
                print(result["fallback_reason"])
        except Exception as exc:  # pragma: no cover - runtime reporting
            message = f"{type(exc).__name__}: {exc}"
            print(message)
            summary["errors"].append({"dataset": config.name, "error": message})

    with open(output_dir / "native_xgboost_public_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    rows = []
    for result in summary["results"]:
        metrics = result["metrics"]
        rows.append(
            {
                "dataset": result["dataset"],
                "target": result["target"],
                "n_rows": result["n_rows"],
                "n_features": result["n_features"],
                "positive_count": result["positive_count"],
                "negative_count": result["negative_count"],
                "device_used": result["device_used"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "ppv_precision": metrics["ppv_precision"],
                "sensitivity_recall": metrics["sensitivity_recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "brier": metrics["brier"],
                "threshold": metrics["threshold"],
                "fit_seconds": result["fit_seconds"],
                "fallback_reason": result["fallback_reason"],
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "native_xgboost_public_summary.csv", index=False)
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()