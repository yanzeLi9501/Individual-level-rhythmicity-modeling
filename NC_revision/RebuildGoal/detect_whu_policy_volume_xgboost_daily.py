from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
READMISSION_ROOT = ROOT.parents[4]
INPUT_CSV = READMISSION_ROOT / "all_admissions.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"

POLICY_DATE = pd.Timestamp("2018-09-01")
ANALYSIS_START = pd.Timestamp("2016-01-01")
ANALYSIS_END = pd.Timestamp("2019-12-31")


def load_daily_volume() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, usecols=["日期", "病案号"], low_memory=False)
    df["date"] = pd.to_datetime(df["日期"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"])
    daily = (
        df.groupby("date")
        .agg(
            n_admissions=("date", "size"),
            n_patients=("病案号", lambda s: s.dropna().nunique()),
        )
        .sort_index()
    )
    full_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_days, fill_value=0)
    daily.index.name = "date"
    return daily.reset_index()


def add_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    out["t"] = np.arange(len(out), dtype=float)
    day = out["date"].dt.dayofweek.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * day / 7)
    out["dow_cos"] = np.cos(2 * np.pi * day / 7)
    iso_week = out["date"].dt.isocalendar().week.astype(float)
    out["week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
    out["week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["date"].dt.dayofweek.isin([5, 6]).astype(int)
    out["is_oct_holiday_band"] = (
        (out["date"].dt.month == 10) & (out["date"].dt.day <= 7)
    ).astype(int)
    out["is_jan_feb_band"] = out["date"].dt.month.isin([1, 2]).astype(int)
    for lag in [1, 2, 3, 7, 14, 28]:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    shifted = out[target].shift(1)
    for width in [7, 14, 28]:
        out[f"{target}_roll{width}_mean"] = shifted.rolling(width).mean()
        out[f"{target}_roll{width}_std"] = shifted.rolling(width).std()
    return out.dropna().reset_index(drop=True)


def model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=550,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=8,
        reg_lambda=8.0,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=4,
    )


def thresholds(train: pd.DataFrame, features: list[str], target: str) -> dict:
    splitter = TimeSeriesSplit(n_splits=5)
    chunks = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        est = model()
        est.fit(train.loc[train_idx, features], train.loc[train_idx, target])
        pred = est.predict(train.loc[valid_idx, features])
        valid = train.loc[valid_idx, ["date", target]].copy()
        valid["fold"] = fold
        valid["pred"] = pred
        valid["residual"] = valid[target] - valid["pred"]
        chunks.append(valid)
    residuals = pd.concat(chunks, ignore_index=True).sort_values("date")
    positive = residuals["residual"].clip(lower=0)
    roll14 = residuals["residual"].rolling(14, min_periods=14).mean().dropna()
    roll28 = residuals["residual"].rolling(28, min_periods=28).mean().dropna()
    return {
        "point_positive_threshold": float(
            max(np.quantile(positive, 0.99), positive.mean() + 3.0 * positive.std(ddof=1))
        ),
        "roll14_mean_threshold": float(np.quantile(roll14, 0.95)),
        "roll28_mean_threshold": float(np.quantile(roll28, 0.95)),
        "oof_mae": float(mean_absolute_error(residuals[target], residuals["pred"])),
        "oof_r2": float(r2_score(residuals[target], residuals["pred"])),
        "oof_residual_mean": float(residuals["residual"].mean()),
        "oof_residual_std": float(residuals["residual"].std(ddof=1)),
        "oof_rows": int(len(residuals)),
    }


def summarize_windows(frame: pd.DataFrame, target: str) -> dict:
    windows = {
        "last_28_days_before_policy": (pd.Timestamp("2018-08-04"), pd.Timestamp("2018-08-31")),
        "first_28_days_after_policy": (pd.Timestamp("2018-09-01"), pd.Timestamp("2018-09-28")),
        "november_2018": (pd.Timestamp("2018-11-01"), pd.Timestamp("2018-11-30")),
        "post_policy_through_2019": (POLICY_DATE, ANALYSIS_END),
    }
    out = {}
    pre_mean = None
    for name, (start, end) in windows.items():
        values = frame.loc[(frame["date"] >= start) & (frame["date"] <= end), target]
        out[name] = {
            "start": str(start.date()),
            "end": str(end.date()),
            "n_days": int(values.shape[0]),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": int(values.min()),
            "max": int(values.max()),
        }
        if name == "last_28_days_before_policy":
            pre_mean = out[name]["mean"]
    for name in ["first_28_days_after_policy", "november_2018", "post_policy_through_2019"]:
        out[name]["mean_delta_vs_last_28_pre"] = out[name]["mean"] - pre_mean
        out[name]["mean_ratio_vs_last_28_pre"] = out[name]["mean"] / pre_mean
    return out


def run_target(daily: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    scoped = daily[(daily["date"] >= ANALYSIS_START) & (daily["date"] <= ANALYSIS_END)].copy()
    featured = add_features(scoped, target)
    feature_cols = [c for c in featured.columns if c not in {"date", "n_admissions", "n_patients"}]
    train = featured[featured["date"] < POLICY_DATE].reset_index(drop=True)
    monitor = featured[featured["date"] >= POLICY_DATE].reset_index(drop=True)
    thresh = thresholds(train, feature_cols, target)
    est = model()
    est.fit(train[feature_cols], train[target])
    result = featured[["date", "n_admissions", "n_patients"]].copy()
    result[f"{target}_pred"] = est.predict(featured[feature_cols])
    result[f"{target}_residual"] = result[target] - result[f"{target}_pred"]
    result[f"{target}_residual_roll14_mean"] = (
        result[f"{target}_residual"].rolling(14, min_periods=14).mean()
    )
    result[f"{target}_residual_roll28_mean"] = (
        result[f"{target}_residual"].rolling(28, min_periods=28).mean()
    )
    result[f"{target}_point_alert"] = (
        (result["date"] >= POLICY_DATE)
        & (result[f"{target}_residual"] > thresh["point_positive_threshold"])
    )
    result[f"{target}_roll14_shift_alert"] = (
        (result["date"] >= POLICY_DATE)
        & (result[f"{target}_residual_roll14_mean"] > thresh["roll14_mean_threshold"])
    )
    result[f"{target}_roll28_shift_alert"] = (
        (result["date"] >= POLICY_DATE)
        & (result[f"{target}_residual_roll28_mean"] > thresh["roll28_mean_threshold"])
    )
    post = result[result["date"] >= POLICY_DATE]

    def first_alert(col: str, residual_col: str) -> dict | None:
        alerts = post[post[col]]
        if alerts.empty:
            return None
        first = alerts.iloc[0]
        return {
            "date": str(first["date"].date()),
            "days_after_policy_date": int((first["date"] - POLICY_DATE).days),
            "observed": int(first[target]),
            "predicted": float(first[f"{target}_pred"]),
            "residual": float(first[f"{target}_residual"]),
            residual_col: float(first[residual_col]),
        }

    summary = {
        "target": target,
        "analysis_start": str(ANALYSIS_START.date()),
        "analysis_end": str(ANALYSIS_END.date()),
        "policy_date": str(POLICY_DATE.date()),
        "n_training_days": int(len(train)),
        "n_monitor_days": int(len(monitor)),
        "thresholding": thresh,
        "in_sample_train_mae": float(mean_absolute_error(train[target], est.predict(train[feature_cols]))),
        "in_sample_train_r2": float(r2_score(train[target], est.predict(train[feature_cols]))),
        "first_point_alert": first_alert(f"{target}_point_alert", f"{target}_residual"),
        "first_roll14_shift_alert": first_alert(
            f"{target}_roll14_shift_alert", f"{target}_residual_roll14_mean"
        ),
        "first_roll28_shift_alert": first_alert(
            f"{target}_roll28_shift_alert", f"{target}_residual_roll28_mean"
        ),
        "n_point_alert_days": int(result[f"{target}_point_alert"].sum()),
        "n_roll14_shift_alert_days": int(result[f"{target}_roll14_shift_alert"].sum()),
        "n_roll28_shift_alert_days": int(result[f"{target}_roll28_shift_alert"].sum()),
        "window_summary": summarize_windows(scoped, target),
    }
    return result[
        [
            "date",
            f"{target}_pred",
            f"{target}_residual",
            f"{target}_residual_roll14_mean",
            f"{target}_residual_roll28_mean",
            f"{target}_point_alert",
            f"{target}_roll14_shift_alert",
            f"{target}_roll28_shift_alert",
        ]
    ], summary


def save_plot(results: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    for ax, target, label in [
        (axes[0], "n_admissions", "Daily admissions"),
        (axes[1], "n_patients", "Daily unique patients"),
    ]:
        ax.plot(results["date"], results[target], color="#263238", lw=0.9, alpha=0.8, label="Observed")
        ax.plot(results["date"], results[f"{target}_pred"], color="#1976d2", lw=1.0, label="XGBoost expected")
        shift = results[results[f"{target}_roll14_shift_alert"]]
        ax.scatter(
            shift["date"],
            shift[target],
            s=16,
            facecolors="none",
            edgecolors="#7b1fa2",
            linewidths=0.9,
            label="14-day shift alert",
        )
        ax.axvline(POLICY_DATE, color="#6a1b9a", lw=1.2, ls="--", label="2018-09-01 policy")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#d9d9d9", lw=0.6)
    axes[0].legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    axes[1].set_xlabel("Date")
    fig.suptitle("WHU-only daily XGBoost detection of the 2018-09-01 policy-related volume shift", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "whu_policy_xgb_daily_volume_detection.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    daily = load_daily_volume()
    daily = daily[(daily["date"] >= ANALYSIS_START) & (daily["date"] <= ANALYSIS_END)].copy()
    results = daily[["date", "n_admissions", "n_patients"]].reset_index(drop=True)
    summaries = {}
    for target in ["n_admissions", "n_patients"]:
        pred, summary = run_target(daily, target)
        summaries[target] = summary
        results = results.merge(pred, on="date", how="left")
        for col in [
            f"{target}_point_alert",
            f"{target}_roll14_shift_alert",
            f"{target}_roll28_shift_alert",
        ]:
            results[col] = results[col].astype("boolean").fillna(False).astype(bool)
    results.to_csv(OUTPUT_DIR / "whu_policy_xgb_daily_volume_detection.csv", index=False)
    fig_path = save_plot(results)
    summaries["_meta"] = {
        "input_csv": str(INPUT_CSV),
        "daily_csv": str(OUTPUT_DIR / "whu_policy_xgb_daily_volume_detection.csv"),
        "figure": str(fig_path),
        "method": (
            "WHU-only daily volume series; XGBoost trained before 2018-09-01; "
            "post-policy detection uses positive residual and rolling residual thresholds "
            "estimated from pre-policy rolling-origin out-of-fold residuals."
        ),
    }
    with open(OUTPUT_DIR / "whu_policy_xgb_daily_volume_detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
