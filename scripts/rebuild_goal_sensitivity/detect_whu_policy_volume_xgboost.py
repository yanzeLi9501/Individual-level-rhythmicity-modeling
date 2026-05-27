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
POLICY_WEEK_START = POLICY_DATE.to_period("W-SUN").start_time
FIRST_FULL_POST_WEEK = pd.Timestamp("2018-09-03")
ANALYSIS_START = pd.Timestamp("2016-01-04")
ANALYSIS_END = pd.Timestamp("2019-12-30")


def load_weekly_volume() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, usecols=["日期", "病案号"], low_memory=False)
    df["date"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        df.groupby("week")
        .agg(
            n_admissions=("date", "size"),
            n_patients=("病案号", lambda s: s.dropna().nunique()),
        )
        .sort_index()
    )
    full_weeks = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
    weekly = weekly.reindex(full_weeks, fill_value=0)
    weekly.index.name = "week"
    return weekly.reset_index()


def add_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    out["t"] = np.arange(len(out), dtype=float)
    iso_week = out["week"].dt.isocalendar().week.astype(float)
    out["week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
    out["week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
    out["month"] = out["week"].dt.month
    out["quarter"] = out["week"].dt.quarter
    out["is_spring_festival_band"] = out["month"].isin([1, 2]).astype(int)
    out["is_oct_holiday_band"] = out["month"].eq(10).astype(int)
    for lag in [1, 2, 4, 8, 13]:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    for width in [4, 8, 13]:
        shifted = out[target].shift(1)
        out[f"{target}_roll{width}_mean"] = shifted.rolling(width).mean()
        out[f"{target}_roll{width}_std"] = shifted.rolling(width).std()
    return out.dropna().reset_index(drop=True)


def model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        max_depth=2,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=4,
        reg_lambda=8.0,
        reg_alpha=0.2,
        random_state=42,
        n_jobs=4,
    )


def out_of_fold_threshold(train: pd.DataFrame, features: list[str], target: str) -> dict:
    n_splits = min(5, max(2, len(train) // 20))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    residual_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        est = model()
        est.fit(train.loc[train_idx, features], train.loc[train_idx, target])
        pred = est.predict(train.loc[valid_idx, features])
        valid = train.loc[valid_idx, ["week", target]].copy()
        valid["fold"] = fold
        valid["pred"] = pred
        valid["residual"] = valid[target] - valid["pred"]
        residual_rows.append(valid)
    residuals = pd.concat(residual_rows, ignore_index=True)
    residuals = residuals.sort_values("week").reset_index(drop=True)
    positive = residuals["residual"].clip(lower=0)
    roll4 = residuals["residual"].rolling(4, min_periods=4).mean().dropna()
    roll8 = residuals["residual"].rolling(8, min_periods=8).mean().dropna()
    threshold = float(max(np.quantile(positive, 0.95), positive.mean() + 2.0 * positive.std(ddof=1)))
    return {
        "threshold": threshold,
        "oof_mae": float(mean_absolute_error(residuals[target], residuals["pred"])),
        "oof_r2": float(r2_score(residuals[target], residuals["pred"])),
        "oof_residual_mean": float(residuals["residual"].mean()),
        "oof_residual_std": float(residuals["residual"].std(ddof=1)),
        "oof_positive_residual_q95": float(np.quantile(positive, 0.95)),
        "oof_positive_residual_mean_plus_2sd": float(positive.mean() + 2.0 * positive.std(ddof=1)),
        "oof_residual_roll4_mean_q95": float(np.quantile(roll4, 0.95)),
        "oof_residual_roll8_mean_q95": float(np.quantile(roll8, 0.95)),
        "oof_rows": int(len(residuals)),
    }


def summarize_windows(frame: pd.DataFrame, target: str) -> dict:
    windows = {
        "last_8_weeks_before_policy_week": (
            pd.Timestamp("2018-07-02"),
            pd.Timestamp("2018-08-20"),
        ),
        "first_8_full_post_policy_weeks": (
            pd.Timestamp("2018-09-03"),
            pd.Timestamp("2018-10-22"),
        ),
        "next_8_post_policy_weeks": (
            pd.Timestamp("2018-10-29"),
            pd.Timestamp("2018-12-17"),
        ),
        "post_policy_through_2019": (
            FIRST_FULL_POST_WEEK,
            ANALYSIS_END,
        ),
    }
    out = {}
    for name, (start, end) in windows.items():
        mask = (frame["week"] >= start) & (frame["week"] <= end)
        values = frame.loc[mask, target]
        out[name] = {
            "start": str(start.date()),
            "end": str(end.date()),
            "n_weeks": int(values.shape[0]),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": int(values.min()),
            "max": int(values.max()),
        }
    pre = out["last_8_weeks_before_policy_week"]["mean"]
    for name in [
        "first_8_full_post_policy_weeks",
        "next_8_post_policy_weeks",
        "post_policy_through_2019",
    ]:
        out[name]["mean_delta_vs_last_8_pre"] = out[name]["mean"] - pre
        out[name]["mean_ratio_vs_last_8_pre"] = out[name]["mean"] / pre
    return out


def run_target(weekly: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    scoped = weekly[(weekly["week"] >= ANALYSIS_START) & (weekly["week"] <= ANALYSIS_END)].copy()
    featured = add_features(scoped, target)
    feature_cols = [
        c
        for c in featured.columns
        if c not in {"week", "n_admissions", "n_patients"}
    ]
    train = featured[featured["week"] < POLICY_WEEK_START].reset_index(drop=True)
    monitor = featured[featured["week"] >= POLICY_WEEK_START].reset_index(drop=True)

    thresh = out_of_fold_threshold(train, feature_cols, target)
    est = model()
    est.fit(train[feature_cols], train[target])
    result = featured[["week", "n_admissions", "n_patients"]].copy()
    result[f"{target}_pred"] = est.predict(featured[feature_cols])
    result[f"{target}_residual"] = result[target] - result[f"{target}_pred"]
    result[f"{target}_residual_roll4_mean"] = (
        result[f"{target}_residual"].rolling(4, min_periods=4).mean()
    )
    result[f"{target}_residual_roll8_mean"] = (
        result[f"{target}_residual"].rolling(8, min_periods=8).mean()
    )
    result[f"{target}_alert"] = (
        (result["week"] >= FIRST_FULL_POST_WEEK)
        & (result[f"{target}_residual"] > thresh["threshold"])
    )
    result[f"{target}_roll4_shift_alert"] = (
        (result["week"] >= FIRST_FULL_POST_WEEK)
        & (result[f"{target}_residual_roll4_mean"] > thresh["oof_residual_roll4_mean_q95"])
    )
    result[f"{target}_roll8_shift_alert"] = (
        (result["week"] >= FIRST_FULL_POST_WEEK)
        & (result[f"{target}_residual_roll8_mean"] > thresh["oof_residual_roll8_mean_q95"])
    )
    post = result[result["week"] >= FIRST_FULL_POST_WEEK].copy()
    alerts = post[post[f"{target}_alert"]].copy()
    if alerts.empty:
        first_alert = None
    else:
        first = alerts.iloc[0]
        first_alert = {
            "week": str(first["week"].date()),
            "days_after_policy_date": int((first["week"] - POLICY_DATE).days),
            "observed": int(first[target]),
            "predicted": float(first[f"{target}_pred"]),
            "residual": float(first[f"{target}_residual"]),
        }
    shift_alerts = {}
    for width in [4, 8]:
        col = f"{target}_roll{width}_shift_alert"
        alert_rows = post[post[col]].copy()
        if alert_rows.empty:
            shift_alerts[f"roll{width}"] = None
        else:
            first = alert_rows.iloc[0]
            shift_alerts[f"roll{width}"] = {
                "week": str(first["week"].date()),
                "days_after_policy_date": int((first["week"] - POLICY_DATE).days),
                "observed": int(first[target]),
                "predicted": float(first[f"{target}_pred"]),
                "residual": float(first[f"{target}_residual"]),
                f"residual_roll{width}_mean": float(first[f"{target}_residual_roll{width}_mean"]),
            }
    top = (
        post.assign(abs_rank_residual=post[f"{target}_residual"])
        .sort_values(f"{target}_residual", ascending=False)
        .head(10)
    )
    summary = {
        "target": target,
        "analysis_start": str(ANALYSIS_START.date()),
        "analysis_end": str(ANALYSIS_END.date()),
        "policy_date": str(POLICY_DATE.date()),
        "policy_week_start": str(POLICY_WEEK_START.date()),
        "first_full_post_policy_week": str(FIRST_FULL_POST_WEEK.date()),
        "n_training_weeks": int(len(train)),
        "n_monitor_weeks": int(len(monitor)),
        "thresholding": thresh,
        "in_sample_train_mae": float(mean_absolute_error(train[target], est.predict(train[feature_cols]))),
        "in_sample_train_r2": float(r2_score(train[target], est.predict(train[feature_cols]))),
        "first_alert": first_alert,
        "first_sustained_shift_alerts": shift_alerts,
        "n_alert_weeks_post_policy_through_2019": int(result[f"{target}_alert"].sum()),
        "n_roll4_shift_alert_weeks_post_policy_through_2019": int(
            result[f"{target}_roll4_shift_alert"].sum()
        ),
        "n_roll8_shift_alert_weeks_post_policy_through_2019": int(
            result[f"{target}_roll8_shift_alert"].sum()
        ),
        "top_positive_residual_weeks": [
            {
                "week": str(row["week"].date()),
                "observed": int(row[target]),
                "predicted": float(row[f"{target}_pred"]),
                "residual": float(row[f"{target}_residual"]),
                "alert": bool(row[f"{target}_alert"]),
            }
            for _, row in top.iterrows()
        ],
        "window_summary": summarize_windows(scoped, target),
    }
    return result[
        [
            "week",
            f"{target}_pred",
            f"{target}_residual",
            f"{target}_residual_roll4_mean",
            f"{target}_residual_roll8_mean",
            f"{target}_alert",
            f"{target}_roll4_shift_alert",
            f"{target}_roll8_shift_alert",
        ]
    ], summary


def save_plot(results: pd.DataFrame, summaries: dict) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, target, label in [
        (axes[0], "n_admissions", "Weekly admissions"),
        (axes[1], "n_patients", "Weekly unique patients"),
    ]:
        pred_col = f"{target}_pred"
        residual_col = f"{target}_residual"
        alert_col = f"{target}_alert"
        ax.plot(results["week"], results[target], color="#263238", lw=1.5, label="Observed")
        ax.plot(results["week"], results[pred_col], color="#1976d2", lw=1.3, label="XGBoost expected")
        alert_rows = results[results[alert_col]]
        shift_rows = results[results[f"{target}_roll4_shift_alert"]]
        ax.scatter(
            alert_rows["week"],
            alert_rows[target],
            s=42,
            color="#d32f2f",
            zorder=3,
            label="Positive-residual alert",
        )
        ax.scatter(
            shift_rows["week"],
            shift_rows[target],
            s=58,
            facecolors="none",
            edgecolors="#7b1fa2",
            marker="s",
            linewidths=1.2,
            zorder=3,
            label="4-week shift alert",
        )
        ax.axvline(POLICY_DATE, color="#6a1b9a", lw=1.2, ls="--", label="2018-09-01 policy")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#d9d9d9", lw=0.6)
        ax2 = ax.twinx()
        ax2.plot(results["week"], results[residual_col], color="#f57c00", alpha=0.35, lw=0.9)
        ax2.axhline(
            summaries[target]["thresholding"]["threshold"],
            color="#f57c00",
            alpha=0.45,
            lw=0.9,
            ls=":",
        )
        ax2.set_ylabel("Residual")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    axes[1].set_xlabel("Week start")
    fig.suptitle("WHU-only XGBoost detection of the 2018-09-01 policy-related volume shift", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "whu_policy_xgb_volume_detection.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    weekly = load_weekly_volume()
    weekly = weekly[(weekly["week"] >= ANALYSIS_START) & (weekly["week"] <= ANALYSIS_END)].copy()
    results = weekly[["week", "n_admissions", "n_patients"]].reset_index(drop=True)
    summaries = {}
    top_rows = []
    for target in ["n_admissions", "n_patients"]:
        pred, summary = run_target(weekly, target)
        summaries[target] = summary
        results = results.merge(pred, on="week", how="left")
        for alert_col in [
            f"{target}_alert",
            f"{target}_roll4_shift_alert",
            f"{target}_roll8_shift_alert",
        ]:
            results[alert_col] = results[alert_col].astype("boolean").fillna(False).astype(bool)
        for row in summary["top_positive_residual_weeks"]:
            top_rows.append({"target": target, **row})
    results.to_csv(OUTPUT_DIR / "whu_policy_xgb_volume_detection_weekly.csv", index=False)
    pd.DataFrame(top_rows).to_csv(
        OUTPUT_DIR / "whu_policy_xgb_volume_detection_top_alerts.csv",
        index=False,
    )
    plot_path = save_plot(results, summaries)
    summaries["_meta"] = {
        "input_csv": str(INPUT_CSV),
        "weekly_csv": str(OUTPUT_DIR / "whu_policy_xgb_volume_detection_weekly.csv"),
        "top_alerts_csv": str(OUTPUT_DIR / "whu_policy_xgb_volume_detection_top_alerts.csv"),
        "figure": str(plot_path),
        "method": (
            "WHU-only weekly volume series; XGBoost trained only on weeks before the "
            "2018-09-01 policy week; post-policy detection uses positive forecast "
            "residuals above a pre-policy rolling-origin out-of-fold threshold."
        ),
    }
    with open(OUTPUT_DIR / "whu_policy_xgb_volume_detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
