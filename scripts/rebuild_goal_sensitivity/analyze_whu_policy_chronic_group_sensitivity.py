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
LGDI_WEEKLY = ROOT / "analysis_outputs" / "lgdi_whu_rolling4_weekly.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"

POLICY_DATE = pd.Timestamp("2018-09-01")
POLICY_WEEK_START = POLICY_DATE.to_period("W-SUN").start_time
FIRST_FULL_POST_WEEK = pd.Timestamp("2018-09-03")

GROUPS = [
    "Cardiovascular",
    "Hypertension",
    "Diabetes",
    "Cerebrovascular",
    "Renal",
    "Respiratory",
]

GROUP_LABELS = {
    "Cardiovascular": "Cardiovascular",
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes",
    "Cerebrovascular": "Cerebrovascular",
    "Renal": "Renal",
    "Respiratory": "Respiratory",
}

COMORBIDITY_PATTERNS = {
    "Cardiovascular": r"冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病",
    "Hypertension": r"高血压",
    "Diabetes": r"糖尿病|血糖",
    "Cerebrovascular": r"脑梗|脑出血|脑血管|脑卒中|中风|腔隙性",
    "Renal": r"肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏",
    "Respiratory": r"肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染|呼吸道感染",
}

WINDOWS_WEEKLY = {
    "pre8": (pd.Timestamp("2018-07-02"), pd.Timestamp("2018-08-20")),
    "transition8": (pd.Timestamp("2018-09-03"), pd.Timestamp("2018-10-22")),
    "postrise8": (pd.Timestamp("2018-10-29"), pd.Timestamp("2018-12-17")),
    "post_policy_through_2019": (pd.Timestamp("2018-09-03"), pd.Timestamp("2019-12-30")),
}

WINDOWS_DAILY = {
    "pre28": (pd.Timestamp("2018-08-04"), pd.Timestamp("2018-08-31")),
    "post28": (pd.Timestamp("2018-09-01"), pd.Timestamp("2018-09-28")),
    "november2018": (pd.Timestamp("2018-11-01"), pd.Timestamp("2018-11-30")),
    "post_policy_through_2019": (pd.Timestamp("2018-09-01"), pd.Timestamp("2019-12-31")),
}


def xgb_model() -> XGBRegressor:
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


def add_weekly_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    out["t"] = np.arange(len(out), dtype=float)
    iso_week = out["window_anchor"].dt.isocalendar().week.astype(float)
    out["week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
    out["week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
    out["month"] = out["window_anchor"].dt.month
    out["quarter"] = out["window_anchor"].dt.quarter
    out["is_oct_holiday_band"] = out["month"].eq(10).astype(int)
    out["is_jan_feb_band"] = out["month"].isin([1, 2]).astype(int)
    for lag in [1, 2, 4, 8, 13]:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    shifted = out[target].shift(1)
    for width in [4, 8, 13]:
        out[f"{target}_roll{width}_mean"] = shifted.rolling(width).mean()
        out[f"{target}_roll{width}_std"] = shifted.rolling(width).std()
    return out.dropna().reset_index(drop=True)


def group_xgb_detection(weekly: pd.DataFrame, group: str) -> dict:
    target = f"n_{group}"
    use = weekly[["window_anchor", target]].dropna().copy()
    featured = add_weekly_features(use, target)
    feature_cols = [c for c in featured.columns if c not in {"window_anchor", target}]
    train = featured[featured["window_anchor"] < POLICY_WEEK_START].reset_index(drop=True)
    monitor = featured[featured["window_anchor"] >= FIRST_FULL_POST_WEEK].reset_index(drop=True)

    splitter = TimeSeriesSplit(n_splits=min(5, max(2, len(train) // 20)))
    oof_chunks = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        est = xgb_model()
        est.fit(train.loc[train_idx, feature_cols], train.loc[train_idx, target])
        valid = train.loc[valid_idx, ["window_anchor", target]].copy()
        valid["fold"] = fold
        valid["pred"] = est.predict(train.loc[valid_idx, feature_cols])
        valid["residual"] = valid[target] - valid["pred"]
        oof_chunks.append(valid)
    oof = pd.concat(oof_chunks, ignore_index=True).sort_values("window_anchor")
    positive = oof["residual"].clip(lower=0)
    point_threshold = float(
        max(np.quantile(positive, 0.95), positive.mean() + 2.0 * positive.std(ddof=1))
    )
    roll4_threshold = float(np.quantile(oof["residual"].rolling(4, min_periods=4).mean().dropna(), 0.95))

    est = xgb_model()
    est.fit(train[feature_cols], train[target])
    scored = featured[["window_anchor", target]].copy()
    scored["pred"] = est.predict(featured[feature_cols])
    scored["residual"] = scored[target] - scored["pred"]
    scored["residual_roll4_mean"] = scored["residual"].rolling(4, min_periods=4).mean()
    scored["point_alert"] = (
        (scored["window_anchor"] >= FIRST_FULL_POST_WEEK)
        & (scored["residual"] > point_threshold)
    )
    scored["roll4_alert"] = (
        (scored["window_anchor"] >= FIRST_FULL_POST_WEEK)
        & (scored["residual_roll4_mean"] > roll4_threshold)
    )

    post = scored[scored["window_anchor"] >= FIRST_FULL_POST_WEEK]

    def first(col: str) -> dict | None:
        alerts = post[post[col]]
        if alerts.empty:
            return None
        row = alerts.iloc[0]
        return {
            "week": str(row["window_anchor"].date()),
            "days_after_policy_date": int((row["window_anchor"] - POLICY_DATE).days),
            "observed": float(row[target]),
            "predicted": float(row["pred"]),
            "residual": float(row["residual"]),
            "residual_roll4_mean": float(row["residual_roll4_mean"])
            if pd.notna(row["residual_roll4_mean"])
            else None,
        }

    return {
        "group": group,
        "point_threshold": point_threshold,
        "roll4_threshold": roll4_threshold,
        "oof_mae": float(mean_absolute_error(oof[target], oof["pred"])),
        "oof_r2": float(r2_score(oof[target], oof["pred"])),
        "first_point_alert": first("point_alert"),
        "first_roll4_alert": first("roll4_alert"),
        "n_point_alert_weeks": int(scored["point_alert"].sum()),
        "n_roll4_alert_weeks": int(scored["roll4_alert"].sum()),
    }


def summarize_lgdi_weekly() -> tuple[pd.DataFrame, dict]:
    weekly = pd.read_csv(LGDI_WEEKLY, parse_dates=["window_anchor"])
    rows = []
    summary: dict[str, dict] = {}
    for group in GROUPS:
        row: dict[str, object] = {"group": group}
        for name, (start, end) in WINDOWS_WEEKLY.items():
            sub = weekly[(weekly["window_anchor"] >= start) & (weekly["window_anchor"] <= end)]
            n_col = f"n_{group}"
            score_col = f"score_{group}"
            row[f"{name}_n_mean"] = float(sub[n_col].mean())
            row[f"{name}_n_median"] = float(sub[n_col].median())
            row[f"{name}_share_mean"] = float((sub[n_col] / sub["n_admissions"]).mean())
            row[f"{name}_score_mean"] = float(sub[score_col].mean())
        pre = float(row["pre8_n_mean"])
        post = float(row["postrise8_n_mean"])
        transition = float(row["transition8_n_mean"])
        row["transition8_delta_vs_pre"] = transition - pre
        row["transition8_ratio_vs_pre"] = transition / pre if pre else np.nan
        row["postrise8_delta_vs_pre"] = post - pre
        row["postrise8_ratio_vs_pre"] = post / pre if pre else np.nan
        row["postrise8_share_delta_pp"] = (
            float(row["postrise8_share_mean"]) - float(row["pre8_share_mean"])
        ) * 100
        row["postrise8_score_delta_vs_pre"] = (
            float(row["postrise8_score_mean"]) - float(row["pre8_score_mean"])
        )
        baseline = weekly[
            (weekly["window_anchor"] >= pd.Timestamp("2016-01-04"))
            & (weekly["window_anchor"] < POLICY_WEEK_START)
        ][f"n_{group}"].dropna()
        row["postrise8_delta_baseline_sd"] = (
            (post - float(baseline.mean())) / float(baseline.std(ddof=1))
            if len(baseline) > 1 and baseline.std(ddof=1) > 0
            else np.nan
        )
        rows.append(row)
        summary[group] = group_xgb_detection(weekly, group)

    table = pd.DataFrame(rows)
    table["rank_by_postrise_ratio"] = table["postrise8_ratio_vs_pre"].rank(
        ascending=False, method="min"
    ).astype(int)
    table["rank_by_share_delta"] = table["postrise8_share_delta_pp"].rank(
        ascending=False, method="min"
    ).astype(int)
    table["rank_by_baseline_sd_delta"] = table["postrise8_delta_baseline_sd"].rank(
        ascending=False, method="min"
    ).astype(int)
    return table.sort_values("postrise8_ratio_vs_pre", ascending=False), summary


def load_daily_keyword_groups() -> pd.DataFrame:
    usecols = [
        "日期",
        "诊断文本",
        "EMR_初步诊断",
        "EMR_既往史",
        "EMR_出院记录",
        "EMR_主诉",
    ]
    raw = pd.read_csv(INPUT_CSV, usecols=lambda c: c in usecols, low_memory=False)
    raw["date"] = pd.to_datetime(raw["日期"], errors="coerce").dt.floor("D")
    components = []
    for col in ["诊断文本", "EMR_初步诊断", "EMR_既往史", "EMR_出院记录", "EMR_主诉"]:
        if col in raw.columns:
            components.append(raw[col].fillna("").astype(str))
    text = components[0]
    for extra in components[1:]:
        text = text + " " + extra
    raw = raw.dropna(subset=["date"]).copy()
    raw["_diagnosis_text"] = text.loc[raw.index]
    daily = pd.DataFrame(index=pd.date_range(raw["date"].min(), raw["date"].max(), freq="D"))
    daily.index.name = "date"
    daily["n_rows_with_date"] = raw.groupby("date").size()
    for group, pattern in COMORBIDITY_PATTERNS.items():
        raw[group] = raw["_diagnosis_text"].str.contains(pattern, case=False, regex=True, na=False)
        daily[f"n_{group}"] = raw.loc[raw[group]].groupby("date").size()
    daily = daily.fillna(0).reset_index()
    return daily


def summarize_daily_keyword() -> pd.DataFrame:
    daily = load_daily_keyword_groups()
    rows = []
    for group in GROUPS:
        row: dict[str, object] = {"group": group}
        for name, (start, end) in WINDOWS_DAILY.items():
            sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
            row[f"{name}_n_mean"] = float(sub[f"n_{group}"].mean())
            row[f"{name}_share_mean"] = float((sub[f"n_{group}"] / sub["n_rows_with_date"]).mean())
        pre = float(row["pre28_n_mean"])
        post = float(row["november2018_n_mean"])
        row["post28_delta_vs_pre"] = float(row["post28_n_mean"]) - pre
        row["post28_ratio_vs_pre"] = float(row["post28_n_mean"]) / pre if pre else np.nan
        row["november_delta_vs_pre"] = post - pre
        row["november_ratio_vs_pre"] = post / pre if pre else np.nan
        row["november_share_delta_pp"] = (
            float(row["november2018_share_mean"]) - float(row["pre28_share_mean"])
        ) * 100
        rows.append(row)
    return pd.DataFrame(rows).sort_values("november_ratio_vs_pre", ascending=False)


def plot_weekly_summary(table: pd.DataFrame) -> Path:
    plot = table.sort_values("postrise8_ratio_vs_pre", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#1976d2" if v >= 0 else "#8d8d8d" for v in plot["postrise8_share_delta_pp"]]
    ax.barh(y, (plot["postrise8_ratio_vs_pre"] - 1.0) * 100, color=colors, height=0.62)
    ax.axvline(0, color="#444444", lw=0.8)
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            (row["postrise8_ratio_vs_pre"] - 1.0) * 100 + 1,
            i,
            f"+{row['postrise8_delta_vs_pre']:.0f}; share {row['postrise8_share_delta_pp']:+.1f} pp",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["group"])
    ax.set_xlabel("Mean 4-week rolling group count increase, Oct 29-Dec 17 vs Jul 2-Aug 20 (%)")
    ax.set_title("WHU chronic-disease group sensitivity after the 2018-09-01 policy date")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    fig.tight_layout()
    out = OUTPUT_DIR / "whu_policy_chronic_group_sensitivity.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    weekly_table, xgb_summary = summarize_lgdi_weekly()
    daily_table = summarize_daily_keyword()

    weekly_out = OUTPUT_DIR / "whu_policy_chronic_group_sensitivity_rolling4.csv"
    daily_out = OUTPUT_DIR / "whu_policy_chronic_group_sensitivity_daily_keyword.csv"
    summary_out = OUTPUT_DIR / "whu_policy_chronic_group_sensitivity_summary.json"
    weekly_table.to_csv(weekly_out, index=False)
    daily_table.to_csv(daily_out, index=False)
    fig_path = plot_weekly_summary(weekly_table)

    summary = {
        "policy_date": str(POLICY_DATE.date()),
        "primary_table": str(weekly_out),
        "daily_keyword_crosscheck": str(daily_out),
        "figure": str(fig_path),
        "weekly_windows": {
            k: {"start": str(v[0].date()), "end": str(v[1].date())}
            for k, v in WINDOWS_WEEKLY.items()
        },
        "daily_windows": {
            k: {"start": str(v[0].date()), "end": str(v[1].date())}
            for k, v in WINDOWS_DAILY.items()
        },
        "xgboost_group_count_detection": xgb_summary,
        "top_groups_by_rolling4_postrise_ratio": weekly_table[
            [
                "group",
                "pre8_n_mean",
                "transition8_n_mean",
                "postrise8_n_mean",
                "postrise8_delta_vs_pre",
                "postrise8_ratio_vs_pre",
                "postrise8_share_delta_pp",
                "postrise8_score_delta_vs_pre",
                "postrise8_delta_baseline_sd",
            ]
        ].to_dict(orient="records"),
        "top_groups_by_daily_keyword_november_ratio": daily_table[
            [
                "group",
                "pre28_n_mean",
                "post28_n_mean",
                "november2018_n_mean",
                "november_delta_vs_pre",
                "november_ratio_vs_pre",
                "november_share_delta_pp",
            ]
        ].to_dict(orient="records"),
        "interpretation_note": (
            "The primary ranking uses the established LGDI WHU 4-week rolling group counts, "
            "which preserve the six chronic-disease groups used in the manuscript. The daily "
            "keyword table is a cross-check from raw WHU diagnosis text and should be treated "
            "as approximate because many original ICD fields are blank or inconsistent."
        ),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
