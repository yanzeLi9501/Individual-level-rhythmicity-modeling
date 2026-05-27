from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis_outputs"
VALIDATION_CSV = ANALYSIS / "lgdi_whu_influenza_validation.csv"
OUT_PREFIX = "alert_sensitivity"

SEED = 20260525
N_BOOT = 5000

GROUPS = [
    "Cardiovascular",
    "Hypertension",
    "Diabetes",
    "Cerebrovascular",
    "Renal",
    "Respiratory",
]
SCORE_COLS = {group: f"score_{group}" for group in GROUPS}
ALERT_LABELS = {
    "alert_s1": "S1 single respiratory",
    "alert_s2": "S2 two-group consensus",
    "alert_s3": "S3 Oct-Apr consensus",
    "alert_s4": "S4 sustained Nov-Mar consensus",
}
THRESHOLD_GRID = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]


def load_validation() -> pd.DataFrame:
    df = pd.read_csv(VALIDATION_CSV, parse_dates=["week_start", "window_anchor"])
    return df.sort_values("week_start").reset_index(drop=True)


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else float("nan")


def metrics(alert: np.ndarray | pd.Series, event: np.ndarray | pd.Series) -> dict[str, float | int]:
    a = np.asarray(alert, dtype=bool)
    e = np.asarray(event, dtype=bool)
    tp = int((a & e).sum())
    fp = int((a & ~e).sum())
    fn = int((~a & e).sum())
    tn = int((~a & ~e).sum())
    sensitivity = safe_div(tp, tp + fn)
    ppv = safe_div(tp, tp + fp)
    far = safe_div(fp, fp + tn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "n_weeks": int(len(a)),
        "event_weeks": int(e.sum()),
        "alerts": int(a.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "false_alarm_rate": far,
        "f1": f1,
    }


def thresholds_from_mask(frame: pd.DataFrame, mask: pd.Series, sd_multiplier: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    valid_train = frame[mask & frame["valid"].astype(bool)].copy()
    for group, col in SCORE_COLS.items():
        values = pd.to_numeric(valid_train[col], errors="coerce").dropna()
        thresholds[group] = float(values.mean() + sd_multiplier * values.std(ddof=1))
    return thresholds


def thresholds_from_training(train: pd.DataFrame, sd_multiplier: float) -> dict[str, float]:
    return thresholds_from_mask(train, pd.Series(True, index=train.index), sd_multiplier)


def baseline_mask_for_fold(df: pd.DataFrame, held_out_season: str) -> pd.Series:
    baseline = df["window_anchor"].dt.year.between(2016, 2018)
    folded = baseline & df["season_id"].ne(held_out_season)
    return folded if int(folded.sum()) >= 20 else baseline


def sustained_consensus_alerts(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    min_groups: int = 2,
    months: tuple[int, ...] = (11, 12, 1, 2, 3),
    persistence_weeks: int = 2,
) -> pd.Series:
    flags = []
    for group, col in SCORE_COLS.items():
        flags.append((pd.to_numeric(frame[col], errors="coerce") >= thresholds[group]).fillna(False).to_numpy())
    consensus = np.vstack(flags).sum(axis=0) >= min_groups
    in_window = frame["week_start"].dt.month.isin(months).to_numpy()
    active = consensus & in_window
    if persistence_weeks <= 1:
        return pd.Series(active, index=frame.index)
    sustained = active.copy()
    for lag in range(1, persistence_weeks):
        sustained &= np.concatenate([np.zeros(lag, dtype=bool), active[:-lag]])
    return pd.Series(sustained, index=frame.index)


def original_rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event = df["flu_event_window"].astype(bool)
    for col, label in ALERT_LABELS.items():
        row = {"analysis": "observed fixed alert column", "strategy": label}
        row.update(metrics(df[col].astype(bool), event))
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_season_out_all_training_weeks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_alerts = pd.Series(False, index=df.index)
    seasons = list(df["season_id"].drop_duplicates())
    for season in seasons:
        train = df[df["season_id"].ne(season)]
        holdout_mask = df["season_id"].eq(season)
        thresholds = thresholds_from_training(train, 1.5)
        alerts = sustained_consensus_alerts(df, thresholds)
        all_alerts.loc[holdout_mask] = alerts.loc[holdout_mask]

        row = {
            "analysis": "LOSO all-nonheld-week S4 recalibration stress test",
            "held_out_season": season,
            "sd_multiplier": 1.5,
            "calibration_weeks": int(train["valid"].astype(bool).sum()),
        }
        row.update(metrics(alerts.loc[holdout_mask], df.loc[holdout_mask, "flu_event_window"]))
        for group in GROUPS:
            row[f"threshold_{group}"] = thresholds[group]
        rows.append(row)

    aggregate = {
        "analysis": "LOSO all-nonheld-week S4 recalibration stress test",
        "held_out_season": "aggregate",
        "sd_multiplier": 1.5,
        "calibration_weeks": "",
    }
    aggregate.update(metrics(all_alerts, df["flu_event_window"]))
    return pd.DataFrame(rows), pd.DataFrame([aggregate])


def leave_one_season_out_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_alerts = pd.Series(False, index=df.index)
    seasons = list(df["season_id"].drop_duplicates())
    for season in seasons:
        holdout_mask = df["season_id"].eq(season)
        calibration_mask = baseline_mask_for_fold(df, season)
        thresholds = thresholds_from_mask(df, calibration_mask, 1.5)
        alerts = sustained_consensus_alerts(df, thresholds)
        all_alerts.loc[holdout_mask] = alerts.loc[holdout_mask]

        row = {
            "analysis": "LOSO baseline-only S4 recalibration",
            "held_out_season": season,
            "sd_multiplier": 1.5,
            "calibration_weeks": int(calibration_mask.sum()),
        }
        row.update(metrics(alerts.loc[holdout_mask], df.loc[holdout_mask, "flu_event_window"]))
        for group in GROUPS:
            row[f"threshold_{group}"] = thresholds[group]
        rows.append(row)

    aggregate = {
        "analysis": "LOSO baseline-only S4 recalibration",
        "held_out_season": "aggregate",
        "sd_multiplier": 1.5,
        "calibration_weeks": "",
    }
    aggregate.update(metrics(all_alerts, df["flu_event_window"]))
    return pd.DataFrame(rows), pd.DataFrame([aggregate])


def metric_ci(values: np.ndarray) -> tuple[float, float, int]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return float("nan"), float("nan"), 0
    low, high = np.nanpercentile(clean, [2.5, 97.5])
    return float(low), float(high), int(len(clean))


def block_bootstrap_alert_ci(df: pd.DataFrame, block_weeks: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + block_weeks)
    n = len(df)
    starts = np.arange(0, n - block_weeks + 1)
    n_blocks = int(np.ceil(n / block_weeks))
    event = df["flu_event_window"].astype(bool).to_numpy()
    rows = []

    for col, label in ALERT_LABELS.items():
        alert = df[col].astype(bool).to_numpy()
        observed = metrics(alert, event)
        boot = {name: np.empty(N_BOOT, dtype=float) for name in ["sensitivity", "ppv", "false_alarm_rate"]}
        for i in range(N_BOOT):
            chosen = rng.choice(starts, size=n_blocks, replace=True)
            idx = np.concatenate([np.arange(start, start + block_weeks) for start in chosen])[:n]
            m = metrics(alert[idx], event[idx])
            for name in boot:
                boot[name][i] = float(m[name])
        row = {
            "strategy": label,
            "block_weeks": block_weeks,
            "n_boot": N_BOOT,
            "alerts": observed["alerts"],
            "tp": observed["tp"],
            "fp": observed["fp"],
            "fn": observed["fn"],
            "tn": observed["tn"],
        }
        for name in ["sensitivity", "ppv", "false_alarm_rate"]:
            low, high, valid_boot = metric_ci(boot[name])
            row[f"{name}_point"] = observed[name]
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
            row[f"{name}_valid_boot"] = valid_boot
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_table(train: pd.DataFrame, full_frame: pd.DataFrame, calibration_mask: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    train_mask = full_frame.index.isin(train.index)
    for sd_multiplier in THRESHOLD_GRID:
        if calibration_mask is None:
            thresholds = thresholds_from_training(train, sd_multiplier)
        else:
            thresholds = thresholds_from_mask(full_frame, calibration_mask, sd_multiplier)
        alerts = sustained_consensus_alerts(full_frame, thresholds)
        row = {"sd_multiplier": sd_multiplier}
        row.update(metrics(alerts.loc[train_mask], full_frame.loc[train_mask, "flu_event_window"]))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_candidate(candidates: pd.DataFrame, objective: str) -> pd.Series:
    ranked = candidates.copy()
    if objective == "max_f1":
        ranked = ranked.sort_values(
            ["f1", "ppv", "sensitivity", "false_alarm_rate", "alerts"],
            ascending=[False, False, False, True, True],
        )
    elif objective == "max_ppv_with_sens_ge_30":
        eligible = ranked[(ranked["sensitivity"] >= 0.30) & (ranked["alerts"] >= 5)].copy()
        if eligible.empty:
            eligible = ranked.copy()
        ranked = eligible.sort_values(
            ["ppv", "sensitivity", "false_alarm_rate", "alerts", "f1"],
            ascending=[False, False, True, True, False],
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return ranked.iloc[0]


def nested_threshold_selection(df: pd.DataFrame, calibration_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    candidate_rows = []
    aggregate_alerts = {
        "max_f1": pd.Series(False, index=df.index),
        "max_ppv_with_sens_ge_30": pd.Series(False, index=df.index),
    }
    seasons = list(df["season_id"].drop_duplicates())
    for season in seasons:
        train = df[df["season_id"].ne(season)]
        holdout_mask = df["season_id"].eq(season)
        calibration_mask = None
        if calibration_mode == "all_nonheld_weeks":
            calibration_mask = None
        elif calibration_mode == "baseline_only":
            calibration_mask = baseline_mask_for_fold(df, season)
        else:
            raise ValueError(f"Unknown calibration mode: {calibration_mode}")

        candidates = candidate_table(train, df, calibration_mask)
        candidates.insert(0, "held_out_season", season)
        candidates.insert(1, "calibration_mode", calibration_mode)
        candidates.insert(
            2,
            "calibration_weeks",
            int(train["valid"].astype(bool).sum()) if calibration_mask is None else int(calibration_mask.sum()),
        )
        candidate_rows.append(candidates)

        for objective in aggregate_alerts:
            selected = choose_candidate(candidates, objective)
            if calibration_mask is None:
                thresholds = thresholds_from_training(train, float(selected["sd_multiplier"]))
            else:
                thresholds = thresholds_from_mask(df, calibration_mask, float(selected["sd_multiplier"]))
            alerts = sustained_consensus_alerts(df, thresholds)
            aggregate_alerts[objective].loc[holdout_mask] = alerts.loc[holdout_mask]

            row = {
                "analysis": "nested threshold selection",
                "calibration_mode": calibration_mode,
                "objective": objective,
                "held_out_season": season,
                "selected_sd_multiplier": float(selected["sd_multiplier"]),
                "calibration_weeks": int(train["valid"].astype(bool).sum()) if calibration_mask is None else int(calibration_mask.sum()),
                "train_alerts": int(selected["alerts"]),
                "train_tp": int(selected["tp"]),
                "train_fp": int(selected["fp"]),
                "train_sensitivity": float(selected["sensitivity"]),
                "train_ppv": float(selected["ppv"]),
                "train_false_alarm_rate": float(selected["false_alarm_rate"]),
                "train_f1": float(selected["f1"]),
            }
            holdout_metrics = metrics(alerts.loc[holdout_mask], df.loc[holdout_mask, "flu_event_window"])
            row.update({f"holdout_{k}": v for k, v in holdout_metrics.items()})
            fold_rows.append(row)

    aggregate_rows = []
    for objective, alerts in aggregate_alerts.items():
        row = {
            "analysis": "nested threshold selection",
            "calibration_mode": calibration_mode,
            "objective": objective,
            "held_out_season": "aggregate",
        }
        row.update(metrics(alerts, df["flu_event_window"]))
        aggregate_rows.append(row)
    return pd.DataFrame(fold_rows), pd.concat(candidate_rows, ignore_index=True), pd.DataFrame(aggregate_rows)


def fmt_pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{100 * value:.1f}%"


def fmt_num(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_summary(
    observed: pd.DataFrame,
    loso_baseline_fold: pd.DataFrame,
    loso_baseline_aggregate: pd.DataFrame,
    loso_all_week_fold: pd.DataFrame,
    loso_all_week_aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nested_fold: pd.DataFrame,
    nested_aggregate: pd.DataFrame,
) -> None:
    s4 = observed[observed["strategy"].eq(ALERT_LABELS["alert_s4"])].iloc[0]
    loso_base = loso_baseline_aggregate.iloc[0]
    loso_all = loso_all_week_aggregate.iloc[0]
    boot4 = bootstrap[(bootstrap["strategy"].eq(ALERT_LABELS["alert_s4"])) & (bootstrap["block_weeks"].eq(4))].iloc[0]
    boot8 = bootstrap[(bootstrap["strategy"].eq(ALERT_LABELS["alert_s4"])) & (bootstrap["block_weeks"].eq(8))].iloc[0]
    nested_base = nested_aggregate[nested_aggregate["calibration_mode"].eq("baseline_only")]
    nested_all = nested_aggregate[nested_aggregate["calibration_mode"].eq("all_nonheld_weeks")]

    lines = [
        "# Additional Alert Sensitivity Analyses",
        "",
        "## Fixed S4 Baseline",
        (
            f"Observed S4 produced {int(s4['alerts'])} alerts, {int(s4['tp'])} true-positive weeks, "
            f"{int(s4['fp'])} false-positive weeks, sensitivity {fmt_pct(float(s4['sensitivity']))}, "
            f"PPV {fmt_pct(float(s4['ppv']))}, and false-alarm rate {fmt_pct(float(s4['false_alarm_rate']))}."
        ),
        "",
        "## Leave-One-Season-Out Fixed-Rule Recalibration",
        (
            f"Baseline-only LOSO, which preserves the manuscript's 2016-2018 calibration design while "
            f"excluding each held-out respiratory season from calibration when applicable, gave "
            f"{int(loso_base['alerts'])} alerts, {int(loso_base['tp'])} true-positive weeks, "
            f"{int(loso_base['fp'])} false-positive weeks, sensitivity {fmt_pct(float(loso_base['sensitivity']))}, "
            f"PPV {fmt_pct(float(loso_base['ppv']))}, and false-alarm rate "
            f"{fmt_pct(float(loso_base['false_alarm_rate']))}."
        ),
        (
            f"A stricter stress test that treated all non-held-out weeks as the calibration distribution gave "
            f"{int(loso_all['alerts'])} alerts, {int(loso_all['tp'])} true-positive weeks, "
            f"{int(loso_all['fp'])} false-positive weeks, sensitivity {fmt_pct(float(loso_all['sensitivity']))}, "
            f"PPV {fmt_pct(float(loso_all['ppv']))}, and false-alarm rate "
            f"{fmt_pct(float(loso_all['false_alarm_rate']))}."
        ),
        "",
        "## Moving-Block Bootstrap Alert-Metric CI",
        (
            "For S4, the 4-week block bootstrap 95% intervals were "
            f"sensitivity {fmt_pct(float(boot4['sensitivity_ci_low']))} to {fmt_pct(float(boot4['sensitivity_ci_high']))}, "
            f"PPV {fmt_pct(float(boot4['ppv_ci_low']))} to {fmt_pct(float(boot4['ppv_ci_high']))}, and "
            f"false-alarm rate {fmt_pct(float(boot4['false_alarm_rate_ci_low']))} to {fmt_pct(float(boot4['false_alarm_rate_ci_high']))}. "
            "The 8-week block bootstrap intervals were "
            f"sensitivity {fmt_pct(float(boot8['sensitivity_ci_low']))} to {fmt_pct(float(boot8['sensitivity_ci_high']))}, "
            f"PPV {fmt_pct(float(boot8['ppv_ci_low']))} to {fmt_pct(float(boot8['ppv_ci_high']))}, and "
            f"false-alarm rate {fmt_pct(float(boot8['false_alarm_rate_ci_low']))} to {fmt_pct(float(boot8['false_alarm_rate_ci_high']))}."
        ),
        "",
        "## Nested Threshold Selection",
        "Baseline-only calibration:",
    ]
    for _, row in nested_base.iterrows():
        lines.append(
            f"- {row['objective']}: aggregate hold-out alerts {int(row['alerts'])}, TP {int(row['tp'])}, "
            f"FP {int(row['fp'])}, sensitivity {fmt_pct(float(row['sensitivity']))}, "
            f"PPV {fmt_pct(float(row['ppv']))}, false-alarm rate {fmt_pct(float(row['false_alarm_rate']))}."
        )
    lines.append("All-nonheld-week calibration stress test:")
    for _, row in nested_all.iterrows():
        lines.append(
            f"- {row['objective']}: aggregate hold-out alerts {int(row['alerts'])}, TP {int(row['tp'])}, "
            f"FP {int(row['fp'])}, sensitivity {fmt_pct(float(row['sensitivity']))}, "
            f"PPV {fmt_pct(float(row['ppv']))}, false-alarm rate {fmt_pct(float(row['false_alarm_rate']))}."
        )
    multipliers = (
        nested_fold.groupby(["calibration_mode", "objective"])["selected_sd_multiplier"]
        .apply(lambda s: ", ".join(f"{v:g}" for v in s.tolist()))
        .to_dict()
    )
    lines.extend(
        [
            "",
            "Selected SD multipliers by held-out season:",
            *[f"- {mode} / {objective}: {values}" for (mode, objective), values in multipliers.items()],
            "",
            "## Figure-Merge Decision",
            (
                "Do not merge as a new main figure panel. The baseline-only LOSO and nested checks "
                "support the existing S4 point estimate, while the all-week calibration stress test "
                "mainly shows that using epidemic weeks as a normal calibration distribution is too "
                "conservative. The block-bootstrap intervals are wide and are better reported as "
                "supplementary audit material or a short robustness sentence than as another figure."
            ),
        ]
    )
    (ANALYSIS / f"{OUT_PREFIX}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ANALYSIS.mkdir(exist_ok=True)
    df = load_validation()
    observed = original_rule_summary(df)
    loso_all_week_fold, loso_all_week_aggregate = leave_one_season_out_all_training_weeks(df)
    loso_baseline_fold, loso_baseline_aggregate = leave_one_season_out_baseline(df)
    bootstrap = pd.concat(
        [block_bootstrap_alert_ci(df, block_weeks=4), block_bootstrap_alert_ci(df, block_weeks=8)],
        ignore_index=True,
    )
    nested_all_week_fold, nested_all_week_candidates, nested_all_week_aggregate = nested_threshold_selection(
        df, "all_nonheld_weeks"
    )
    nested_baseline_fold, nested_baseline_candidates, nested_baseline_aggregate = nested_threshold_selection(
        df, "baseline_only"
    )
    nested_fold = pd.concat([nested_baseline_fold, nested_all_week_fold], ignore_index=True)
    nested_candidates = pd.concat([nested_baseline_candidates, nested_all_week_candidates], ignore_index=True)
    nested_aggregate = pd.concat([nested_baseline_aggregate, nested_all_week_aggregate], ignore_index=True)

    observed.to_csv(ANALYSIS / f"{OUT_PREFIX}_observed_rules.csv", index=False)
    loso_baseline_fold.to_csv(ANALYSIS / f"{OUT_PREFIX}_leave_one_season_out_baseline_folds.csv", index=False)
    loso_baseline_aggregate.to_csv(ANALYSIS / f"{OUT_PREFIX}_leave_one_season_out_baseline_aggregate.csv", index=False)
    loso_all_week_fold.to_csv(ANALYSIS / f"{OUT_PREFIX}_leave_one_season_out_all_week_folds.csv", index=False)
    loso_all_week_aggregate.to_csv(
        ANALYSIS / f"{OUT_PREFIX}_leave_one_season_out_all_week_aggregate.csv", index=False
    )
    pd.concat([loso_baseline_fold, loso_all_week_fold], ignore_index=True).to_csv(
        ANALYSIS / f"{OUT_PREFIX}_leave_one_season_out_folds.csv", index=False
    )
    pd.concat([loso_baseline_aggregate, loso_all_week_aggregate], ignore_index=True).to_csv(
        ANALYSIS / f"{OUT_PREFIX}_leave_one_season_out_aggregate.csv", index=False
    )
    bootstrap.to_csv(ANALYSIS / f"{OUT_PREFIX}_block_bootstrap_ci.csv", index=False)
    nested_fold.to_csv(ANALYSIS / f"{OUT_PREFIX}_nested_threshold_folds.csv", index=False)
    nested_candidates.to_csv(ANALYSIS / f"{OUT_PREFIX}_nested_threshold_candidates.csv", index=False)
    nested_aggregate.to_csv(ANALYSIS / f"{OUT_PREFIX}_nested_threshold_aggregate.csv", index=False)

    payload = {
        "seed": SEED,
        "n_boot": N_BOOT,
        "threshold_grid": THRESHOLD_GRID,
        "rule": {
            "score_columns": SCORE_COLS,
            "min_groups": 2,
            "season_months": [11, 12, 1, 2, 3],
            "persistence_weeks": 2,
            "threshold_source": (
                "baseline-only analyses use 2016-2018 calibration weeks after excluding the held-out "
                "respiratory season when applicable; all-nonheld-week stress tests use every valid "
                "non-held-out week as the calibration distribution"
            ),
        },
        "outputs": {
            "observed": f"{OUT_PREFIX}_observed_rules.csv",
            "leave_one_season_out_folds": f"{OUT_PREFIX}_leave_one_season_out_folds.csv",
            "leave_one_season_out_aggregate": f"{OUT_PREFIX}_leave_one_season_out_aggregate.csv",
            "leave_one_season_out_baseline_folds": f"{OUT_PREFIX}_leave_one_season_out_baseline_folds.csv",
            "leave_one_season_out_baseline_aggregate": f"{OUT_PREFIX}_leave_one_season_out_baseline_aggregate.csv",
            "leave_one_season_out_all_week_folds": f"{OUT_PREFIX}_leave_one_season_out_all_week_folds.csv",
            "leave_one_season_out_all_week_aggregate": f"{OUT_PREFIX}_leave_one_season_out_all_week_aggregate.csv",
            "block_bootstrap_ci": f"{OUT_PREFIX}_block_bootstrap_ci.csv",
            "nested_threshold_folds": f"{OUT_PREFIX}_nested_threshold_folds.csv",
            "nested_threshold_candidates": f"{OUT_PREFIX}_nested_threshold_candidates.csv",
            "nested_threshold_aggregate": f"{OUT_PREFIX}_nested_threshold_aggregate.csv",
            "summary": f"{OUT_PREFIX}_summary.md",
        },
    }
    (ANALYSIS / f"{OUT_PREFIX}_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_summary(
        observed,
        loso_baseline_fold,
        loso_baseline_aggregate,
        loso_all_week_fold,
        loso_all_week_aggregate,
        bootstrap,
        nested_fold,
        nested_aggregate,
    )

    print("Observed alert rules")
    print(observed[["strategy", "alerts", "tp", "fp", "sensitivity", "ppv", "false_alarm_rate"]].to_string(index=False))
    print("\nLOSO fixed S4 aggregate")
    print(
        pd.concat([loso_baseline_aggregate, loso_all_week_aggregate], ignore_index=True)[
            ["analysis", "alerts", "tp", "fp", "sensitivity", "ppv", "false_alarm_rate"]
        ].to_string(index=False)
    )
    print("\nNested threshold aggregate")
    print(
        nested_aggregate[
            ["calibration_mode", "objective", "alerts", "tp", "fp", "sensitivity", "ppv", "false_alarm_rate"]
        ].to_string(index=False)
    )
    print(f"\nWrote {ANALYSIS / (OUT_PREFIX + '_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
