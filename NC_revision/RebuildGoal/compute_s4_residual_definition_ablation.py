from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis_outputs"

VALIDATION_CSV = ANALYSIS / "lgdi_whu_influenza_validation.csv"
WEEKLY_CSV = ANALYSIS / "lgdi_whu_rolling4_weekly.csv"
GROUP_WINDOW_CSV = ANALYSIS / "lgdi_whu_group_window_metrics.csv"
OUT_CSV = ANALYSIS / "s4_residual_definition_ablation.csv"
OUT_JSON = ANALYSIS / "s4_residual_definition_ablation.json"

GROUPS = [
    "Cardiovascular",
    "Hypertension",
    "Diabetes",
    "Cerebrovascular",
    "Renal",
    "Respiratory",
]


def pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{100 * value:.1f}%"


def baseline_z(frame: pd.DataFrame, baseline_mask: pd.Series) -> pd.DataFrame:
    z = pd.DataFrame(index=frame.index)
    for group in GROUPS:
        values = pd.to_numeric(frame[group], errors="coerce")
        mu = values[baseline_mask].mean()
        sd = values[baseline_mask].std(ddof=1)
        z[group] = (values - mu) / sd if np.isfinite(sd) and sd > 0 else np.nan
    return z


def s4_metrics(scores: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict[str, object], pd.Series]:
    baseline_mask = validation["window_anchor"].dt.year.between(2016, 2018)
    flags = []
    thresholds = {}
    for group in GROUPS:
        values = pd.to_numeric(scores[group], errors="coerce")
        mu = values[baseline_mask].mean()
        sd = values[baseline_mask].std(ddof=1)
        threshold = mu + 1.5 * sd
        thresholds[group] = {"mean": float(mu), "sd": float(sd), "threshold": float(threshold)}
        flags.append((values >= threshold).fillna(False).astype(int).to_numpy())

    consensus2 = np.vstack(flags).sum(axis=0) >= 2
    in_core_season = validation["week_start"].dt.month.isin([11, 12, 1, 2, 3]).to_numpy()
    season_consensus = consensus2 & in_core_season
    s4 = season_consensus & np.concatenate([[False], season_consensus[:-1]])

    event = validation["flu_event_window"].astype(bool).to_numpy()
    tp = int((s4 & event).sum())
    fp = int((s4 & ~event).sum())
    fn = int((~s4 & event).sum())
    tn = int((~s4 & ~event).sum())
    metrics = {
        "alerts": int(s4.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "PPV": tp / (tp + fp) if (tp + fp) else float("nan"),
        "FAR": fp / (fp + tn) if (fp + tn) else float("nan"),
        "thresholds": thresholds,
    }
    return metrics, pd.Series(s4.astype(int), index=validation.index)


def pivot_observed_mean(group_window: pd.DataFrame, outcome: str, validation: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        group_window[group_window["outcome"].eq(outcome)]
        .pivot(index="label", columns="group", values="observed_mean")
        .reset_index()
    )
    return validation[["label"]].merge(pivot, on="label", how="left")[GROUPS]


def main() -> int:
    validation = pd.read_csv(VALIDATION_CSV, parse_dates=["week_start", "window_anchor"])
    validation = validation.sort_values("week_start").reset_index(drop=True)
    weekly = pd.read_csv(WEEKLY_CSV)
    group_window = pd.read_csv(GROUP_WINDOW_CSV)

    if len(validation) != 213:
        raise ValueError(f"Expected 213 validation weeks, found {len(validation)}")
    event_weeks = int(validation["flu_event_window"].astype(bool).sum())
    if event_weeks != 51:
        raise ValueError(f"Expected 51 event weeks, found {event_weeks}")

    baseline_mask = validation["window_anchor"].dt.year.between(2016, 2018)

    xgb_scores = validation[[f"score_{group}" for group in GROUPS]].rename(
        columns={f"score_{group}": group for group in GROUPS}
    )

    raw_counts = validation[["label"]].merge(
        weekly[["label"] + [f"n_{group}" for group in GROUPS]].rename(
            columns={f"n_{group}": group for group in GROUPS}
        ),
        on="label",
        how="left",
    )[GROUPS]
    raw_admission_z = baseline_z(raw_counts, baseline_mask)

    rolling_los_mean = pivot_observed_mean(group_window, "next_los_days", validation)
    rolling_los_z = baseline_z(rolling_los_mean, baseline_mask)

    los_mean = pivot_observed_mean(group_window, "next_los_days", validation)
    gap_mean = pivot_observed_mean(group_window, "next_gap_days", validation)
    los_z = baseline_z(los_mean, baseline_mask)
    gap_z = baseline_z(gap_mean, baseline_mask)
    simple_group_week_z = pd.DataFrame(index=validation.index)
    for group in GROUPS:
        # Shorter-than-baseline return intervals are aligned as positive pressure.
        simple_group_week_z[group] = pd.concat([los_z[group], -gap_z[group]], axis=1).mean(axis=1)

    score_sets = {
        "XGBoost residual (ours)": {
            "scores": xgb_scores,
            "definition": (
                "Mean signed XGBoost next-LOS and next-gap residuals per chronic-disease "
                "group and rolling four-week window, scaled by baseline mean absolute residual."
            ),
        },
        "Raw weekly admission Z": {
            "scores": raw_admission_z,
            "definition": (
                "Baseline-standardized chronic-disease group admission counts in the same "
                "rolling four-week windows."
            ),
        },
        "Rolling-mean LOS deviat.": {
            "scores": rolling_los_z,
            "definition": (
                "Baseline-standardized observed mean next-LOS for each chronic-disease group "
                "and rolling four-week window, without individual prediction."
            ),
        },
        "Simple group-week Z": {
            "scores": simple_group_week_z,
            "definition": (
                "Mean of baseline-standardized observed next-LOS and sign-reversed observed "
                "next-gap group-week deviations, without individual prediction."
            ),
        },
    }

    table_rows = []
    detail = {
        "n_weeks": int(len(validation)),
        "event_weeks": event_weeks,
        "alert_rule": (
            "S4: at least two chronic-disease groups above each group's 2016-2018 "
            "baseline mean plus 1.5 SD, restricted to November-March and requiring "
            "activation in the preceding week."
        ),
        "methods": {},
    }
    for method, payload in score_sets.items():
        metrics, alerts = s4_metrics(payload["scores"], validation)
        table_rows.append(
            {
                "Method": method,
                "alerts": metrics["alerts"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "Sensitivity": pct(float(metrics["sensitivity"])),
                "PPV": pct(float(metrics["PPV"])),
                "FAR": pct(float(metrics["FAR"])),
            }
        )
        detail["methods"][method] = {
            "definition": payload["definition"],
            "alerts": metrics["alerts"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TN": metrics["TN"],
            "sensitivity": metrics["sensitivity"],
            "PPV": metrics["PPV"],
            "FAR": metrics["FAR"],
            "alert_weeks": validation.loc[alerts.eq(1), "label"].tolist(),
            "thresholds": metrics["thresholds"],
        }

    pd.DataFrame(table_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(table_rows).to_string(index=False))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
