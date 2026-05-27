from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LGDI_DIR = BASE_DIR / "lgdi_results"
OUT_DIR = BASE_DIR / "whu_32k_lgdi_weekly_ppv_results"
PACKAGE_ANALYSIS = BASE_DIR / "resubmission_package_20260512" / "analysis_outputs"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv")
    weekly["window_anchor_dt"] = pd.to_datetime(weekly["window_anchor"])
    validation = pd.read_csv(LGDI_DIR / "lgdi_whu_influenza_validation.csv")
    validation["week_start_dt"] = pd.to_datetime(validation["week_start"])
    metrics = pd.read_csv(LGDI_DIR / "lgdi_whu_influenza_metrics.csv")

    baseline = weekly[weekly["valid"].eq(True) & weekly["window_anchor_dt"].between("2016-01-01", "2018-12-31")]
    resp_threshold = float(baseline["resp_score"].mean() + 1.5 * baseline["resp_score"].std())
    lgdi_threshold = float(baseline["lgdi"].mean() + 1.5 * baseline["lgdi"].std())

    line = weekly.merge(
        validation[["week_start_dt", "flu_event_window", "positivity"]],
        how="left",
        left_on="window_anchor_dt",
        right_on="week_start_dt",
    )
    line["resp_alert_mean_plus_1_5sd"] = line["resp_score"] >= resp_threshold
    line["lgdi_alert_mean_plus_1_5sd"] = line["lgdi"] >= lgdi_threshold
    line = line.drop(columns=["week_start_dt"])
    line.to_csv(OUT_DIR / "whu_32k_lgdi_weekly_line.csv", index=False, encoding="utf-8-sig")

    ppv = metrics[["strategy", "threshold", "alerts", "tp", "fp", "fn", "tn", "sensitivity", "ppv", "false_alarm_rate"]].copy()
    ppv.to_csv(OUT_DIR / "whu_32k_lgdi_ppv_table.csv", index=False, encoding="utf-8-sig")

    consensus = ppv[ppv["strategy"].eq("consensus_2groups_mean_plus_1_5sd")]
    def json_clean(value):
        if isinstance(value, dict):
            return {key: json_clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    primary = json_clean(consensus.iloc[0].to_dict()) if len(consensus) else {}
    summary = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cohort": "WHU primary original cohort; 32,056 linkable patient record numbers; 71,414 admissions",
        "algorithm": "LOS-Gap Deviation Index (LGDI) from XGBoost residual monitoring, not the older cosine/RDI proxy.",
        "baseline_window": {"start": "2016-01-01", "end": "2018-12-31"},
        "weekly_line": {
            "rows": int(len(line)),
            "valid_rows": int(line["valid"].sum()),
            "resp_score_threshold_mean_plus_1_5sd": resp_threshold,
            "lgdi_threshold_mean_plus_1_5sd": lgdi_threshold,
            "output": str(OUT_DIR / "whu_32k_lgdi_weekly_line.csv"),
        },
        "primary_ppv_operating_point": primary,
        "ppv_output": str(OUT_DIR / "whu_32k_lgdi_ppv_table.csv"),
        "source_files": [
            str(LGDI_DIR / "lgdi_whu_rolling4_weekly.csv"),
            str(LGDI_DIR / "lgdi_whu_influenza_validation.csv"),
            str(LGDI_DIR / "lgdi_whu_influenza_metrics.csv"),
        ],
    }
    summary = json_clean(summary)
    (OUT_DIR / "whu_32k_lgdi_weekly_ppv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    PACKAGE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    for name in ["whu_32k_lgdi_weekly_line.csv", "whu_32k_lgdi_ppv_table.csv", "whu_32k_lgdi_weekly_ppv_summary.json"]:
        (PACKAGE_ANALYSIS / name).write_bytes((OUT_DIR / name).read_bytes())

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
