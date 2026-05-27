from __future__ import annotations

import pandas as pd

from rebuild_common import (
    GPU_XGB_DIR,
    LGDI_DIR,
    NC_DIR,
    TABLES_DIR,
    XGB_VISIT_DIR,
    ensure_dirs,
    num,
    pct,
    read_csv,
    read_json,
    short_label,
    write_table,
)


def table1() -> pd.DataFrame:
    whu_summary = read_json(LGDI_DIR / "lgdi_whu_summary.json")
    flu_summary = read_json(LGDI_DIR / "lgdi_whu_influenza_summary.json")
    external = read_csv(GPU_XGB_DIR / "native_xgboost_public_summary.csv")
    rows = [
        {
            "dataset": "WHU primary EHR",
            "role": "Discovery set and FluNet proof-of-concept",
            "setting": "Chinese general hospital",
            "records": whu_summary.get("cohort", {}).get("admissions_with_dates", "NA"),
            "patients_or_positive_labels": whu_summary.get("cohort", {}).get("unique_patient_record_numbers", "NA"),
            "endpoint_or_note": f"{flu_summary.get('n_weeks_total')} monitor weeks; {flu_summary.get('n_weeks_flu_event')} FluNet event weeks",
        },
        {
            "dataset": "Expanded cardiac WHU cohort",
            "role": "Same-system audit",
            "setting": "Chinese general hospital, overlapping source system",
            "records": "299728",
            "patients_or_positive_labels": "42795",
            "endpoint_or_note": "Not independent external validation",
        },
    ]
    for row in external.itertuples(index=False):
        rows.append(
            {
                "dataset": short_label(row.dataset),
                "role": "Cross-setting positive control",
                "setting": "Public ICU or hospital dataset; native features",
                "records": int(row.n_rows),
                "patients_or_positive_labels": int(row.positive_count),
                "endpoint_or_note": row.target,
            }
        )
    return pd.DataFrame(rows)


def table2() -> pd.DataFrame:
    metrics = read_csv(LGDI_DIR / "lgdi_whu_influenza_metrics.csv")
    wanted = [
        "REFERENCE_resp_mean_plus_1_5sd",
        "consensus_2groups_mean_plus_1_5sd",
        "consensus_2groups_season_oct_apr",
        "season_sustained_consensus2grp_nov_mar",
    ]
    table = metrics[metrics["strategy"].isin(wanted)].copy()
    table["strategy_label"] = table["strategy"].map(short_label)
    table = table[["strategy_label", "alerts", "tp", "fp", "fn", "tn", "sensitivity", "ppv", "false_alarm_rate"]]
    for col in ["sensitivity", "ppv", "false_alarm_rate"]:
        table[col] = table[col].map(pct)
    return table.rename(
        columns={
            "strategy_label": "strategy",
            "tp": "true_positive_weeks",
            "fp": "false_positive_weeks",
            "fn": "false_negative_weeks",
            "tn": "true_negative_weeks",
        }
    )


def table3() -> pd.DataFrame:
    whu = read_csv(XGB_VISIT_DIR / "visit_order_first_version_audit.csv")
    external = read_csv(GPU_XGB_DIR / "native_xgboost_public_summary.csv")
    rows = []
    for row in whu.itertuples(index=False):
        rows.append(
            {
                "dataset": "WHU individual XGBoost",
                "feature_strategy": f"WHU visit-history features ({int(row.n_features)} features)",
                "model_role": "Representative" if row.min_visit_order == 5 else "Selective frequent-visitor subset",
                "n": int(row.n),
                "primary_metric": "R2",
                "metric_value": num(row.r2_mean, 3),
                "secondary_metric": f"MAE {row.mae_mean:.2f}; RMSE {row.rmse_mean:.2f}",
                "device": "cuda in source audit",
                "interpretation": "Use broad patient split as main text" if row.min_visit_order == 5 else "Supplementary only",
            }
        )
    for row in external.itertuples(index=False):
        rows.append(
            {
                "dataset": short_label(row.dataset),
                "feature_strategy": f"Native public-dataset features ({int(row.n_features)} features)",
                "model_role": "Cross-setting positive control",
                "n": int(row.n_rows),
                "primary_metric": "ROC-AUC",
                "metric_value": num(row.roc_auc, 3),
                "secondary_metric": f"AP {row.average_precision:.3f}; PPV {row.ppv_precision:.3f}; sensitivity {row.sensitivity_recall:.3f}",
                "device": row.device_used,
                "interpretation": "Within-dataset fit only; not external validation",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    write_table(table1(), TABLES_DIR / "Table1_cohort_and_data_summary")
    write_table(table2(), TABLES_DIR / "Table2_fluNet_strategy_operating_characteristics")
    write_table(table3(), TABLES_DIR / "Table3_xgboost_performance_summary")


if __name__ == "__main__":
    main()