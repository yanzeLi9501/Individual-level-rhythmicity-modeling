#!/usr/bin/env python3
"""Run LGDI XGBoost surveillance on the WHU primary cohort (Task 3).

Adapts the WHU primary `all_admissions.csv` to the schema expected by
`run_lgdi_surveillance.run_analysis` and runs the same pipeline (XGBoost LOS
and inter-admission gap regressors trained on 2016-2018, weekly LGDI windows).
The objective is the *2019 first-prediction* sanity check: train on
pre-pandemic baseline only, then read the 2019 weekly respiratory `group_score`
and LGDI series. Output prefix: `lgdi_whu`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from run_lgdi_surveillance import run_analysis, BASE_DIR  # type: ignore  # noqa: E402

WHU_RAW = Path(r"data\readmission_output\all_admissions.csv")
ADAPTED = BASE / "_tmp_whu_primary_for_lgdi.csv"


def adapt() -> None:
    print(f"Reading WHU primary: {WHU_RAW}")
    raw = pd.read_csv(WHU_RAW, dtype={"病案号": str, "身份证号": str}, low_memory=False)
    print(f"Raw rows: {len(raw)}, cols: {len(raw.columns)}")
    keep_text_cols = [c for c in ["EMR_初步诊断", "EMR_既往史", "EMR_出院记录", "EMR_主诉"] if c in raw.columns]
    diag_components = []
    if "诊断文本" in raw.columns:
        diag_components.append(raw["诊断文本"].fillna("").astype(str))
    for c in keep_text_cols:
        diag_components.append(raw[c].fillna("").astype(str))
    if diag_components:
        diagnosis = diag_components[0]
        for extra in diag_components[1:]:
            diagnosis = diagnosis + " " + extra
    else:
        diagnosis = pd.Series("", index=raw.index)

    out = pd.DataFrame({
        "病案号": raw["病案号"].astype(str) if "病案号" in raw.columns else "",
        "入院时间": pd.to_datetime(raw.get("入院日期"), errors="coerce"),
        "出院时间": pd.to_datetime(raw.get("出院日期"), errors="coerce"),
        "主要诊断": diagnosis,
        "入院科室": raw.get("入院科别", ""),
        "出院科室": raw.get("出院科别", ""),
        "上次诊断": "",
    })
    for src, alias in [("lab_WBC", "白细胞"), ("lab_CRP", "超敏C反应蛋白"), ("lab_HGB", "血红蛋白"),
                        ("lab_ALB", "白蛋白"), ("lab_CREA", "肌酐"), ("lab_GLU", "空腹血糖"),
                        ("lab_K", "钾"), ("lab_Na", "钠")]:
        if src in raw.columns:
            out[alias] = pd.to_numeric(raw[src], errors="coerce")
    out["病案号"] = out["病案号"].fillna("").astype(str).str.strip()
    out = out[(out["病案号"] != "") & (out["病案号"].str.lower() != "nan")].copy()
    out = out.dropna(subset=["入院时间"]).copy()
    print(f"Adapted rows after dropping missing MRN/admit: {len(out)} | unique patients: {out['病案号'].nunique()}")
    out.to_csv(ADAPTED, index=False, encoding="utf-8-sig")
    print(f"Wrote adapter file: {ADAPTED}")


def main() -> int:
    adapt()
    print("\nLaunching LGDI pipeline on WHU primary (this trains XGBoost, takes several minutes)...\n")
    summary = run_analysis(ADAPTED, prefix="lgdi_whu", include_external=False, cohort_label="WHU primary cohort")
    print("\n=== WHU LGDI summary ===")
    print(json.dumps(summary.get("cohort", {}), ensure_ascii=False, indent=2))

    # 2019 first-prediction sanity check
    timeline_csv = BASE / "lgdi_results" / "lgdi_whu_rolling4_weekly.csv"
    if timeline_csv.exists():
        timeline = pd.read_csv(timeline_csv)
        timeline["window_anchor_dt"] = pd.to_datetime(timeline["window_anchor"])
        valid = timeline[timeline["valid"].astype(bool)].copy()
        baseline = valid[valid["window_anchor_dt"].between("2016-01-01", "2018-12-31")]
        focus_2019 = valid[valid["window_anchor_dt"].between("2019-01-01", "2020-03-31")].copy()
        focus_2019.to_csv(BASE / "lgdi_results" / "lgdi_whu_2019_focus.csv", index=False, encoding="utf-8-sig")
        peak = focus_2019.sort_values("resp_score", ascending=False).head(10)
        report = {
            "scope": "WHU primary cohort, baseline 2016-2018; first-prediction window 2019-01 to 2020-03",
            "baseline_n_weeks": int(len(baseline)),
            "baseline_resp_score_mean": float(baseline["resp_score"].mean()) if len(baseline) else None,
            "baseline_resp_score_std": float(baseline["resp_score"].std()) if len(baseline) else None,
            "baseline_lgdi_mean": float(baseline["lgdi"].mean()) if len(baseline) else None,
            "baseline_lgdi_std": float(baseline["lgdi"].std()) if len(baseline) else None,
            "monitor_n_weeks_2019_2020Q1": int(len(focus_2019)),
            "monitor_resp_score_mean": float(focus_2019["resp_score"].mean()) if len(focus_2019) else None,
            "monitor_resp_score_max": float(focus_2019["resp_score"].max()) if len(focus_2019) else None,
            "monitor_lgdi_max": float(focus_2019["lgdi"].max()) if len(focus_2019) else None,
            "top10_resp_score_weeks": peak[["window_anchor", "resp_score", "mean_other_score", "lgdi", "n_admissions"]]
                .to_dict(orient="records"),
        }
        out_json = BASE / "lgdi_results" / "lgdi_whu_2019_focus.json"
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: None if pd.isna(o) else o), encoding="utf-8")
        print(f"\nWrote first-prediction focus: {out_json}")
        for k, v in report.items():
            if k != "top10_resp_score_weeks":
                print(f"  {k}: {v}")
        print("  top10 weeks:")
        for row in report["top10_resp_score_weeks"]:
            print(f"    {row['window_anchor']}  resp={row['resp_score']:.3f}  lgdi={row['lgdi']:.3f}  n={int(row['n_admissions'])}")

        pkg_dir = BASE / "resubmission_package_20260512" / "analysis_outputs"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        for src in [BASE / "lgdi_results" / "lgdi_whu_2019_focus.csv", out_json]:
            shutil.copy2(src, pkg_dir / src.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
