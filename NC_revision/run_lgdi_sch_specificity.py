#!/usr/bin/env python3
"""SCH (Shandong Cancer Hospital) oncology-center specificity control.

For each post-COVID northern-hemisphere influenza season quarter (Q4 2021 -
Q1 2024) and each of six chronic-disease groups, compute the Pearson
correlation of that group's standardized healthcare-utilization profile
against the WHU pandemic-positive reference profile. Unlike the WHU
general hospital, the respiratory group should NOT consistently rank highest
at SCH during seasonal influenza peaks (organ-system specificity control).

The Mar-May 2022 Shanghai-region lockdown weeks are excluded as a documented
policy-shock confounder.

Input:  NC_revision/expanded_cardiac_wide_table.csv  (the SCH data)
Output: NC_revision/lgdi_results/sch_quarterly_profile_correlation.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
WIDE = BASE / "expanded_cardiac_wide_table.csv"
RESULTS = BASE / "lgdi_results"
OUT_CSV = RESULTS / "sch_quarterly_profile_correlation.csv"

# Disease-group keyword definitions
GROUPS_KW = {
    "Cardiovascular": ("心", "冠心病", "心肌", "心衰", "心力衰竭", "心律", "房颤",
                       "瓣膜", "动脉粥样"),
    "Hypertension":   ("高血压",),
    "Diabetes":       ("糖尿病",),
    "Cerebrovascular":("脑梗", "脑出血", "脑卒中", "卒中", "中风"),
    "Renal":          ("肾", "尿毒症", "透析"),
    "Respiratory":    ("肺", "呼吸", "支气管", "哮喘", "慢阻肺", "copd",
                       "肺炎", "流感", "上呼吸道"),
}

# Lab feature columns (shared utilization profile features)
LAB_COLS = ["白细胞", "超敏C反应蛋白", "血红蛋白", "白蛋白",
            "肌酐", "空腹血糖", "钾", "钠"]

# Lockdown exclusion window (Shanghai 2022)
LOCKDOWN_START = pd.Timestamp("2022-03-01")
LOCKDOWN_END   = pd.Timestamp("2022-05-31")


def assign_group(text: str) -> list[str]:
    s = (text or "").lower()
    found = []
    for g, kws in GROUPS_KW.items():
        for kw in kws:
            if kw.lower() in s:
                found.append(g); break
    return found


def standardize(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    mu = np.nanmean(a); sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(a)
    return (a - mu) / sd


def main() -> None:
    print(f"[SCH] Loading {WIDE} ...")
    df = pd.read_csv(WIDE, dtype={"病案号": str}, low_memory=False)
    print(f"  rows={len(df)}  patients={df['病案号'].nunique()}")
    df["入院时间"] = pd.to_datetime(df["入院时间"], errors="coerce")
    df = df.dropna(subset=["入院时间"]).copy()

    # Restrict to post-COVID NH flu seasons (Q4 2021 - Q1 2024)
    df = df[(df["入院时间"] >= "2021-10-01") & (df["入院时间"] < "2024-04-01")].copy()
    # Exclude lockdown
    df = df[~df["入院时间"].between(LOCKDOWN_START, LOCKDOWN_END)].copy()
    print(f"  after season+lockdown filter: rows={len(df)}")

    # Multi-label disease group assignment
    diag_text = df.get("主要诊断", pd.Series("", index=df.index)).fillna("").astype(str) \
              + " " + df.get("上次诊断", pd.Series("", index=df.index)).fillna("").astype(str)
    df["_groups"] = diag_text.map(assign_group)

    # Lab features (numeric)
    feats = []
    for c in LAB_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            feats.append(c)
    # Add utilization features: length of stay (days) and admission_year_week count proxy
    if "出院时间" in df.columns:
        df["出院时间"] = pd.to_datetime(df["出院时间"], errors="coerce")
        df["los_days"] = (df["出院时间"] - df["入院时间"]).dt.total_seconds() / 86400
        feats.append("los_days")
    print(f"  feature columns: {feats}")

    # Build SCH-wide reference profile = pandemic-positive proxy (respiratory-coded admissions)
    resp_mask = df["_groups"].apply(lambda lst: "Respiratory" in lst)
    ref_df = df[resp_mask]
    ref_profile = {}
    for f in feats:
        ref_profile[f] = float(np.nanmean(pd.to_numeric(ref_df[f], errors="coerce")))
    ref_vec = np.array([ref_profile[f] for f in feats], dtype=float)
    # Standardize globally for Pearson correlation comparability
    glob_mu = np.array([np.nanmean(pd.to_numeric(df[f], errors="coerce")) for f in feats])
    glob_sd = np.array([np.nanstd(pd.to_numeric(df[f], errors="coerce"))  for f in feats])
    glob_sd[glob_sd == 0] = 1.0
    ref_z = (ref_vec - glob_mu) / glob_sd

    # Quarter labels
    df["quarter"] = df["入院时间"].dt.to_period("Q").astype(str)

    rows = []
    for q, qdf in df.groupby("quarter"):
        for g in GROUPS_KW.keys():
            sub = qdf[qdf["_groups"].apply(lambda lst, gg=g: gg in lst)]
            if len(sub) < 5:
                rows.append({"quarter": q, "group": g, "pearson_r": float("nan"),
                             "n_admissions": int(len(sub))})
                continue
            grp_vec = np.array([np.nanmean(pd.to_numeric(sub[f], errors="coerce")) for f in feats])
            grp_z = (grp_vec - glob_mu) / glob_sd
            # Pearson r between the two z-vectors
            mask = np.isfinite(grp_z) & np.isfinite(ref_z)
            if mask.sum() < 3:
                r = float("nan")
            else:
                gz = grp_z[mask] - grp_z[mask].mean()
                rz = ref_z[mask] - ref_z[mask].mean()
                denom = (np.sqrt((gz**2).sum()) * np.sqrt((rz**2).sum()))
                r = float((gz * rz).sum() / denom) if denom > 0 else float("nan")
            rows.append({"quarter": q, "group": g, "pearson_r": r,
                         "n_admissions": int(len(sub))})

    out = pd.DataFrame(rows).sort_values(["quarter", "group"])
    out.to_csv(OUT_CSV, index=False)
    print(f"[SCH] wrote {OUT_CSV}")
    print(out.to_string(index=False))

    # Quick rank summary
    print("\n[SCH] Per-quarter group ranking by Pearson r (desc):")
    for q, qsub in out.groupby("quarter"):
        order = qsub.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=False)
        ranking = " > ".join(f"{g}({r:+.2f})" for g, r in zip(order["group"], order["pearson_r"]))
        resp_rank = (order["group"].tolist().index("Respiratory") + 1) if "Respiratory" in order["group"].tolist() else "NA"
        print(f"  {q}: respiratory rank={resp_rank}  | {ranking}")


if __name__ == "__main__":
    main()
