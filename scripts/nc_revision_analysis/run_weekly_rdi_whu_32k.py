"""Compute monthly + 4-week-rolling weekly RDI for the WHU primary cohort
(32,056 patient record numbers, 71,414 admissions in all_admissions.csv).

Outputs to NC_revision/weekly_rdi_whu_32k_results/:
  - weekly_rdi_whu_32k_monthly.csv
  - weekly_rdi_whu_32k_rolling4_weekly.csv
  - weekly_rdi_whu_32k_summary.json

Schema mirrors weekly_rdi_42k_*.csv so panels can plot both cohorts side by side.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

BASE_DIR = Path(__file__).resolve().parent
INPUT_TABLE = Path(r"data/readmission_output/all_admissions.csv")
OUT_DIR = BASE_DIR / "weekly_rdi_whu_32k_results"
OUT_DIR.mkdir(exist_ok=True)

COMORBIDITIES = ["Cardiovascular", "Hypertension", "Diabetes", "Cerebrovascular", "Renal", "Respiratory"]
COMORBIDITY_PATTERNS = {
    "Cardiovascular": r"冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病",
    "Hypertension": r"高血压",
    "Diabetes": r"糖尿病|血糖",
    "Cerebrovascular": r"脑梗|脑出血|脑血管|脑卒中|中风|腔隙性",
    "Renal": r"肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏",
    "Respiratory": r"肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染",
}

# Surrogate "lab" features built from administrative timing fields only, because the
# WHU admissions table does not contain per-encounter lab panels. The behavioral
# profile is therefore a function of LOS + admission spacing (consistent with the
# behavioral-only LGDI WHU pipeline already used in lgdi_results/).
METRICS = ["los_days", "pid_visit_idx"]

EVENT_START = pd.Timestamp("2019-12-30")  # influenza event reference for WHU
EVENT_END = pd.Timestamp("2020-02-09")
LEAD_START = EVENT_START - pd.Timedelta(days=28)
MONITOR_START = pd.Timestamp("2019-01-01")
MONITOR_END = pd.Timestamp("2024-12-31")


@dataclass(frozen=True)
class WindowSpec:
    name: str
    span_days: int
    unit: str
    min_total_n: int
    min_group_n: int


WINDOW_SPECS = [
    WindowSpec("rolling4_weekly", 28, "week", 50, 10),
    WindowSpec("monthly", 31, "month", 50, 10),
]


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(INPUT_TABLE, low_memory=False)
    df["admit_dt"] = pd.to_datetime(df["入院日期"], errors="coerce")
    df["discharge_dt"] = pd.to_datetime(df["出院日期"], errors="coerce")
    df = df.dropna(subset=["admit_dt"]).copy()
    df["los_days"] = (df["discharge_dt"] - df["admit_dt"]).dt.days
    df.loc[(df["los_days"] < 0) | (df["los_days"] > 180), "los_days"] = np.nan
    df["year"] = df["admit_dt"].dt.year
    df["week_start"] = (df["admit_dt"] - pd.to_timedelta(df["admit_dt"].dt.weekday, unit="D")).dt.normalize()
    df["pid_visit_idx"] = df.groupby("住院流水号")["admit_dt"].rank(method="first").astype(float)

    diag = df["EMR_初步诊断"].fillna("").astype(str)
    for group, pattern in COMORBIDITY_PATTERNS.items():
        df[group] = diag.str.contains(pattern, case=False, regex=True, na=False).astype(int)
    df["is_covid_positive"] = diag.str.contains(r"新型冠状病毒|冠状病毒感染|冠状病毒肺炎", regex=True, na=False).astype(int)
    return df


def reference_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline = df[(df["year"] >= 2016) & (df["year"] <= 2018)]
    out: dict[str, dict[str, float]] = {}
    for col in METRICS:
        v = pd.to_numeric(baseline[col], errors="coerce").dropna()
        m = float(v.mean()) if len(v) else 0.0
        s = float(v.std()) if len(v) else 1.0
        if not np.isfinite(s) or s <= 1e-9:
            s = 1.0
        out[col] = {"mean": m, "std": s, "n": int(len(v))}
    return out


def profile_vector(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> np.ndarray:
    vals: list[float] = []
    for col in METRICS:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s):
            vals.append((float(s.mean()) - stats[col]["mean"]) / stats[col]["std"])
        else:
            vals.append(0.0)
    return np.nan_to_num(np.array(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def pearson_corr(ref: np.ndarray, vec: np.ndarray) -> float:
    if np.linalg.norm(ref) <= 0 or np.linalg.norm(vec) <= 0:
        return math.nan
    return float(1 - cosine(ref, vec))


def build_reference_profile(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> np.ndarray:
    baseline = df[(df["year"] >= 2016) & (df["year"] <= 2018) & df["Respiratory"].eq(1)]
    return profile_vector(baseline, stats)


def row_for(label: str, start: pd.Timestamp, end: pd.Timestamp, frame: pd.DataFrame,
            spec: WindowSpec, ref: np.ndarray, stats: dict[str, dict[str, float]]) -> dict[str, object]:
    n = int(len(frame))
    base = {
        "window_type": spec.name, "label": label,
        "window_start": start.date().isoformat(), "window_end": end.date().isoformat(),
        "n_admissions": n, "n_admissions_excluding_covid_reference": int((frame["is_covid_positive"] == 0).sum()),
        "event_window": bool(start <= EVENT_END and end >= EVENT_START),
        "lead_or_event_window": bool(start <= EVENT_END and end >= LEAD_START),
    }
    if n < spec.min_total_n:
        return {**base, "valid": False, "resp_sim": math.nan, "mean_other": math.nan, "rdi": math.nan}
    monitor = frame[frame["is_covid_positive"] == 0]
    sims: dict[str, float] = {}
    counts: dict[str, int] = {}
    for group in COMORBIDITIES:
        sub = monitor[monitor[group] == 1]
        counts[group] = int(len(sub))
        sims[group] = pearson_corr(ref, profile_vector(sub, stats)) if len(sub) >= spec.min_group_n else math.nan
    resp = sims.get("Respiratory", math.nan)
    others = [v for g, v in sims.items() if g != "Respiratory" and np.isfinite(v)]
    mo = float(np.mean(others)) if others else math.nan
    rdi = resp - mo if np.isfinite(resp) and np.isfinite(mo) else math.nan
    row = {**base, "valid": bool(np.isfinite(rdi)), "resp_sim": resp, "mean_other": mo, "rdi": rdi}
    for g in COMORBIDITIES:
        row[f"sim_{g}"] = sims.get(g, math.nan)
        row[f"n_{g}"] = counts.get(g, 0)
    return row


def make_windows(df: pd.DataFrame, spec: WindowSpec, ref: np.ndarray, stats: dict) -> pd.DataFrame:
    rows = []
    if spec.unit == "week":
        first = pd.Timestamp("2016-01-04")
        last = df["week_start"].max()
        for wk in pd.date_range(first, last, freq="W-MON"):
            start = wk - pd.Timedelta(days=spec.span_days - 7)
            end = wk + pd.Timedelta(days=6)
            frame = df[(df["admit_dt"] >= start) & (df["admit_dt"] <= end)]
            rows.append(row_for(f"{wk.isocalendar().year}W{wk.isocalendar().week:02d}", start, end, frame, spec, ref, stats))
    else:
        for ms in pd.date_range("2016-01-01", df["admit_dt"].max().replace(day=1), freq="MS"):
            end = ms + pd.offsets.MonthEnd(1)
            frame = df[(df["admit_dt"] >= ms) & (df["admit_dt"] <= end)]
            rows.append(row_for(ms.strftime("%Y-%m"), ms, pd.Timestamp(end), frame, spec, ref, stats))
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading WHU admissions...")
    df = load_and_prepare()
    n_pat = df["住院流水号"].nunique()
    print(f"  rows={len(df):,}  unique patient record numbers={n_pat:,}")
    stats = reference_stats(df)
    ref = build_reference_profile(df, stats)
    print(f"  reference vector (2016-2018, respiratory): {ref.round(3).tolist()}")

    outputs: dict[str, pd.DataFrame] = {}
    for spec in WINDOW_SPECS:
        print(f"Building {spec.name}...")
        out = make_windows(df, spec, ref, stats)
        outputs[spec.name] = out
        out.to_csv(OUT_DIR / f"weekly_rdi_whu_32k_{spec.name}.csv", index=False)

    # threshold + simple sens/PPV summary on monitor period
    summary: dict[str, object] = {
        "n_admissions": int(len(df)), "n_patient_record_numbers": int(n_pat),
        "event_window": {"start": EVENT_START.date().isoformat(), "end": EVENT_END.date().isoformat()},
        "lead_start": LEAD_START.date().isoformat(),
        "windows": {},
    }
    for name, frame in outputs.items():
        valid = frame[frame["valid"] == True].copy()
        valid["wsd"] = pd.to_datetime(valid["window_start"])
        base = valid[(valid["wsd"] >= "2016-01-01") & (valid["wsd"] <= "2018-12-31")]["rdi"]
        thr15 = float(base.mean() + 1.5 * base.std()) if len(base) else math.nan
        monitor = valid[(valid["wsd"] >= MONITOR_START) & (valid["wsd"] <= MONITOR_END)].copy()
        monitor["alert"] = monitor["rdi"] >= thr15
        ev = monitor["event_window"].astype(bool)
        al = monitor["alert"].astype(bool)
        tp = int((al & ev).sum()); fp = int((al & ~ev).sum())
        fn = int((~al & ev).sum()); tn = int((~al & ~ev).sum())
        summary["windows"][name] = {
            "valid_count": int(len(valid)),
            "baseline_n": int(len(base)),
            "threshold_mean_plus_1_5sd": thr15,
            "monitor_count": int(len(monitor)),
            "alerts": int(al.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": (tp / (tp + fn)) if (tp + fn) else math.nan,
            "ppv": (tp / (tp + fp)) if (tp + fp) else math.nan,
            "false_alarm_rate": (fp / (fp + tn)) if (fp + tn) else math.nan,
        }
    (OUT_DIR / "weekly_rdi_whu_32k_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["windows"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
