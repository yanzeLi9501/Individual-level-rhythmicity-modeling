#!/usr/bin/env python3
"""
Generate optimized FigureS4_early_warning_extended using Pearson profile correlation strategy.

Panels:
  A - 2019 Monthly Behavioral Monitoring (per-group Pearson profile correlation vs COVID+ reference,
      P97.5 threshold from 2016-2018 baseline)
  B - Detection Performance vs. Threshold (Pearson_resp > baseline_mean + σ×SD)

Data sources:
  - all_admissions.csv  (behavioral profiles)
  - covid_test_results.csv  (COVID+ reference patients)

Saves to: NC_revision/submit/figures/FigureS4_early_warning_extended.{png,pdf,tif}

Run from: NC_revision/ directory
  python gen_figure_s4_optimized.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_CSV     = r'data\readmission_output\all_admissions.csv'
COVID_CSV    = r'data\readmission_output\covid_test_results.csv'
OUT_DIR      = os.path.join(BASE, 'submit', 'figures')
DEEP_OUT_DIR = os.path.join(BASE, 'DeepseekRevision', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEEP_OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
GROUP_COLORS = {
    'Respiratory':    '#E8636E',
    'Cardiovascular': '#4A90D9',
    'Hypertension':   '#F5A623',
    'Diabetes':       '#5BBD8C',
    'Cerebrovascular':'#9B7ED8',
    'Renal':          '#4DD0E1',
}
GROUP_LW = {
    'Respiratory':    2.5,
    'Cardiovascular': 1.2,
    'Hypertension':   1.2,
    'Diabetes':       1.2,
    'Cerebrovascular':1.2,
    'Renal':          1.2,
}
GROUPS     = ['Cardiovascular', 'Hypertension', 'Diabetes', 'Cerebrovascular', 'Renal', 'Respiratory']
COMORBIDITY_PATTERNS = {
    'Cardiovascular': r'冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入',
    'Hypertension':   r'高血压',
    'Diabetes':       r'糖尿病|血糖',
    'Cerebrovascular':r'脑梗|脑出血|脑血管|脑卒中|中风|腔隙性',
    'Renal':          r'肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏',
    'Respiratory':    r'肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭',
}
BEHAVIOR_COLS = ['实际住院天数', '检验项目数', '医嘱数量', '药品医嘱', '检查数量']
LAB_COLS      = ['lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']
ALL_METRICS   = BEHAVIOR_COLS + LAB_COLS
MONTH_ABBR    = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def panel_label(ax, label, x=-0.10, y=1.03):
    # NC submission spec: external panel labels at axes top-left (R5)
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left', clip_on=False)

def save_figure(fig, name):
    for out_dir in (OUT_DIR, DEEP_OUT_DIR):
        for ext in ('png', 'pdf', 'svg', 'tif'):
            path = os.path.join(out_dir, f'{name}.{ext}')
            kw = {}
            if ext in ('png', 'tif'):
                kw['dpi'] = 300
            if ext == 'tif':
                kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
            fig.savefig(path, bbox_inches='tight', facecolor='white', **kw)
            print(f'  Saved: {path}')

def profile_pearsonr(v1, v2):
    """Pearson r between two z-score profile vectors; returns 0.0 on degenerate input."""
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    mask = np.isfinite(v1) & np.isfinite(v2)
    if mask.sum() < 4 or np.std(v1[mask]) < 1e-10 or np.std(v2[mask]) < 1e-10:
        return 0.0
    r, _ = pearsonr(v1[mask], v2[mask])
    return float(r)

print("=" * 60)
print("FigureS4: 2019 Monthly Behavioral Monitoring (Pearson profile correlation)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load admission data
# ═══════════════════════════════════════════════════════════════
print("[1] Loading all_admissions.csv...")
td = pd.read_csv(DATA_CSV, encoding='utf-8-sig', low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year']  = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month

diag = td['EMR_初步诊断'].fillna('').astype(str)
for grp, pat in COMORBIDITY_PATTERNS.items():
    td[grp] = diag.str.contains(pat, na=False)

print(f"  Total admissions: {len(td):,}")
print(f"  2019 admissions: {td[td['year']==2019].shape[0]:,}")

# ═══════════════════════════════════════════════════════════════
# 2. Build z-score baseline (2016-2018)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Computing 2016-2018 baseline statistics...")
base = td[td['year'].between(2016, 2018)]
ref_stats = {}
for col in ALL_METRICS:
    vals = pd.to_numeric(base[col], errors='coerce').dropna()
    ref_stats[col] = {'mean': float(vals.mean()), 'std': max(float(vals.std()), 1e-6)}

def group_zscore_profile(sub_df):
    """13-dim z-score profile vector for a group subset."""
    vec = []
    for col in ALL_METRICS:
        vals = pd.to_numeric(sub_df[col], errors='coerce').dropna()
        z = float((vals.mean() - ref_stats[col]['mean']) / ref_stats[col]['std']) if len(vals) > 0 else np.nan
        vec.append(z)
    return np.array(vec)

# ═══════════════════════════════════════════════════════════════
# 3. Build COVID+ reference profile
# ═══════════════════════════════════════════════════════════════
print("\n[3] Building COVID+ reference profile...")
covid_tests = pd.read_csv(COVID_CSV, encoding='utf-8-sig', low_memory=False)
pos_ids = set(covid_tests[covid_tests['status'] == 'positive']['patient_id'].astype(str).unique())
td['pid_str'] = td['住院流水号'].astype(str)
covid_rows = td[td['pid_str'].isin(pos_ids)]
print(f"  COVID+ patients matched: {len(covid_rows['pid_str'].unique())} patients, {len(covid_rows)} admissions")

if len(covid_rows) >= 10:
    vec_covid_ref = group_zscore_profile(covid_rows)
    print(f"  Reference profile computed (n={len(covid_rows)})")
else:
    print("  WARNING: too few matched COVID+ records; using 2020 Q1 as proxy reference")
    proxy = td[(td['year'] == 2020) & (td['month'] <= 4)]
    vec_covid_ref = group_zscore_profile(proxy)

print(f"  COVID+ reference (non-NaN dims): {np.sum(np.isfinite(vec_covid_ref))}/13")

# ═══════════════════════════════════════════════════════════════
# 4. Compute monthly Pearson profile correlation (2016-2020)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Computing monthly Pearson profile correlations...")
monthly_pearson = {}   # {(year, month, group): pearson_r}

for year in range(2016, 2021):
    for month in range(1, 13):
        subset_yr_mo = td[(td['year'] == year) & (td['month'] == month)]
        if len(subset_yr_mo) < 10:
            continue
        for grp in GROUPS:
            sub = subset_yr_mo[subset_yr_mo[grp]]
            if len(sub) < 5:
                monthly_pearson[(year, month, grp)] = np.nan
                continue
            vec = group_zscore_profile(sub)
            r = profile_pearsonr(vec, vec_covid_ref)
            monthly_pearson[(year, month, grp)] = r

# ═══════════════════════════════════════════════════════════════
# 5. Baseline statistics for P97.5 threshold (2016-2018, Respiratory)
# ═══════════════════════════════════════════════════════════════
baseline_resp = [monthly_pearson[(y, m, 'Respiratory')]
                 for y in range(2016, 2019) for m in range(1, 13)
                 if (y, m, 'Respiratory') in monthly_pearson
                 and np.isfinite(monthly_pearson[(y, m, 'Respiratory')])]
resp_base_mean = float(np.mean(baseline_resp))
resp_base_std  = float(np.std(baseline_resp))
resp_p975      = float(np.percentile(baseline_resp, 97.5))
print(f"  Respiratory baseline (2016-2018): mean={resp_base_mean:.4f}, "
      f"std={resp_base_std:.4f}, P97.5={resp_p975:.4f}  (n={len(baseline_resp)} months)")

# 2019 monthly data per group
months = list(range(1, 13))
monthly_2019 = {}
for grp in GROUPS:
    monthly_2019[grp] = [monthly_pearson.get((2019, m, grp), np.nan) for m in months]
    above = sum(1 for v in monthly_2019[grp] if np.isfinite(v) and v > resp_p975)
    print(f"  {grp} 2019: {above}/12 months above P97.5")

# ═══════════════════════════════════════════════════════════════
# 6. Panel B: threshold sweep (monthly Pearson r, Respiratory, 2019-2020)
# ═══════════════════════════════════════════════════════════════
print("\n[5] Panel B: threshold sweep (Pearson r, 2019-2020)...")

monitor_rows = []
for year in [2019, 2020]:
    for month in range(1, 13):
        r = monthly_pearson.get((year, month, 'Respiratory'), np.nan)
        if np.isfinite(r):
            monitor_rows.append({'year': year, 'month': month, 'pearson_r': r})

monitor_df = pd.DataFrame(monitor_rows)
monitor_df['ym'] = list(zip(monitor_df['year'], monitor_df['month']))
gt_months = {(2019, m) for m in range(9, 13)} | {(2020, m) for m in range(1, 6)}
monitor_df['truth'] = monitor_df['ym'].isin(gt_months).astype(int)
total_pos = int(monitor_df['truth'].sum())
total_neg = int((~monitor_df['truth'].astype(bool)).sum())
print(f"  Available months: {len(monitor_df)}, positive={total_pos}, negative={total_neg}")

thresholds_sigma = np.arange(0.25, 4.01, 0.25)
perf = []
for sigma in thresholds_sigma:
    thresh = resp_base_mean + sigma * resp_base_std
    alerts = (monitor_df['pearson_r'] > thresh).astype(int)
    truth  = monitor_df['truth']
    tp = int(((alerts == 1) & (truth == 1)).sum())
    fp = int(((alerts == 1) & (truth == 0)).sum())
    tn = int(((alerts == 0) & (truth == 0)).sum())
    fn = int(((alerts == 0) & (truth == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    perf.append({'sigma': sigma, 'thresh': thresh,
                 'sensitivity': sensitivity, 'specificity': specificity,
                 'ppv': ppv, 'f1': f1})
perf_df = pd.DataFrame(perf)
sig1 = perf_df[np.isclose(perf_df['sigma'], 1.0)].iloc[0]
print(f"  σ=1.0: sens={sig1['sensitivity']:.2f}, spec={sig1['specificity']:.2f}, "
      f"ppv={sig1['ppv']:.2f}, F1={sig1['f1']:.2f}")

# ═══════════════════════════════════════════════════════════════
# 7. Generate figure
# ═══════════════════════════════════════════════════════════════
print("\n[6] Generating Figure S4 (2-panel)...")

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.14, wspace=0.34)

# ─── Panel A: 2019 Monthly Behavioral Monitoring (Pearson) ─────
panel_label(ax_a, 'A')

for grp in GROUPS:
    vals = monthly_2019[grp]
    ax_a.plot(months, vals,
              color=GROUP_COLORS[grp],
              linewidth=GROUP_LW[grp],
              linestyle='-' if grp == 'Respiratory' else '--',
              label=grp, zorder=3 if grp == 'Respiratory' else 2,
              alpha=1.0 if grp == 'Respiratory' else 0.8)

# P97.5 threshold line
ax_a.axhline(resp_p975, color='#CC2529', linewidth=1.5, linestyle=':',
             zorder=4, label=f'P97.5 ({resp_p975:.3f})')

# Shade months where Respiratory exceeds threshold
resp_2019 = monthly_2019['Respiratory']
for m, v in zip(months, resp_2019):
    if np.isfinite(v) and v > resp_p975:
        ax_a.axvspan(m - 0.45, m + 0.45, alpha=0.10, color='#E8636E', zorder=1)

# Baseline ±1 SD band
ax_a.fill_between(months,
                  resp_base_mean - resp_base_std,
                  resp_base_mean + resp_base_std,
                  alpha=0.06, color='grey', zorder=0, label='_nolegend_')
ax_a.axhline(resp_base_mean, color='grey', linewidth=0.7,
             linestyle=':', alpha=0.6, label='_nolegend_')

ax_a.set_xticks(months)
ax_a.set_xticklabels(MONTH_ABBR, fontsize=8)
ax_a.set_xlabel('Month (2019)', fontsize=9)
ax_a.set_ylabel('Pearson Profile Correlation\n(vs COVID+ Reference)', fontsize=9)
ax_a.set_title('2019 Monthly Behavioral Monitoring\n(Pearson Optimized Strategy)', fontsize=9.5)
ax_a.grid(axis='y', color='#DDDDDD', linewidth=0.5, zorder=0)

n_exceed = sum(1 for v in resp_2019 if np.isfinite(v) and v > resp_p975)
ax_a.annotate(f'Respiratory exceeds\nP97.5 in {n_exceed}/12 months',
              xy=(0.97, 0.97), xycoords='axes fraction',
              fontsize=7.5, color='#CC2529', ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.85, edgecolor='#DDDDDD'))

legend_handles = (
    [Line2D([0], [0], color=GROUP_COLORS[g], lw=GROUP_LW[g],
             linestyle='-' if g == 'Respiratory' else '--', label=g)
     for g in GROUPS] +
    [Line2D([0], [0], color='#CC2529', lw=1.5, linestyle=':',
             label=f'P97.5 ({resp_p975:.3f})')]
)
ax_a.legend(handles=legend_handles, fontsize=7, loc='upper left',
            ncol=2, framealpha=0.9, labelspacing=0.3)
ax_a.set_xlim(0.5, 12.5)

# ─── Panel B: Detection Performance vs Threshold ────────────────
panel_label(ax_b, 'B')

ax_b.plot(perf_df['sigma'], perf_df['sensitivity'],
          color='#E8636E', linewidth=2, label='Sensitivity', marker='o', markersize=4)
ax_b.plot(perf_df['sigma'], perf_df['specificity'],
          color='#4A90D9', linewidth=2, label='Specificity', marker='s', markersize=4)
ax_b.plot(perf_df['sigma'], perf_df['ppv'],
          color='#5BBD8C', linewidth=2, label='PPV', marker='^', markersize=4)
ax_b.plot(perf_df['sigma'], perf_df['f1'],
          color='#F5A623', linewidth=1.5, linestyle='--',
          label='F1 Score', marker='D', markersize=3.5)

best_row = perf_df.loc[perf_df['f1'].idxmax()]
ax_b.axvline(best_row['sigma'], color='grey', linewidth=1.0,
             linestyle=':', alpha=0.7, zorder=0)
ax_b.scatter([best_row['sigma']], [best_row['f1']], color='#F5A623', s=60, zorder=5)
xtext = best_row['sigma'] + 0.3 if best_row['sigma'] < 3.0 else best_row['sigma'] - 1.5
ax_b.annotate(
    f"Optimal σ={best_row['sigma']:.2f}\n"
    f"F1={best_row['f1']:.2f}  "
    f"Sens={best_row['sensitivity']:.2f}\n"
    f"Spec={best_row['specificity']:.2f}",
    xy=(best_row['sigma'], best_row['f1']),
    xytext=(xtext, max(best_row['f1'] - 0.18, 0.05)),
    fontsize=7, color='#555555',
    arrowprops=dict(arrowstyle='->', color='grey', lw=0.8))

ax_b.set_xlabel('Detection Threshold σ (× Baseline SD)', fontsize=9)
ax_b.set_ylabel('Metric Value', fontsize=9)
ax_b.set_title('Detection Performance vs. Threshold\n(Pearson r > μ + σ×SD)', fontsize=9.5)
ax_b.set_xlim(0.25, 4.0)
ax_b.set_ylim(0, 1.05)
ax_b.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax_b.grid(axis='y', color='#DDDDDD', linewidth=0.5, zorder=0)
ax_b.text(0.03, 0.03,
          f'Ground truth: Sep 2019–May 2020\n({total_pos} positive, {total_neg} negative months)',
          transform=ax_b.transAxes, fontsize=7, color='#555555',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    alpha=0.85, edgecolor='#DDDDDD'))

save_figure(fig, 'FigureS4_early_warning_extended')
plt.close(fig)
print("\nDone!")
