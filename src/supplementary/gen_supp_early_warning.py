"""
Supplementary Figure S9: Early Warning Simulation Extended (2×1)
  (A) 2019 monthly behavioral monitoring across six comorbidity subgroups,
      demonstrating persistent respiratory elevation above the P97.5 threshold.
  (B) Detection performance curves (sensitivity, specificity, PPV) at varying
      RDI thresholds, benchmarked against traditional surveillance reference.

Panel B reuses the threshold sweep logic from gen_supp_leadtime.py.

Generates:
  - FigureS9_early_warning_extended (PNG/PDF/TIF)

Run: python gen_supp_early_warning.py
"""
import sys, io, warnings, os, json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure S5: Early Warning Simulation Extended")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data & build profiles
# ═══════════════════════════════════════════════════════════════
print("[1] Loading data...")
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month

COMORBIDITY_PATTERNS = {
    'Cardiovascular': r'冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入',
    'Hypertension': r'高血压',
    'Diabetes': r'糖尿病|血糖',
    'Cerebrovascular': r'脑梗|脑出血|脑血管|脑卒中|中风|腔隙性',
    'Renal': r'肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏',
    'Respiratory': r'肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭',
}
diag_text = td['EMR_初步诊断'].fillna('').astype(str)
for name, pattern in COMORBIDITY_PATTERNS.items():
    td[name] = diag_text.str.contains(pattern, na=False).astype(int)

behavior_cols = ['实际住院天数', '检验项目数', '医嘱数量', '药品医嘱', '检查数量']
lab_cols = ['lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']
all_metrics = behavior_cols + lab_cols

baseline = td[(td['year'] >= 2016) & (td['year'] <= 2018)]
ref_stats = {}
for col in all_metrics:
    vals = pd.to_numeric(baseline[col], errors='coerce').dropna()
    ref_stats[col] = {'mean': vals.mean(), 'std': max(vals.std(), 1e-6)}

def zscore_profile(sub_df):
    p = {}
    for col in all_metrics:
        vals = pd.to_numeric(sub_df[col], errors='coerce').dropna()
        p[col] = (vals.mean() - ref_stats[col]['mean']) / ref_stats[col]['std'] if len(vals) > 0 else 0.0
    return p

def profile_to_vec(prof):
    return np.nan_to_num(np.array([prof.get(k, 0) for k in all_metrics], dtype=float), 0)

def cos_sim(v1, v2):
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        return 1 - cosine(v1, v2)
    return 0.0

# COVID+ reference
covid_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'covid_test_results.csv'), encoding='utf-8-sig')
positive_pids = set(covid_test[covid_test['status'] == 'positive']['patient_id'].astype(str).unique())
td['pid_str'] = td['住院流水号'].astype(str)
covid_pos = td[(td['pid_str'].isin(positive_pids)) & (td['year'] == 2020)]
vec_pos = profile_to_vec(zscore_profile(covid_pos))
print(f"  COVID+ reference: n={len(covid_pos)}")

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ═══════════════════════════════════════════════════════════════
# 2. Panel A: 2019 monthly monitoring across comorbidities
# ═══════════════════════════════════════════════════════════════
print("\n[2] Computing 2019 monthly similarities...")
td_2019 = td[td['year'] == 2019]

monthly_sim = {c: [] for c in comorbidities}
monthly_n = {c: [] for c in comorbidities}

for comor in comorbidities:
    for m in range(1, 13):
        sub = td_2019[(td_2019[comor] == 1) & (td_2019['month'] == m)]
        monthly_n[comor].append(len(sub))
        if len(sub) >= 5:
            v = profile_to_vec(zscore_profile(sub))
            monthly_sim[comor].append(cos_sim(vec_pos, v))
        else:
            monthly_sim[comor].append(np.nan)

# Compute P97.5 threshold from baseline (2016-2018) monthly similarities
print("  Computing P97.5 threshold from baseline...")
baseline_sims = []
for year in [2016, 2017, 2018]:
    for m in range(1, 13):
        for comor in comorbidities:
            sub = td[(td[comor] == 1) & (td['year'] == year) & (td['month'] == m)]
            if len(sub) >= 5:
                v = profile_to_vec(zscore_profile(sub))
                baseline_sims.append(cos_sim(vec_pos, v))

p97_5 = np.percentile(baseline_sims, 97.5) if baseline_sims else 0.5
print(f"  P97.5 threshold: {p97_5:.3f} (from {len(baseline_sims)} baseline observations)")

# ═══════════════════════════════════════════════════════════════
# 3. Panel B: Threshold sweep (sensitivity/specificity/PPV)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Computing threshold performance curves...")

# Monthly RDI computation (2016-2020)
monthly_records = []
for year in range(2016, 2021):
    for month in range(1, 13):
        if year == 2020 and month > 6:
            break
        sub_month = td[(td['year'] == year) & (td['month'] == month)]
        if len(sub_month) < 10:
            continue
        sims = {}
        for comor in comorbidities:
            csub = sub_month[sub_month[comor] == 1]
            if len(csub) >= 3:
                v = profile_to_vec(zscore_profile(csub))
                sims[comor] = cos_sim(vec_pos, v)
            else:
                sims[comor] = np.nan
        resp_sim = sims.get('Respiratory', np.nan)
        other_sims = [s for c, s in sims.items() if c != 'Respiratory' and not np.isnan(s)]
        mean_other = np.mean(other_sims) if other_sims else np.nan
        rdi = resp_sim - mean_other if not np.isnan(resp_sim) and not np.isnan(mean_other) else np.nan
        monthly_records.append({
            'year': year, 'month': month,
            'resp_sim': resp_sim, 'rdi': rdi,
        })

df_monthly = pd.DataFrame(monthly_records)

# Baseline stats
baseline_months = df_monthly[(df_monthly['year'] >= 2017) & (df_monthly['year'] <= 2018)]
rdi_baseline_mean = baseline_months['rdi'].mean()
rdi_baseline_std = baseline_months['rdi'].std()
resp_sim_baseline = baseline_months['resp_sim'].dropna()
pct_975 = np.percentile(resp_sim_baseline, 97.5)

# Ground truth: Liberal (Sep 2019+)
GROUND_TRUTH = {(2019, m) for m in range(9, 13)} | {(2020, m) for m in range(1, 7)}
monitoring = df_monthly[(df_monthly['year'] >= 2019) & (df_monthly['year'] * 100 + df_monthly['month'] <= 202006)].copy()

# Traditional surveillance reference
resp_rate_baseline = td[(td['year'] >= 2017) & (td['year'] <= 2018)].groupby('month')['Respiratory'].mean()
trad_first_alert = None
for _, row in monitoring.iterrows():
    m = int(row['month'])
    yr = int(row['year'])
    sub = td[(td['year'] == yr) & (td['month'] == m)]
    rate = sub['Respiratory'].mean() if len(sub) > 0 else 0
    thr = resp_rate_baseline.get(m, 0) + 2 * resp_rate_baseline.get(m, 0) * 0.3  # approx 2σ
    if rate > thr and trad_first_alert is None:
        trad_first_alert = (yr, m)

# Threshold sweep
thresholds_sd = np.arange(0.5, 3.1, 0.1)
sweep_results = {'sensitivity': [], 'specificity': [], 'ppv': []}

for thr_sd in thresholds_sd:
    rdi_thr = rdi_baseline_mean + thr_sd * rdi_baseline_std

    tp = fp = tn = fn = 0
    for _, row in monitoring.iterrows():
        is_pos = (row['year'], row['month']) in GROUND_TRUTH
        alert = ((not np.isnan(row['rdi'])) and row['rdi'] > rdi_thr) or \
                ((not np.isnan(row['resp_sim'])) and row['resp_sim'] > pct_975)
        if alert and is_pos:
            tp += 1
        elif alert and not is_pos:
            fp += 1
        elif not alert and is_pos:
            fn += 1
        else:
            tn += 1

    sweep_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    sweep_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    sweep_results['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)

# ═══════════════════════════════════════════════════════════════
# 4. Generate Figure S9 (1×2)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Generating FigureS5_early_warning_extended...")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, wspace=0.30,
                       left=0.07, right=0.97, top=0.88, bottom=0.15)

comor_colors = {
    'Respiratory': COLORS['secondary'],
    'Cardiovascular': COLORS['primary'],
    'Hypertension': COLORS['accent1'],
    'Diabetes': COLORS['accent2'],
    'Cerebrovascular': COLORS['accent3'],
    'Renal': COLORS['accent5'],
}

# ─── Panel A: 2019 monthly monitoring ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

x_months = np.arange(12)
for comor in comorbidities:
    vals = monthly_sim[comor]
    valid_x = [x for x, v in zip(x_months, vals) if not np.isnan(v)]
    valid_y = [v for v in vals if not np.isnan(v)]
    lw = 2.5 if comor == 'Respiratory' else 1.0
    alpha = 1.0 if comor == 'Respiratory' else 0.5
    ms = 6 if comor == 'Respiratory' else 3
    zorder = 10 if comor == 'Respiratory' else 5
    ax_a.plot(valid_x, valid_y, '-o', color=comor_colors[comor],
              linewidth=lw, markersize=ms, alpha=alpha, label=comor, zorder=zorder)

# P97.5 threshold line
ax_a.axhline(p97_5, color='red', linewidth=1.5, linestyle='--', alpha=0.8,
             label=f'P97.5 threshold ({p97_5:.3f})')

# Shade months where Respiratory exceeds threshold
resp_vals = monthly_sim['Respiratory']
for m_idx in range(12):
    if not np.isnan(resp_vals[m_idx]) and resp_vals[m_idx] > p97_5:
        ax_a.axvspan(m_idx - 0.4, m_idx + 0.4, alpha=0.08, color=COLORS['secondary'])

ax_a.set_xticks(x_months)
ax_a.set_xticklabels(month_names, fontsize=8)
ax_a.set_ylabel('Cosine Similarity to COVID+ Profile', fontsize=9)
ax_a.set_title('2019 Monthly Behavioral Monitoring', fontsize=10)
ax_a.legend(fontsize=6, ncol=2, loc='lower left', framealpha=0.9)
ax_a.set_ylim(-0.4, 1.1)
ax_a.axhline(0, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)

# Count months above threshold
n_above = sum(1 for v in resp_vals if not np.isnan(v) and v > p97_5)
ax_a.text(0.98, 0.02, f'Respiratory above P97.5: {n_above}/12 months',
          transform=ax_a.transAxes, fontsize=7, ha='right', va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['grid'], alpha=0.9))

# ─── Panel B: Detection performance curves ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

ax_b.plot(thresholds_sd, sweep_results['sensitivity'], '-', color=COLORS['secondary'],
          linewidth=2, label='Sensitivity', marker='o', markersize=2)
ax_b.plot(thresholds_sd, sweep_results['specificity'], '-', color=COLORS['primary'],
          linewidth=2, label='Specificity', marker='s', markersize=2)
ax_b.plot(thresholds_sd, sweep_results['ppv'], '-', color=COLORS['accent1'],
          linewidth=2, label='PPV', marker='^', markersize=2)

# Default threshold marker
ax_b.axvline(1.5, color='grey', linestyle=':', alpha=0.5, label='Default (1.5σ)')

# Traditional surveillance reference lines (if available)
# Draw a reference zone for typical surveillance performance
ax_b.axhspan(0.4, 0.7, alpha=0.05, color='grey')
ax_b.text(2.8, 0.55, 'Traditional\nsurveillance\nrange', fontsize=6,
          ha='right', va='center', color='grey', fontstyle='italic')

ax_b.set_xlabel('RDI Threshold (σ above baseline)', fontsize=9)
ax_b.set_ylabel('Rate', fontsize=9)
ax_b.set_title('Detection Performance vs. RDI Threshold', fontsize=10)
ax_b.legend(fontsize=7, loc='center right', framealpha=0.9)
ax_b.set_ylim(-0.05, 1.1)

save_figure(fig, 'FigureS5_early_warning_extended')
print("  FigureS5_early_warning_extended saved.")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
