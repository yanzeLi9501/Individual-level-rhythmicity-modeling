"""
Figure 3: Early Warning — 1×2 merged figure (2 panels).
Panels: A) RDI timeline comparison (behavioral vs traditional surveillance)
        B) Monthly deviation heatmap
Run: python gen_fig_new3.py
"""
import sys, io, warnings, json, os
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
print("Figure 3: Early Warning (merged 2-panel)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data
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

print(f"  Total admissions: {len(td):,}")
print(f"  Year range: {td['year'].min()}-{td['year'].max()}")

# ═══════════════════════════════════════════════════════════════
# 2. Behavioral profile infrastructure
# ═══════════════════════════════════════════════════════════════
behavior_cols = ['实际住院天数', '检验项目数', '医嘱数量', '药品医嘱', '检查数量']
lab_cols = ['lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']
all_metrics = behavior_cols + lab_cols

# Baseline from 2017-2018
train_data = td[(td['year'] >= 2017) & (td['year'] <= 2018)]
ref_stats = {}
for col in all_metrics:
    vals = pd.to_numeric(train_data[col], errors='coerce').dropna()
    ref_stats[col] = {'mean': vals.mean(), 'std': max(vals.std(), 1e-6)}

def zscore_profile(sub_df, ref=ref_stats):
    p = {}
    for col in all_metrics:
        vals = pd.to_numeric(sub_df[col], errors='coerce').dropna()
        p[col] = (vals.mean() - ref[col]['mean']) / ref[col]['std'] if len(vals) > 0 else 0.0
    return p

def profile_to_vec(prof):
    return np.nan_to_num(np.array([prof.get(k, 0) for k in all_metrics], dtype=float), 0)

# ═══════════════════════════════════════════════════════════════
# 3. COVID+ reference from 2020
# ═══════════════════════════════════════════════════════════════
covid_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'covid_test_results.csv'), encoding='utf-8-sig')
positive_pids = set(covid_test[covid_test['status'] == 'positive']['patient_id'].astype(str).unique())
td['pid_str'] = td['住院流水号'].astype(str)

covid_pos_2020 = td[(td['pid_str'].isin(positive_pids)) & (td['year'] == 2020)]
covid_pos_profile = zscore_profile(covid_pos_2020)
vec_pos = profile_to_vec(covid_pos_profile)
print(f"  COVID+ reference: n={len(covid_pos_2020)} admissions, norm={np.linalg.norm(vec_pos):.3f}")

# ═══════════════════════════════════════════════════════════════
# 4. Baseline profiles per comorbidity & monthly monitoring
# ═══════════════════════════════════════════════════════════════
comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']

baseline_profiles = {}
for comor in comorbidities:
    sub = train_data[train_data[comor] == 1]
    if len(sub) >= 10:
        prof = zscore_profile(sub)
        baseline_profiles[comor] = profile_to_vec(prof)

# Monthly monitoring 2019-01 to 2020-06
print("\n  Monthly monitoring (2019-01 to 2020-06)...")
monthly_results = []
monitor_period = [(2019, m) for m in range(1, 13)] + [(2020, m) for m in range(1, 7)]

for yr, mo in monitor_period:
    month_data = {}
    for comor in comorbidities:
        sub = td[(td[comor] == 1) & (td['year'] == yr) & (td['month'] == mo)]
        if len(sub) >= 5:
            prof = zscore_profile(sub)
            v = profile_to_vec(prof)
            sim = 1 - cosine(vec_pos, v) if np.linalg.norm(v) > 0 else np.nan
            if comor in baseline_profiles:
                baseline_sim = 1 - cosine(vec_pos, baseline_profiles[comor]) if np.linalg.norm(baseline_profiles[comor]) > 0 else np.nan
                deviation = sim - baseline_sim if not np.isnan(baseline_sim) else np.nan
            else:
                baseline_sim = np.nan
                deviation = np.nan
        else:
            sim = np.nan
            deviation = np.nan
            baseline_sim = np.nan

        month_data[comor] = {
            'similarity': float(sim) if not np.isnan(sim) else None,
            'baseline_sim': float(baseline_sim) if not np.isnan(baseline_sim) else None,
            'deviation': float(deviation) if not np.isnan(deviation) else None,
            'n': int(len(sub)) if len(sub) >= 5 else 0,
        }

    monthly_results.append({
        'year': yr, 'month': mo,
        'label': f"{yr}-{mo:02d}",
        'comorbidities': month_data,
    })

# ═══════════════════════════════════════════════════════════════
# 5. Compute RDI and traditional surveillance
# ═══════════════════════════════════════════════════════════════
print("\n  Computing Respiratory Dominance Index (RDI)...")

# Traditional surveillance: respiratory admission rate
monthly_resp_rate = []
for yr, mo in monitor_period:
    total = len(td[(td['year'] == yr) & (td['month'] == mo)])
    resp = len(td[(td['Respiratory'] == 1) & (td['year'] == yr) & (td['month'] == mo)])
    rate = resp / total * 100 if total > 0 else 0
    monthly_resp_rate.append({
        'year': yr, 'month': mo,
        'label': f"{yr}-{mo:02d}",
        'total_admissions': total,
        'resp_admissions': resp,
        'resp_rate': rate,
    })

# Seasonal baseline from 2017-2018
seasonal_baseline = {}
for mo in range(1, 13):
    rates = []
    for yr in [2017, 2018]:
        total = len(td[(td['year'] == yr) & (td['month'] == mo)])
        resp = len(td[(td['Respiratory'] == 1) & (td['year'] == yr) & (td['month'] == mo)])
        if total > 0:
            rates.append(resp / total * 100)
    seasonal_baseline[mo] = {
        'mean': np.mean(rates) if rates else 0,
        'std': max(np.std(rates), 1e-6) if rates else 1,
        'upper_2sd': np.mean(rates) + 2 * np.std(rates) if rates else 0,
    }

# RDI per month
for r in monthly_results:
    resp_sim = r['comorbidities']['Respiratory']['similarity']
    other_sims = [r['comorbidities'][c]['similarity'] for c in comorbidities
                  if c != 'Respiratory' and r['comorbidities'][c]['similarity'] is not None]
    if resp_sim is not None and len(other_sims) > 0:
        r['rdi'] = resp_sim - np.mean(other_sims)
    else:
        r['rdi'] = None

# Baseline RDI from 2017-2018
baseline_rdis = []
for yr in [2017, 2018]:
    for mo in range(1, 13):
        sub_resp = td[(td['Respiratory'] == 1) & (td['year'] == yr) & (td['month'] == mo)]
        if len(sub_resp) < 5:
            continue
        prof_resp = zscore_profile(sub_resp)
        v_resp = profile_to_vec(prof_resp)
        if np.linalg.norm(v_resp) == 0:
            continue
        sim_resp = 1 - cosine(vec_pos, v_resp)

        other_sims_bl = []
        for comor in comorbidities:
            if comor == 'Respiratory':
                continue
            sub_c = td[(td[comor] == 1) & (td['year'] == yr) & (td['month'] == mo)]
            if len(sub_c) >= 5:
                prof_c = zscore_profile(sub_c)
                v_c = profile_to_vec(prof_c)
                if np.linalg.norm(v_c) > 0:
                    other_sims_bl.append(1 - cosine(vec_pos, v_c))
        if other_sims_bl:
            baseline_rdis.append(sim_resp - np.mean(other_sims_bl))

rdi_baseline_mean = np.mean(baseline_rdis) if baseline_rdis else 0
rdi_baseline_std = max(np.std(baseline_rdis), 1e-6) if baseline_rdis else 1
rdi_threshold_15sd = rdi_baseline_mean + 1.5 * rdi_baseline_std

print(f"  RDI baseline: mean={rdi_baseline_mean:.3f}, std={rdi_baseline_std:.3f}")
print(f"  RDI threshold (1.5SD): {rdi_threshold_15sd:.3f}")

# Similarity baseline
baseline_monthly_sims = []
for yr in [2017, 2018]:
    for mo in range(1, 13):
        sub = td[(td['Respiratory'] == 1) & (td['year'] == yr) & (td['month'] == mo)]
        if len(sub) >= 5:
            prof = zscore_profile(sub)
            v = profile_to_vec(prof)
            if np.linalg.norm(v) > 0:
                baseline_monthly_sims.append(1 - cosine(vec_pos, v))

sim_p975 = np.percentile(baseline_monthly_sims, 97.5) if baseline_monthly_sims else 0.92

# Detect alerts
pandemic_onset_idx = 12  # 2020-01

behavioral_alerts = []
traditional_alerts = []
first_behavioral_alert = None
first_traditional_alert = None

for i, r in enumerate(monthly_results):
    mo = r['month']
    resp_sim = r['comorbidities']['Respiratory']['similarity']
    rdi = r.get('rdi')
    behavioral_alert = False
    if rdi is not None and rdi > rdi_threshold_15sd:
        behavioral_alert = True
    if resp_sim is not None and resp_sim > sim_p975:
        behavioral_alert = True

    if behavioral_alert:
        alert_val = rdi if rdi is not None else 0
        behavioral_alerts.append({'label': r['label'], 'rdi': alert_val,
                                  'sim': resp_sim, 'idx': i})
        if first_behavioral_alert is None:
            first_behavioral_alert = {'label': r['label'], 'rdi': alert_val,
                                      'sim': resp_sim, 'idx': i}

    trad = monthly_resp_rate[i]
    sb = seasonal_baseline[mo]
    if trad['resp_rate'] > sb['upper_2sd']:
        traditional_alerts.append({'label': r['label'], 'value': trad['resp_rate'],
                                   'threshold': sb['upper_2sd'], 'idx': i})
        if first_traditional_alert is None:
            first_traditional_alert = {'label': r['label'], 'value': trad['resp_rate'],
                                       'threshold': sb['upper_2sd'], 'idx': i}

if first_behavioral_alert:
    lead_behavioral = pandemic_onset_idx - first_behavioral_alert['idx']
    print(f"  Behavioral alert: {first_behavioral_alert['label']} (lead {lead_behavioral} mo)")
if first_traditional_alert:
    lead_traditional = pandemic_onset_idx - first_traditional_alert['idx']
    print(f"  Traditional alert: {first_traditional_alert['label']} (lead {lead_traditional} mo)")

# ═══════════════════════════════════════════════════════════════
# 6. FIGURE: 3-panel Early Warning (Figure 3)
# ═══════════════════════════════════════════════════════════════
print("\n[6] Generating Figure3_early_warning...")

comor_colors = {
    'Respiratory': COLORS['secondary'],
    'Cardiovascular': COLORS['primary'],
    'Hypertension': COLORS['accent1'],
    'Diabetes': COLORS['accent2'],
    'Cerebrovascular': COLORS['accent3'],
    'Renal': COLORS['accent5'],
}

fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(1, 2, wspace=0.35,
                       left=0.07, right=0.96, top=0.92, bottom=0.14)

# ─── Panel A: RDI timeline (left top) ────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_box_aspect(1)
panel_label(ax_a, 'A')

x_months = np.arange(len(monthly_results))
labels = [r['label'] for r in monthly_results]

# RDI line
rdi_vals = [r.get('rdi') for r in monthly_results]
valid_mask_rdi = [v is not None for v in rdi_vals]
valid_x_rdi = [x for x, v in zip(x_months, valid_mask_rdi) if v]
valid_y_rdi = [y for y, v in zip(rdi_vals, valid_mask_rdi) if v]

ax_a.plot(valid_x_rdi, valid_y_rdi, '-o', color=COLORS['secondary'], linewidth=2,
          markersize=5, label='Respiratory Dominance\nIndex (RDI)', zorder=5)
ax_a.axhline(rdi_threshold_15sd, color=COLORS['secondary'], linestyle='--',
             linewidth=1.2, alpha=0.7, label=f'RDI alert (1.5SD={rdi_threshold_15sd:.2f})')
ax_a.fill_between(valid_x_rdi, rdi_threshold_15sd, valid_y_rdi,
                   where=[y > rdi_threshold_15sd for y in valid_y_rdi],
                   alpha=0.15, color=COLORS['secondary'])

# Traditional respiratory rate on secondary axis
ax_a2 = ax_a.twinx()
resp_rates = [r['resp_rate'] for r in monthly_resp_rate]
seasonal_thresholds = [seasonal_baseline[r['month']]['upper_2sd'] for r in monthly_resp_rate]

ax_a2.bar(x_months, resp_rates, width=0.5, alpha=0.3, color=COLORS['accent2'],
          label='Respiratory rate (%)')
ax_a2.plot(x_months, seasonal_thresholds, ':', color=COLORS['accent2'],
           linewidth=1, alpha=0.7, label='Seasonal threshold (2SD)')
ax_a2.set_ylabel('Respiratory Admission Rate (%)', color=COLORS['accent2'], fontsize=8)
ax_a2.tick_params(axis='y', labelcolor=COLORS['accent2'])

# Pandemic onset line
ax_a.axvline(pandemic_onset_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax_a.text(pandemic_onset_idx + 0.2, 0.95, 'Pandemic\nOnset',
          fontsize=7, color='red', va='top', transform=ax_a.get_xaxis_transform())

# First behavioral alert annotation
if first_behavioral_alert:
    ax_a.axvline(first_behavioral_alert['idx'], color=COLORS['secondary'],
                 linestyle='-', linewidth=1.5, alpha=0.5)
    ax_a.annotate(f"Behavioral alert\n{first_behavioral_alert['label']}",
                  xy=(first_behavioral_alert['idx'], first_behavioral_alert['rdi']),
                  xytext=(max(0, first_behavioral_alert['idx'] - 3),
                          first_behavioral_alert['rdi'] + 0.15),
                  fontsize=7, fontweight='bold', color=COLORS['secondary'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.2),
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=COLORS['secondary'], alpha=0.9))

ax_a.set_xticks(x_months[::2])
ax_a.set_xticklabels([labels[i] for i in range(0, len(labels), 2)],
                     fontsize=6.5, rotation=45, ha='right')
ax_a.set_ylabel('Cosine Similarity', fontsize=9)
ax_a.set_title('Behavioral vs. Traditional Surveillance Timeline', fontsize=10)
ax_a.legend(loc='upper left', fontsize=6.5, framealpha=0.9)
ax_a2.legend(loc='upper right', fontsize=6.5, framealpha=0.9)

# ─── Panel B: Deviation heatmap ──────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_box_aspect(1)
panel_label(ax_b, 'B')

dev_matrix = np.full((len(comorbidities), len(monthly_results)), np.nan)
for j, r in enumerate(monthly_results):
    for i, comor in enumerate(comorbidities):
        dev = r['comorbidities'][comor].get('deviation')
        if dev is not None:
            dev_matrix[i, j] = dev

im = ax_b.imshow(dev_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-0.3, vmax=0.3)
ax_b.set_xticks(range(0, len(monthly_results), 2))
ax_b.set_xticklabels([monthly_results[i]['label'] for i in range(0, len(monthly_results), 2)],
                     fontsize=6, rotation=45, ha='right')
ax_b.set_yticks(range(len(comorbidities)))
ax_b.set_yticklabels(comorbidities, fontsize=8)
ax_b.set_title('Behavioral Deviation from 2017-2018 Baseline', fontsize=10)
ax_b.axvline(pandemic_onset_idx - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
ax_b.text(pandemic_onset_idx, -0.6, 'Pandemic', fontsize=7, color='red', ha='center')

cbar = plt.colorbar(im, ax=ax_b, fraction=0.02, pad=0.03)
cbar.set_label('Deviation from baseline', fontsize=7)
cbar.ax.tick_params(labelsize=6)

plt.tight_layout()
save_figure(fig, 'Figure3_early_warning')

print("\nDone — Figure3_early_warning saved.")
