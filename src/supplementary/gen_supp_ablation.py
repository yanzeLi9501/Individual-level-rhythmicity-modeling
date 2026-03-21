"""
Supplementary Figure S12: Behavioral Dimension Ablation Analysis
+ Table S4: Data Completeness Summary

Generates:
  - FigureS12_dimension_ablation (PNG/PDF/TIF)
  - Prints Table S4 data completeness to stdout

Run: python gen_supp_ablation.py
"""
import sys, io, warnings, os
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
print("Supplementary: Dimension Ablation + Data Completeness")
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

# ═══════════════════════════════════════════════════════════════
# 2. Behavioral profile infrastructure
# ═══════════════════════════════════════════════════════════════
behavior_cols = ['实际住院天数', '检验项目数', '医嘱数量', '药品医嘱', '检查数量']
lab_cols = ['lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']
all_metrics = behavior_cols + lab_cols

# Readable English names for the 13 dimensions
dim_labels = [
    'LOS', 'Lab Tests', 'Orders', 'Drug Orders', 'Examinations',
    'WBC', 'CRP', 'HGB', 'ALB', 'Creatinine', 'Glucose', 'K', 'Na'
]

# Group labels
dim_groups = ['Utilization'] * 5 + ['Laboratory'] * 8

baseline = td[(td['year'] >= 2016) & (td['year'] <= 2018)]
ref = {}
for col in all_metrics:
    vals = pd.to_numeric(baseline[col], errors='coerce').dropna()
    ref[col] = {'mean': vals.mean(), 'std': max(vals.std(), 1e-6)}

def zscore_profile(sub_df):
    p = {}
    for col in all_metrics:
        vals = pd.to_numeric(sub_df[col], errors='coerce').dropna()
        p[col] = (vals.mean() - ref[col]['mean']) / ref[col]['std'] if len(vals) > 0 else 0.0
    return p

def profile_to_vec(prof):
    v = np.array([prof.get(k, 0) for k in all_metrics], dtype=float)
    return np.nan_to_num(v, 0)

# ═══════════════════════════════════════════════════════════════
# 3. COVID+ reference
# ═══════════════════════════════════════════════════════════════
covid_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'covid_test_results.csv'), encoding='utf-8-sig')
positive_pids = set(covid_test[covid_test['status'] == 'positive']['patient_id'].astype(str).unique())
td['pid_str'] = td['住院流水号'].astype(str)

covid_pos_2020 = td[(td['pid_str'].isin(positive_pids)) & (td['year'] == 2020)]
covid_pos_profile = zscore_profile(covid_pos_2020)
vec_pos = profile_to_vec(covid_pos_profile)
print(f"  COVID+ reference: n={len(covid_pos_2020)} admissions")

# ═══════════════════════════════════════════════════════════════
# 4. Dimension ablation analysis
# ═══════════════════════════════════════════════════════════════
print("\n[2] Dimension ablation analysis...")

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']

# Full similarity (Q4 2019) per comorbidity
q4_2019 = td[(td['year'] == 2019) & (td['month'].isin([10, 11, 12]))]

full_sims = {}
for comor in comorbidities:
    sub = q4_2019[q4_2019[comor] == 1]
    if len(sub) >= 5:
        prof = zscore_profile(sub)
        v = profile_to_vec(prof)
        full_sims[comor] = 1 - cosine(vec_pos, v) if np.linalg.norm(v) > 0 else 0

print(f"  Full 13-dim respiratory similarity (Q4 2019): {full_sims.get('Respiratory', 0):.3f}")

# Leave-one-out ablation: remove one dimension at a time
ablation_results = []
for dim_idx in range(len(all_metrics)):
    # Create masked vectors (zero out one dimension)
    mask = np.ones(len(all_metrics), dtype=bool)
    mask[dim_idx] = False

    vec_pos_masked = vec_pos.copy()
    vec_pos_masked[dim_idx] = 0

    dim_sims = {}
    for comor in comorbidities:
        sub = q4_2019[q4_2019[comor] == 1]
        if len(sub) >= 5:
            prof = zscore_profile(sub)
            v = profile_to_vec(prof)
            v_masked = v.copy()
            v_masked[dim_idx] = 0
            dim_sims[comor] = 1 - cosine(vec_pos_masked, v_masked) if np.linalg.norm(v_masked) > 0 else 0

    # Compute drop in respiratory similarity and change in respiratory rank
    resp_drop = full_sims.get('Respiratory', 0) - dim_sims.get('Respiratory', 0)

    # Check if respiratory is still #1
    sorted_comors = sorted(dim_sims.items(), key=lambda x: x[1], reverse=True)
    resp_rank = next((i+1 for i, (c, _) in enumerate(sorted_comors) if c == 'Respiratory'), len(comorbidities))

    ablation_results.append({
        'dim_idx': dim_idx,
        'dim_name': dim_labels[dim_idx],
        'dim_col': all_metrics[dim_idx],
        'resp_sim_ablated': dim_sims.get('Respiratory', 0),
        'resp_drop': resp_drop,
        'resp_rank': resp_rank,
        'group': dim_groups[dim_idx],
    })

    print(f"    Remove {dim_labels[dim_idx]:>12s}: resp_sim={dim_sims.get('Respiratory', 0):.3f}, "
          f"drop={resp_drop:+.3f}, rank={resp_rank}")

# ═══════════════════════════════════════════════════════════════
# 5. Group ablation (remove all utilization or all lab)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Group ablation...")

group_results = {}
for group_name, indices in [('Utilization only', list(range(5, 13))),
                             ('Laboratory only', list(range(0, 5))),
                             ('Full 13-dim', [])]:
    vec_pos_g = vec_pos.copy()
    for idx in indices:
        vec_pos_g[idx] = 0

    g_sims = {}
    for comor in comorbidities:
        sub = q4_2019[q4_2019[comor] == 1]
        if len(sub) >= 5:
            prof = zscore_profile(sub)
            v = profile_to_vec(prof)
            v_g = v.copy()
            for idx in indices:
                v_g[idx] = 0
            g_sims[comor] = 1 - cosine(vec_pos_g, v_g) if np.linalg.norm(v_g) > 0 else 0

    sorted_g = sorted(g_sims.items(), key=lambda x: x[1], reverse=True)
    resp_rank_g = next((i+1 for i, (c, _) in enumerate(sorted_g) if c == 'Respiratory'), len(comorbidities))

    group_results[group_name] = {
        'resp_sim': g_sims.get('Respiratory', 0),
        'resp_rank': resp_rank_g,
        'all_sims': g_sims,
    }
    print(f"  {group_name}: resp_sim={g_sims.get('Respiratory', 0):.3f}, rank={resp_rank_g}")

# ═══════════════════════════════════════════════════════════════
# 6. Generate Figure S12: Ablation Analysis (1×2)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Generating FigureS4_dimension_ablation...")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, wspace=0.35,
                       left=0.08, right=0.96, top=0.90, bottom=0.18)

# ─── Panel A: Leave-one-out ablation bar chart ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

# Sort by absolute drop
ablation_sorted = sorted(ablation_results, key=lambda x: abs(x['resp_drop']), reverse=True)
names = [r['dim_name'] for r in ablation_sorted]
drops = [r['resp_drop'] for r in ablation_sorted]
colors_bar = [COLORS['secondary'] if r['group'] == 'Utilization' else COLORS['primary']
              for r in ablation_sorted]

y_pos = np.arange(len(names))
bars = ax_a.barh(y_pos, drops, color=colors_bar, edgecolor='white', height=0.7)

# Add value labels
for bar, drop in zip(bars, drops):
    x_pos = bar.get_width()
    ax_a.text(x_pos + 0.002 if x_pos >= 0 else x_pos - 0.002,
              bar.get_y() + bar.get_height() / 2,
              f'{drop:+.3f}', va='center',
              ha='left' if x_pos >= 0 else 'right',
              fontsize=7, color='black')

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(names, fontsize=8)
ax_a.set_xlabel('Change in Respiratory Similarity\n(Full - Ablated)', fontsize=9)
ax_a.set_title('Leave-One-Out Dimension Ablation', fontsize=11)
ax_a.axvline(0, color='black', linewidth=0.5, linestyle='-')
ax_a.invert_yaxis()

# Legend for groups
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['secondary'], label='Utilization (5 dims)'),
    Patch(facecolor=COLORS['primary'], label='Laboratory (8 dims)')
]
ax_a.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

# ─── Panel B: Group ablation comparison ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

group_names = ['Full 13-dim', 'Utilization only', 'Laboratory only']
comor_short = {'Cardiovascular': 'Cardio.', 'Hypertension': 'Hypert.',
               'Diabetes': 'Diabetes', 'Cerebrovascular': 'Cerebro.',
               'Renal': 'Renal', 'Respiratory': 'Resp.'}

x = np.arange(len(comorbidities))
width = 0.25
group_colors = [COLORS['accent3'], COLORS['secondary'], COLORS['primary']]

for i, gname in enumerate(group_names):
    sims = [group_results[gname]['all_sims'].get(c, 0) for c in comorbidities]
    bars_g = ax_b.bar(x + (i - 1) * width, sims, width,
                      label=gname, color=group_colors[i],
                      edgecolor='white', alpha=0.85)

ax_b.set_xlabel('Comorbidity Group', fontsize=9)
ax_b.set_ylabel('Cosine Similarity to COVID+ Reference', fontsize=9)
ax_b.set_title('Feature Group Ablation (Q4 2019)', fontsize=11)
ax_b.set_xticks(x)
ax_b.set_xticklabels([comor_short.get(c, c) for c in comorbidities],
                      fontsize=8, rotation=20, ha='right')
ax_b.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax_b.set_ylim(0, 1.05)

# Highlight respiratory group
ax_b.axvspan(x[comorbidities.index('Respiratory')] - 0.45,
             x[comorbidities.index('Respiratory')] + 0.45,
             alpha=0.08, color=COLORS['secondary'])

save_figure(fig, 'FigureS4_dimension_ablation')
print("  FigureS4_dimension_ablation saved.")

# ═══════════════════════════════════════════════════════════════
# 7. Data Completeness Summary (Table S4)
# ═══════════════════════════════════════════════════════════════
print("\n[5] Data completeness analysis (Table S4)...")

# Check all key columns for missingness
check_cols = {
    'Gap (days)': 'incoming_gap',
    'LOS (days)': '实际住院天数',
    'Lab: WBC': 'lab_WBC',
    'Lab: CRP': 'lab_CRP',
    'Lab: HGB': 'lab_HGB',
    'Lab: ALB': 'lab_ALB',
    'Lab: Creatinine': 'lab_CREA',
    'Lab: Glucose': 'lab_GLU',
    'Lab: K': 'lab_K',
    'Lab: Na': 'lab_Na',
    'Lab Test Count': '检验项目数',
    'Order Count': '医嘱数量',
    'Drug Orders': '药品医嘱',
    'Exam Count': '检查数量',
    'Diagnosis Text': 'EMR_初步诊断',
    'Total Cost': '总费用',
}

completeness_data = []
for label, col in check_cols.items():
    if col in td.columns:
        total = len(td)
        if td[col].dtype == 'object':
            n_missing = td[col].isna().sum() + (td[col] == '').sum()
        else:
            vals = pd.to_numeric(td[col], errors='coerce')
            n_missing = vals.isna().sum()
        n_valid = total - n_missing
        pct = n_valid / total * 100
        completeness_data.append({
            'Feature': label,
            'Total': int(total),
            'Valid': int(n_valid),
            'Missing': int(n_missing),
            'Completeness (%)': f'{pct:.1f}',
        })
    else:
        completeness_data.append({
            'Feature': label,
            'Total': int(len(td)),
            'Valid': 0,
            'Missing': int(len(td)),
            'Completeness (%)': '0.0 (column absent)',
        })

print("\n  Table S4: Data Completeness Summary")
print(f"  {'Feature':<20s} {'Total':>8s} {'Valid':>8s} {'Missing':>8s} {'Complete%':>10s}")
print("  " + "-" * 60)
for row in completeness_data:
    print(f"  {row['Feature']:<20s} {row['Total']:>8,d} {row['Valid']:>8,d} "
          f"{row['Missing']:>8,d} {row['Completeness (%)']:>10s}")

# Save completeness as JSON for gen_paper_v2.py to reference
import json
completeness_path = os.path.join(OUTPUT_DIR, 'data_completeness.json')
with open(completeness_path, 'w', encoding='utf-8') as f:
    json.dump(completeness_data, f, indent=2, ensure_ascii=False)
print(f"\n  Saved completeness data: {completeness_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
