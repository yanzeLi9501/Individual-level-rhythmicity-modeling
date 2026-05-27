#!/usr/bin/env python3
"""
Generate updated FigureS3_dimension_ablation with COVID-cardiac exclusion panel.

Panels:
  A - Leave-One-Out Dimension Ablation
  B - Feature Group Ablation (Q4 2019)
  C - COVID-cardiac exclusion breakdown (CARMEN-I, Cardiac-42k WHU)

Saves to: NC_revision/submit/figures/FigureS3_dimension_ablation.{png,pdf,tif}

Run from: NC_revision/ directory
  python gen_figure_s3_updated.py
"""

import sys, io, warnings, os, json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = r'data\readmission_output\all_admissions.csv'
CARMEN_JSON = os.path.join(BASE, 'external_positive_control_results', 'carmen_i_cardiac_summary.json')
WHU_JSON    = os.path.join(BASE, 'external_positive_control_results', 'whu_covid_cardiac_summary.json')
OUT_DIR      = os.path.join(BASE, 'submit', 'figures')
DEEP_OUT_DIR = os.path.join(BASE, 'DeepseekRevision', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEEP_OUT_DIR, exist_ok=True)

# ── Style constants (inline from fig_config) ───────────────────────────────────
COLORS = {
    'primary':   '#4A90D9',
    'secondary': '#E8636E',
    'accent1':   '#5BBD8C',
    'accent2':   '#F5A623',
    'accent3':   '#9B7ED8',
    'accent4':   '#F48FB1',
    'accent5':   '#4DD0E1',
    'grid':      '#DDDDDD',
}
SEGMENT_COLORS = {
    'De-novo COVID-cardiac': '#CC2529',
    'COVID exacerbation\n(pre-existing cardiac)': '#F5A623',
    'Pre-existing cardiac\n(no acute COVID event)': '#9B7ED8',
    'No cardiac involvement': '#5BBD8C',
    'Other cardiac mention': '#AAAAAA',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def panel_label(ax, label, x=0.02, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left', clip_on=False)

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

print("=" * 60)
print("FigureS3 updated: Ablation + COVID-cardiac exclusion")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data for ablation panels
# ═══════════════════════════════════════════════════════════════
print("[1] Loading admission data for ablation...")
td = pd.read_csv(DATA_CSV, low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year']  = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month

COMORBIDITY_PATTERNS = {
    'Cardiovascular': r'冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入',
    'Hypertension':   r'高血压',
    'Diabetes':       r'糖尿病|血糖',
    'Cerebrovascular':r'脑梗|脑出血|脑血管|脑卒中|中风|腔隙性',
    'Renal':          r'肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏',
    'Respiratory':    r'肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭',
}
diag_text = td['EMR_初步诊断'].fillna('').astype(str)
for name, pattern in COMORBIDITY_PATTERNS.items():
    td[name] = diag_text.str.contains(pattern, na=False).astype(int)

print(f"  Total admissions: {len(td):,}")

behavior_cols = ['实际住院天数', '检验项目数', '医嘱数量', '药品医嘱', '检查数量']
lab_cols = ['lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']
all_metrics = behavior_cols + lab_cols

dim_labels = [
    'LOS', 'Lab Tests', 'Orders', 'Drug Orders', 'Examinations',
    'WBC', 'CRP', 'HGB', 'ALB', 'Creatinine', 'Glucose', 'K', 'Na'
]
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
    return np.nan_to_num(np.array([prof.get(k, 0) for k in all_metrics], dtype=float), 0)

def cos_sim(v1, v2):
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        return 1 - cosine(v1, v2)
    return 0.0

# COVID+ reference
try:
    covid_test = pd.read_csv(
        os.path.join(os.path.dirname(DATA_CSV), 'covid_test_results.csv'),
        encoding='utf-8-sig')
    positive_pids = set(covid_test[covid_test['status'] == 'positive']['patient_id'].astype(str).unique())
    td['pid_str'] = td['住院流水号'].astype(str)
    covid_pos = td[(td['pid_str'].isin(positive_pids)) & (td['year'] == 2020)]
    vec_covid = profile_to_vec(zscore_profile(covid_pos))
    print(f"  COVID+ reference: n={len(covid_pos)}")
except Exception as e:
    print(f"  WARNING: could not load COVID+ reference: {e}. Using 2020 Q1 as proxy.")
    covid_pos = td[(td['year'] == 2020) & (td['month'] <= 4)]
    vec_covid = profile_to_vec(zscore_profile(covid_pos))

# ═══════════════════════════════════════════════════════════════
# 2. Panel A: Leave-One-Out Dimension Ablation
# ═══════════════════════════════════════════════════════════════
print("\n[2] Computing leave-one-out dimension ablation...")
q4_2019 = td[(td['year'] == 2019) & (td['month'] >= 10)]
vec_full = profile_to_vec(zscore_profile(q4_2019))
sim_full = cos_sim(vec_covid, vec_full)

ablation_results = []
for i, (col, label, group) in enumerate(zip(all_metrics, dim_labels, dim_groups)):
    remaining = [c for c in all_metrics if c != col]
    vec_ablated = np.array([zscore_profile(q4_2019).get(c, 0) for c in remaining])
    vec_ref_ablated = np.array([zscore_profile(covid_pos if hasattr(covid_pos, 'iterrows') else td).get(c, 0) for c in remaining])
    # Use full vectors, zeroing out the ablated dimension
    v_full_zeroed = vec_full.copy()
    v_full_zeroed[i] = 0.0
    v_ref_zeroed = vec_covid.copy()
    v_ref_zeroed[i] = 0.0
    sim_ablated = cos_sim(v_ref_zeroed, v_full_zeroed)
    delta = sim_full - sim_ablated
    ablation_results.append({'label': label, 'group': group, 'delta': delta})

# Sort by absolute delta (largest = most important)
ablation_df = pd.DataFrame(ablation_results)
ablation_df = ablation_df.sort_values('delta', ascending=True)  # for horizontal bar

# ═══════════════════════════════════════════════════════════════
# 3. Panel B: Feature Group Ablation (Q4 2019)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Computing feature group ablation (Q4 2019)...")
comorbidities = list(COMORBIDITY_PATTERNS.keys())

def group_vec(sub_df, dims):
    p = zscore_profile(sub_df)
    return np.array([p.get(c, 0) for c in dims])

group_ablation = {g: {} for g in comorbidities}
dims_util = behavior_cols
dims_lab  = lab_cols

for comor in comorbidities:
    sub = q4_2019[q4_2019[comor] == 1]
    if len(sub) < 5:
        continue
    v_full  = group_vec(sub, all_metrics)
    v_ref   = group_vec(covid_pos if hasattr(covid_pos, 'iterrows') else td, all_metrics)
    v_util  = group_vec(sub, dims_util)
    v_rutil = group_vec(covid_pos if hasattr(covid_pos, 'iterrows') else td, dims_util)
    v_lab   = group_vec(sub, dims_lab)
    v_rlab  = group_vec(covid_pos if hasattr(covid_pos, 'iterrows') else td, dims_lab)
    group_ablation[comor] = {
        'Full':        cos_sim(v_ref, v_full),
        'Utilization': cos_sim(v_rutil, v_util),
        'Laboratory':  cos_sim(v_rlab, v_lab),
    }

# ═══════════════════════════════════════════════════════════════
# 4. Panel C: COVID-cardiac exclusion statistics
# ═══════════════════════════════════════════════════════════════
print("\n[4] Loading COVID-cardiac exclusion statistics...")
with open(CARMEN_JSON, encoding='utf-8') as f:
    carmen = json.load(f)
with open(WHU_JSON, encoding='utf-8') as f:
    whu = json.load(f)

# Prepare stacked bar data
# Categories: de-novo, exacerbation, pre-existing only, no cardiac, other (any but uncat.)
datasets = {
    'CARMEN-I\n(all COVID-19, n=151)': {
        'De-novo COVID-cardiac':            carmen['de_novo_covid_cardiac_pct'],
        'COVID exacerbation\n(pre-existing cardiac)': carmen['covid_exacerbation_of_preexisting_pct'],
        'Pre-existing cardiac\n(no acute COVID event)': carmen['preexisting_only_no_acute_pct'],
        'No cardiac involvement':           carmen['no_cardiac_pct'],
        'Other cardiac mention':            round(100 - carmen['de_novo_covid_cardiac_pct']
                                                    - carmen['covid_exacerbation_of_preexisting_pct']
                                                    - carmen['preexisting_only_no_acute_pct']
                                                    - carmen['no_cardiac_pct'], 1),
    },
    'Cardiac-42k\n(COVID admissions, n=281)': {
        'De-novo COVID-cardiac':            whu['de_novo_covid_cardiac_pct'],
        'COVID exacerbation\n(pre-existing cardiac)': whu['exacerbation_existing_pct'],
        'Pre-existing cardiac\n(no acute COVID event)': whu['preexisting_only_pct'],
        'No cardiac involvement':           whu['no_cardiac_pct'],
        'Other cardiac mention':            0.0,
    },
}

print(f"  CARMEN-I: de-novo={carmen['de_novo_covid_cardiac_pct']}%,  "
      f"no-cardiac={carmen['no_cardiac_pct']}%")
print(f"  Cardiac-42k COVID: de-novo={whu['de_novo_covid_cardiac_pct']}%,  "
      f"exacerbation={whu['exacerbation_existing_pct']}%")

# ═══════════════════════════════════════════════════════════════
# 5. Generate combined 3-panel figure
# ═══════════════════════════════════════════════════════════════
print("\n[5] Generating Figure S3 (3-panel)...")

fig = plt.figure(figsize=(20, 7))
gs = gridspec.GridSpec(1, 3, wspace=0.38,
                       left=0.06, right=0.97, top=0.97, bottom=0.18)

comor_labels_short = ['Cardio.', 'Hypert.', 'Diabetes', 'Cerebro.', 'Renal', 'Resp.']

# ─── Panel A: Dimension Ablation ───────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

colors_a = [COLORS['secondary'] if g == 'Utilization' else COLORS['primary']
            for g in ablation_df['group']]
bars = ax_a.barh(range(len(ablation_df)), ablation_df['delta'], color=colors_a, height=0.7)
ax_a.set_yticks(range(len(ablation_df)))
ax_a.set_yticklabels(ablation_df['label'], fontsize=8)
ax_a.axvline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.5)
for i, (val, bar) in enumerate(zip(ablation_df['delta'], bars)):
    offset = 0.003 if val >= 0 else -0.003
    ha = 'left' if val >= 0 else 'right'
    ax_a.text(val + offset, bar.get_y() + bar.get_height() / 2,
              f'{val:+.3f}', va='center', ha=ha, fontsize=7)
ax_a.set_xlabel('Change in Respiratory Similarity\n(Full − Ablated)', fontsize=8)
ax_a.set_title('Leave-One-Out Dimension Ablation', fontsize=9)
patch_u = mpatches.Patch(color=COLORS['secondary'], label='Utilization (5 dims)')
patch_l = mpatches.Patch(color=COLORS['primary'], label='Laboratory (8 dims)')
ax_a.legend(handles=[patch_u, patch_l], fontsize=7, loc='lower right', framealpha=0.9)

# ─── Panel B: Feature Group Ablation ───────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

n_comor = len(comor_labels_short)
x = np.arange(n_comor)
w = 0.26
for j, (subset_name, color, offset) in enumerate([
    ('Full', COLORS['accent3'], -w),
    ('Utilization', COLORS['secondary'], 0),
    ('Laboratory', COLORS['primary'], w),
]):
    vals = [group_ablation.get(c, {}).get(subset_name, 0) for c in comorbidities]
    ax_b.bar(x + offset, vals, width=w, color=color, label=subset_name + (' 13-dim' if subset_name == 'Full' else ' only'),
             alpha=0.9, edgecolor='white', linewidth=0.5)

ax_b.set_xticks(x)
ax_b.set_xticklabels(comor_labels_short, fontsize=8)
ax_b.set_ylabel('Pearson profile-correlation to COVID+ reference', fontsize=8)
ax_b.set_title('Feature Group Ablation (Q4 2019)', fontsize=9)
ax_b.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax_b.set_ylim(0, 1.05)
ax_b.axhline(0, color='grey', linewidth=0.5, linestyle=':')

# ─── Panel C: COVID-cardiac exclusion breakdown ─────────────────
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, 'C')

cat_names = [
    'De-novo COVID-cardiac',
    'COVID exacerbation\n(pre-existing cardiac)',
    'Pre-existing cardiac\n(no acute COVID event)',
    'No cardiac involvement',
    'Other cardiac mention',
]
cat_colors = ['#CC2529', '#F5A623', '#9B7ED8', '#5BBD8C', '#AAAAAA']

dataset_labels = list(datasets.keys())
x_c = np.arange(len(dataset_labels))
bottoms = np.zeros(len(dataset_labels))

bars_c = []
for cat, col in zip(cat_names, cat_colors):
    heights = [datasets[ds][cat] for ds in dataset_labels]
    bar = ax_c.bar(x_c, heights, bottom=bottoms, color=col, label=cat.replace('\n', ' '),
                   edgecolor='white', linewidth=0.5, width=0.5)
    # Add value labels for notable segments
    for xi, (h, b) in enumerate(zip(heights, bottoms)):
        if h >= 3.0:
            ax_c.text(xi, b + h / 2, f'{h:.1f}%', ha='center', va='center',
                      fontsize=7.5, fontweight='bold' if cat == 'De-novo COVID-cardiac' else 'normal',
                      color='white' if col in ('#CC2529', '#4A90D9', '#5BBD8C') else 'black')
    bars_c.append(bar)
    bottoms += np.array(heights)

ax_c.set_xticks(x_c)
ax_c.set_xticklabels(dataset_labels, fontsize=8)
ax_c.set_ylabel('Proportion (%)', fontsize=8)
ax_c.set_title('COVID-Related Cardiac Involvement\nin Validation Cohorts', fontsize=9)
ax_c.set_ylim(0, 105)
ax_c.axhline(100, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)

# Legend (only meaningful categories)
legend_patches = [mpatches.Patch(color=col, label=cat.replace('\n', ' '))
                  for cat, col in zip(cat_names, cat_colors) if cat != 'Other cardiac mention'
                  or any(datasets[ds]['Other cardiac mention'] > 1 for ds in dataset_labels)]
ax_c.legend(handles=legend_patches, fontsize=6.5, loc='upper right',
            bbox_to_anchor=(1.0, 0.98), framealpha=0.9, ncol=1)

# Annotation arrow pointing to de-novo bar
ax_c.annotate(
    f'De-novo\n{carmen["de_novo_covid_cardiac_pct"]:.1f}%',
    xy=(0, carmen['de_novo_covid_cardiac_pct'] / 2),
    xytext=(-0.35, 25),
    fontsize=7, color='#CC2529', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#CC2529', lw=1.2),
    ha='center',
)
ax_c.annotate(
    f'De-novo\n{whu["de_novo_covid_cardiac_pct"]:.1f}%',
    xy=(1, whu['de_novo_covid_cardiac_pct'] / 2),
    xytext=(1.35, 25),
    fontsize=7, color='#CC2529', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#CC2529', lw=1.2),
    ha='center',
)

save_figure(fig, 'FigureS3_dimension_ablation')
plt.close(fig)
print("\nDone!")
