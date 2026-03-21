"""
Supplementary Figure S3: Best Model Analysis (6-panel)
  (A) Gap R² by cohort for top 5 models
  (B) LOS R² by cohort for top 5 models
  (C) Best ML vs. deep learning comparison
  (D) Subgroup R² stability: XGBoost vs. LightGBM
  (E) Impact of comorbidity vs. treatment stratification on R² range
  (F) Cross-cohort performance degradation

Generates:
  - FigureS3_best_model_analysis (PNG/PDF/TIF)

Run: python gen_supp_best_model.py
"""
import sys, io, warnings, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure S10: Best Model Analysis")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("[1] Loading model comparison data...")
MC_DIR = os.path.join(OUTPUT_DIR, 'model_comparison')

mc_table = pd.read_csv(os.path.join(MC_DIR, 'model_comparison_table.csv'))
multicohort = pd.read_csv(os.path.join(MC_DIR, 'multicohort_comparison.csv'))
subgroup = pd.read_csv(os.path.join(MC_DIR, 'subgroup_stability.csv'))

with open(os.path.join(MC_DIR, 'best_model_report.json'), encoding='utf-8') as f:
    best_report = json.load(f)

print(f"  Model table: {len(mc_table)} rows")
print(f"  Multicohort: {len(multicohort)} rows")
print(f"  Subgroup: {len(subgroup)} rows")

# ═══════════════════════════════════════════════════════════════
# 2. Prepare data subsets
# ═══════════════════════════════════════════════════════════════

# Top 5 models by gap R² on WHU (from mc_table)
gap_models = mc_table[mc_table['Task'] == 'gap'].sort_values('R2', ascending=False)
top5_gap = gap_models.head(5)['Model'].values

los_models = mc_table[mc_table['Task'] == 'los'].sort_values('R2', ascending=False)
top5_los = los_models.head(5)['Model'].values

# Combine top models (union)
top_models_set = list(dict.fromkeys(list(top5_gap) + list(top5_los)))

# Multicohort data: model name column may differ
# Standardize model names between tables
mc_model_col = 'Model'
datasets = multicohort['Dataset'].unique()
tasks = ['gap', 'los']

# ML vs DL categories
ml_cats = {'ML', 'Boosting'}
dl_cats = {'Deep Learning', 'Deep Learning (Opt)'}

print(f"  Top 5 gap models: {list(top5_gap)}")
print(f"  Top 5 LOS models: {list(top5_los)}")
print(f"  Datasets: {list(datasets)}")

# ═══════════════════════════════════════════════════════════════
# 3. Generate Figure S3 (3×2)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Generating FigureS9_best_model_analysis...")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, wspace=0.30, hspace=0.40,
                       left=0.07, right=0.97, top=0.95, bottom=0.06)

cohort_colors = {
    'WHU': COLORS['primary'],
    'SCH': COLORS['accent1'],
    'MIMIC-IV': COLORS['accent3'],
}

# ─── Panel A: Top 5 Gap R² by cohort ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

gap_mc = multicohort[multicohort['Task'] == 'gap']
models_a = [m for m in top5_gap if m in gap_mc['Model'].values]
if not models_a:
    models_a = gap_mc.sort_values('R2', ascending=False)['Model'].unique()[:5]

x = np.arange(len(models_a))
width = 0.25
for i, ds in enumerate(datasets):
    sub = gap_mc[gap_mc['Dataset'] == ds]
    vals = [sub[sub['Model'] == m]['R2'].values[0] if m in sub['Model'].values else 0 for m in models_a]
    errs = [sub[sub['Model'] == m]['R2_std'].values[0] if m in sub['Model'].values else 0 for m in models_a]
    ax_a.bar(x + i * width, vals, width, yerr=errs, capsize=2,
             color=cohort_colors.get(ds, PAL[i]), alpha=0.85, label=ds, edgecolor='white')

ax_a.set_xticks(x + width)
ax_a.set_xticklabels(models_a, fontsize=7, rotation=20, ha='right')
ax_a.set_ylabel('R²', fontsize=9)
ax_a.set_title('Top 5 Models: Gap Prediction R² by Cohort', fontsize=10)
ax_a.legend(fontsize=7, framealpha=0.9)

# ─── Panel B: Top 5 LOS R² by cohort ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

los_mc = multicohort[multicohort['Task'] == 'los']
models_b = [m for m in top5_los if m in los_mc['Model'].values]
if not models_b:
    models_b = los_mc.sort_values('R2', ascending=False)['Model'].unique()[:5]

x = np.arange(len(models_b))
for i, ds in enumerate(datasets):
    sub = los_mc[los_mc['Dataset'] == ds]
    vals = [sub[sub['Model'] == m]['R2'].values[0] if m in sub['Model'].values else 0 for m in models_b]
    errs = [sub[sub['Model'] == m]['R2_std'].values[0] if m in sub['Model'].values else 0 for m in models_b]
    ax_b.bar(x + i * width, vals, width, yerr=errs, capsize=2,
             color=cohort_colors.get(ds, PAL[i]), alpha=0.85, label=ds, edgecolor='white')

ax_b.set_xticks(x + width)
ax_b.set_xticklabels(models_b, fontsize=7, rotation=20, ha='right')
ax_b.set_ylabel('R²', fontsize=9)
ax_b.set_title('Top 5 Models: LOS Prediction R² by Cohort', fontsize=10)
ax_b.legend(fontsize=7, framealpha=0.9)

# ─── Panel C: ML vs Deep Learning ───
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, 'C')

for task_i, task in enumerate(tasks):
    mc_task = multicohort[multicohort['Task'] == task]
    ml_vals = mc_task[mc_task['Category'].isin(ml_cats)]['R2'].values
    dl_vals = mc_task[mc_task['Category'].isin(dl_cats)]['R2'].values

    positions = [task_i * 3, task_i * 3 + 1]
    bp = ax_c.boxplot([ml_vals, dl_vals], positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='white', markersize=4))
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLORS['secondary'])
    bp['boxes'][1].set_alpha(0.7)

ax_c.set_xticks([0.5, 3.5])
ax_c.set_xticklabels(['Gap Prediction', 'LOS Prediction'], fontsize=9)
ax_c.set_ylabel('R²', fontsize=9)
ax_c.set_title('ML (Boosting) vs. Deep Learning', fontsize=10)

from matplotlib.patches import Patch
ax_c.legend(handles=[
    Patch(facecolor=COLORS['primary'], alpha=0.7, label='ML / Boosting'),
    Patch(facecolor=COLORS['secondary'], alpha=0.7, label='Deep Learning'),
], fontsize=7, framealpha=0.9)

# ─── Panel D: XGBoost vs LightGBM stability ───
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, 'D')

sg_gap = subgroup[subgroup['Task'] == 'gap']
subgroups_unique = sg_gap['Subgroup'].unique()

xgb_vals = []
lgb_vals = []
labels_d = []
for sg in subgroups_unique:
    sg_sub = sg_gap[sg_gap['Subgroup'] == sg]
    xgb_row = sg_sub[sg_sub['Model'] == 'XGBoost']
    lgb_row = sg_sub[sg_sub['Model'] == 'LightGBM']
    if len(xgb_row) > 0 and len(lgb_row) > 0:
        xgb_vals.append(xgb_row['r2'].values[0])
        lgb_vals.append(lgb_row['r2'].values[0])
        labels_d.append(sg)

x = np.arange(len(labels_d))
width_d = 0.35
ax_d.bar(x - width_d / 2, xgb_vals, width_d, color=COLORS['primary'],
         alpha=0.85, label='XGBoost', edgecolor='white')
ax_d.bar(x + width_d / 2, lgb_vals, width_d, color=COLORS['accent1'],
         alpha=0.85, label='LightGBM', edgecolor='white')

ax_d.set_xticks(x)
ax_d.set_xticklabels(labels_d, fontsize=6.5, rotation=30, ha='right')
ax_d.set_ylabel('R² (gap)', fontsize=9)
ax_d.set_title('XGBoost vs. LightGBM Subgroup Stability', fontsize=10)
ax_d.legend(fontsize=7, framealpha=0.9)

# ─── Panel E: Comorbidity vs Treatment stratification ───
ax_e = fig.add_subplot(gs[2, 0])
panel_label(ax_e, 'E')

comor_sub = subgroup[(subgroup['Subgroup_Type'] == 'Comorbidity') & (subgroup['Task'] == 'gap')]
treat_sub = subgroup[(subgroup['Subgroup_Type'] == 'Treatment') & (subgroup['Task'] == 'gap')]

# R² range per model within comorbidity vs treatment strats
strat_data = []
for model_name in ['XGBoost', 'LightGBM']:
    c_vals = comor_sub[comor_sub['Model'] == model_name]['r2'].values
    t_vals = treat_sub[treat_sub['Model'] == model_name]['r2'].values
    if len(c_vals) > 1:
        strat_data.append({
            'Model': model_name, 'Type': 'Comorbidity',
            'R2_range': c_vals.max() - c_vals.min(),
            'R2_mean': c_vals.mean(), 'R2_std': c_vals.std(),
        })
    if len(t_vals) > 1:
        strat_data.append({
            'Model': model_name, 'Type': 'Treatment',
            'R2_range': t_vals.max() - t_vals.min(),
            'R2_mean': t_vals.mean(), 'R2_std': t_vals.std(),
        })

if strat_data:
    df_strat = pd.DataFrame(strat_data)
    types = df_strat['Type'].unique()
    models_e = df_strat['Model'].unique()
    x = np.arange(len(models_e))
    width_e = 0.35
    type_colors = {'Comorbidity': COLORS['primary'], 'Treatment': COLORS['accent2']}

    for i, t in enumerate(types):
        sub = df_strat[df_strat['Type'] == t]
        vals = [sub[sub['Model'] == m]['R2_range'].values[0]
                if m in sub['Model'].values else 0 for m in models_e]
        ax_e.bar(x + i * width_e, vals, width_e,
                 color=type_colors.get(t, PAL[i]), alpha=0.85, label=t, edgecolor='white')

    ax_e.set_xticks(x + width_e / 2)
    ax_e.set_xticklabels(models_e, fontsize=9)
    ax_e.set_ylabel('R² Range (max − min)', fontsize=9)
    ax_e.set_title('Stratification Impact on Performance Variability', fontsize=10)
    ax_e.legend(fontsize=7, framealpha=0.9)

# ─── Panel F: Cross-cohort degradation ───
ax_f = fig.add_subplot(gs[2, 1])
panel_label(ax_f, 'F')

# For each model, show R² drop from WHU → SCH → MIMIC
cohort_order = ['WHU', 'SCH', 'MIMIC-IV']
for task in tasks:
    mc_task = multicohort[(multicohort['Task'] == task) & (multicohort['Model'] == 'XGBoost')]
    r2_by_cohort = []
    for ds in cohort_order:
        row = mc_task[mc_task['Dataset'] == ds]
        r2_by_cohort.append(row['R2'].values[0] if len(row) > 0 else np.nan)
    ls = '-o' if task == 'gap' else '--s'
    ax_f.plot(range(len(cohort_order)), r2_by_cohort, ls,
              color=COLORS['primary'] if task == 'gap' else COLORS['secondary'],
              linewidth=2, markersize=7, label=f'XGBoost ({task})')

    mc_task2 = multicohort[(multicohort['Task'] == task) & (multicohort['Model'] == 'LightGBM')]
    r2_by_cohort2 = []
    for ds in cohort_order:
        row = mc_task2[mc_task2['Dataset'] == ds]
        r2_by_cohort2.append(row['R2'].values[0] if len(row) > 0 else np.nan)
    ls2 = '-^' if task == 'gap' else '--v'
    ax_f.plot(range(len(cohort_order)), r2_by_cohort2, ls2,
              color=COLORS['accent1'] if task == 'gap' else COLORS['accent2'],
              linewidth=1.5, markersize=6, alpha=0.8, label=f'LightGBM ({task})')

ax_f.set_xticks(range(len(cohort_order)))
ax_f.set_xticklabels(cohort_order, fontsize=9)
ax_f.set_ylabel('R²', fontsize=9)
ax_f.set_title('Cross-Cohort Performance Degradation', fontsize=10)
ax_f.legend(fontsize=6.5, ncol=2, framealpha=0.9)

# Annotate % drop
for task in tasks:
    mc_task = multicohort[(multicohort['Task'] == task) & (multicohort['Model'] == 'XGBoost')]
    whu_r2 = mc_task[mc_task['Dataset'] == 'WHU']['R2'].values
    mimic_r2 = mc_task[mc_task['Dataset'] == 'MIMIC-IV']['R2'].values
    if len(whu_r2) > 0 and len(mimic_r2) > 0:
        drop_pct = (1 - mimic_r2[0] / whu_r2[0]) * 100
        ax_f.annotate(f'−{drop_pct:.0f}%', xy=(2, mimic_r2[0]),
                      xytext=(2.15, mimic_r2[0] + 0.02),
                      fontsize=7, color='grey')

save_figure(fig, 'FigureS8_best_model_analysis')
print("  FigureS9_best_model_analysis saved.")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
