"""
Module B: Figure 2 — Cross-Dataset Prediction, Disease Stratification & Feature Importance
Merged from original Figure 2 + Figure 3. 6 panels (3x2):
  A) ML vs DL head-to-head comparison (3 cohorts)
  B) WHU + MIMIC comorbidity-stratified R² (normalized ratio)
  C) SCH treatment-stratified R²
  D) WHU↔MIMIC rank comparison (bump chart)
  E) Top 20 feature importance (log10 x-axis)
  F) Feature importance heatmap by comorbidity
Requires: cross_dataset_disease_results.json, history_features.csv, results.json
Run: python gen_fig2.py
"""
import sys, io, json, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 50)
print("Figure 2: Cross-Dataset Prediction & Feature Importance (merged)")
print("=" * 50)

# ── Load cross-dataset results ──
xd_path = os.path.join(OUTPUT_DIR, 'cross_dataset_disease_results.json')
with open(xd_path, 'r') as f:
    xd = json.load(f)

# ── Model comparison data (for Panel A) ──
comp_csv = os.path.join(OUTPUT_DIR, 'model_comparison', 'model_comparison_table.csv')
comp_df = pd.read_csv(comp_csv) if os.path.exists(comp_csv) else None

# ── Extract data ──
common_cats = xd['cross_dataset']['common_categories']
r2_table = xd['cross_dataset']['r2_table']
sch_tx = xd['SCH'].get('treatment', {})
spearman = xd['cross_dataset'].get('spearman', {})

whu_overall_r2 = xd['WHU']['overall']['r2']
mimic_overall_r2 = xd['MIMIC']['overall']['r2']

print("  Comorbidity R² (WHU vs MIMIC):")
for cat in common_cats:
    print(f"    {cat}: WHU={r2_table[cat]['WHU']:.4f}, MIMIC={r2_table[cat]['MIMIC']:.4f}")
print(f"  SCH treatment types: {len(sch_tx)}")

# ═══════════════════════════════════════════════════
# Load data for Panels E-F (feature importance)
# ═══════════════════════════════════════════════════
print("  Loading feature data...")
features = pd.read_csv(os.path.join(HISTORY_DIR, 'history_features.csv'))
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), low_memory=False)
with open(os.path.join(V5_DIR, 'results.json'), 'r') as f:
    results = json.load(f)

features['target_gap_days'] = td['target_gap_days'].values
features['target_next_los'] = td['target_next_los'].values
if '病案号' in td.columns:
    features['patient_id'] = td['病案号'].values

# Add enhanced features
ig = features.get('incoming_gap', pd.Series(dtype=float)).values
gm = features.get('gap_mean_prev', pd.Series(dtype=float)).values
gcv = features.get('gap_cv_prev', pd.Series(dtype=float)).values
g2 = features.get('prev_gap_2', pd.Series(dtype=float)).values
g3 = features.get('prev_gap_3', pd.Series(dtype=float)).values
gema = features.get('gap_ema', pd.Series(dtype=float)).values
ld = features.get('los_days', pd.Series(dtype=float)).values
lm = features.get('los_mean_prev', pd.Series(dtype=float)).values
lema = features.get('los_ema', pd.Series(dtype=float)).values
pl1 = features.get('prev_los_1', pd.Series(dtype=float)).values
pl2 = features.get('prev_los_2', pd.Series(dtype=float)).values
freq = features.get('admission_frequency', pd.Series(dtype=float)).values

with np.errstate(divide='ignore', invalid='ignore'):
    features['gap_regularity'] = 1.0 / (1.0 + np.where(np.isnan(gcv), 10, gcv))
    features['gap_deviation'] = (ig - gm) / np.where(gm == 0, np.nan, gm)
    features['gap_shortening'] = (ig < gm).astype(float)
    features['gap_last_diff'] = ig - g2
    features['gap_last_ratio'] = ig / np.where(g2 == 0, np.nan, g2)
    features['gap_accel'] = (ig - g2) - (g2 - g3)
    features['gap_ema_ratio'] = ig / np.where(gema == 0, np.nan, gema)
    features['log_incoming_gap'] = np.log1p(np.maximum(ig, 0))
    features['log_gap_mean'] = np.log1p(np.maximum(gm, 0))
    features['gap_range'] = features.get('gap_max_prev', pd.Series(dtype=float)).values - features.get('gap_min_prev', pd.Series(dtype=float)).values
    features['los_deviation'] = (ld - lm) / np.where(lm == 0, np.nan, lm)
    features['los_ema_ratio'] = ld / np.where(lema == 0, np.nan, lema)
    features['log_los_days'] = np.log1p(np.maximum(ld, 0))
    features['los_days_sq'] = ld ** 2
    features['los_days_sqrt'] = np.sqrt(np.maximum(ld, 0))
    features['los_wavg_2'] = 0.7 * ld + 0.3 * pl1
    features['los_wavg_3'] = 0.5 * ld + 0.3 * pl1 + 0.2 * pl2
    features['gap_x_los'] = ig * ld
    features['gap_x_freq'] = ig * freq
    features['los_x_freq'] = ld * freq

for col in features.columns:
    if features[col].dtype in ['float64', 'float32']:
        features[col] = np.where(np.isinf(features[col].values), np.nan, features[col].values)

exclude = {'visit_id', 'patient_id', 'target_gap_days', 'target_next_los'}
feat_cols = [c for c in features.columns if c not in exclude and features[c].dtype in ['float64', 'int64', 'float32', 'int32']]

gap_params = results['gap']['params']
los_params = results['los']['params']

# Train Gap model for feature importance
print("  Training Gap model for feature importance...")
df_g = features[features['visit_order'] >= 20].copy()
df_g['target_gap_days'] = df_g['target_gap_days'].clip(upper=10)
df_g = df_g[df_g['target_gap_days'].notna()]
X_g = df_g[feat_cols].values.astype(np.float32)
y_g = df_g['target_gap_days'].values.astype(np.float32)
cm_g = np.nanmean(X_g, 0); cm_g = np.where(np.isnan(cm_g), 0, cm_g)
for j in range(X_g.shape[1]):
    mk = np.isnan(X_g[:, j]); X_g[mk, j] = cm_g[j]
m_gap = xgb.XGBRegressor(**gap_params, device='cuda', random_state=42, n_jobs=-1)
m_gap.fit(X_g, y_g, verbose=False)
gap_imp = dict(zip(feat_cols, m_gap.feature_importances_))

# Train LOS model for feature importance
print("  Training LOS model for feature importance...")
df_l = features[features['visit_order'] >= 5].copy()
df_l['target_next_los'] = df_l['target_next_los'].clip(upper=7)
df_l = df_l[df_l['target_next_los'].notna()]
X_l = df_l[feat_cols].values.astype(np.float32)
y_l = df_l['target_next_los'].values.astype(np.float32)
cm_l = np.nanmean(X_l, 0); cm_l = np.where(np.isnan(cm_l), 0, cm_l)
for j in range(X_l.shape[1]):
    mk = np.isnan(X_l[:, j]); X_l[mk, j] = cm_l[j]
m_los = xgb.XGBRegressor(**los_params, device='cuda', random_state=42, n_jobs=-1)
m_los.fit(X_l, y_l, verbose=False)
los_imp = dict(zip(feat_cols, m_los.feature_importances_))

top20_gap = sorted(gap_imp.items(), key=lambda x: x[1], reverse=True)[:20]
top20_names = [x[0] for x in top20_gap]

# ═══════════════════════════════════════════════════
# Create Figure 2 (3x2)
# ═══════════════════════════════════════════════════
print("  Generating Figure 2 (merged 6-panel)...")
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.32)

# ── Panel A: ML vs DL comparison (multi-cohort) ──
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, 'A')

# Try to load multi-cohort results first
mc_csv = os.path.join(OUTPUT_DIR, 'model_comparison', 'multicohort_comparison.csv')
if os.path.exists(mc_csv):
    mc_df = pd.read_csv(mc_csv)
    # Show gap R2 for ML best vs DL best across all 3 datasets
    ds_names = ['WHU', 'SCH', 'MIMIC-IV']
    gap_data = mc_df[mc_df['Task'].str.lower().str.strip() == 'gap']

    ml_r2s, dl_r2s, ml_errs, dl_errs = [], [], [], []
    ml_labels, dl_labels = [], []
    for ds in ds_names:
        ds_gap = gap_data[gap_data['Dataset'] == ds]
        if len(ds_gap) == 0:
            ml_r2s.append(0); dl_r2s.append(0)
            ml_errs.append(0); dl_errs.append(0)
            ml_labels.append('ML'); dl_labels.append('DL')
            continue
        ml_rows = ds_gap[ds_gap['Category'] == 'ML'].sort_values('R2', ascending=False)
        dl_rows = ds_gap[ds_gap['Category'] == 'Deep Learning'].sort_values('R2', ascending=False)
        if len(ml_rows) > 0:
            ml_r2s.append(ml_rows.iloc[0]['R2'])
            ml_errs.append(ml_rows.iloc[0].get('R2_std', 0))
            ml_labels.append(ml_rows.iloc[0]['Model'])
        else:
            ml_r2s.append(0); ml_errs.append(0); ml_labels.append('ML')
        if len(dl_rows) > 0:
            dl_r2s.append(dl_rows.iloc[0]['R2'])
            dl_errs.append(dl_rows.iloc[0].get('R2_std', 0))
            dl_labels.append(dl_rows.iloc[0]['Model'])
        else:
            dl_r2s.append(0); dl_errs.append(0); dl_labels.append('DL')

    x = np.arange(len(ds_names))
    w = 0.3
    b1 = ax.bar(x - w/2, ml_r2s, w, yerr=ml_errs, color=COLORS['primary'], alpha=0.85,
                label=f'Best ML', edgecolor='white', capsize=4)
    b2 = ax.bar(x + w/2, dl_r2s, w, yerr=dl_errs, color=COLORS['accent3'], alpha=0.85,
                label=f'Best DL', edgecolor='white', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=10)
    ax.set_ylabel('Gap $R^2$ Score')
    ax.set_title('ML vs Deep Learning (3 Cohorts)')
    ax.legend(frameon=True, fontsize=7, loc='upper right')
    for i in range(len(ds_names)):
        if ml_r2s[i] > 0:
            ax.text(x[i]-w/2, ml_r2s[i]+ml_errs[i]+0.01, f'{ml_r2s[i]:.3f}',
                    ha='center', fontsize=7, fontweight='bold')
        if dl_r2s[i] > 0:
            ax.text(x[i]+w/2, dl_r2s[i]+dl_errs[i]+0.01, f'{dl_r2s[i]:.3f}',
                    ha='center', fontsize=7)
    ax.set_ylim(0, max(max(ml_r2s), max(dl_r2s)) * 1.2 if max(ml_r2s) > 0 else 1.0)

elif comp_df is not None:
    # Fallback: WHU-only comparison
    ml_names = ['XGBoost (GPU)', 'LightGBM', 'Random Forest', 'ElasticNet']
    gap_comp = comp_df[comp_df['Task'].str.lower().str.strip() == 'gap']
    los_comp = comp_df[comp_df['Task'].str.lower().str.strip() == 'los']
    gap_ml = gap_comp[gap_comp['Model'].isin(ml_names)].sort_values('R2', ascending=False).iloc[0]
    gap_dl = gap_comp[~gap_comp['Model'].isin(ml_names)].sort_values('R2', ascending=False).iloc[0]
    los_ml = los_comp[los_comp['Model'].isin(ml_names)].sort_values('R2', ascending=False).iloc[0]
    los_dl = los_comp[~los_comp['Model'].isin(ml_names)].sort_values('R2', ascending=False).iloc[0]

    x = np.arange(2)
    w = 0.3
    ml_vals = [gap_ml['R2'], los_ml['R2']]
    dl_vals = [gap_dl['R2'], los_dl['R2']]
    ml_errs = [gap_ml.get('R2_std', 0), los_ml.get('R2_std', 0)]
    dl_errs = [gap_dl.get('R2_std', 0), los_dl.get('R2_std', 0)]

    b1 = ax.bar(x - w/2, ml_vals, w, yerr=ml_errs, color=COLORS['primary'], alpha=0.85,
                label=f"ML: {gap_ml['Model']}", edgecolor='white', capsize=4)
    b2 = ax.bar(x + w/2, dl_vals, w, yerr=dl_errs, color=COLORS['accent3'], alpha=0.85,
                label=f"DL: {gap_dl['Model']}", edgecolor='white', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(['Gap Days', 'LOS'], fontsize=10)
    ax.set_ylabel('$R^2$ Score')
    ax.set_title('ML vs Deep Learning')
    ax.legend(frameon=True, fontsize=7, loc='lower right')
    for i in range(2):
        ax.text(x[i]-w/2, ml_vals[i]+ml_errs[i]+0.01, f'{ml_vals[i]:.3f}',
                ha='center', fontsize=8, fontweight='bold')
        ax.text(x[i]+w/2, dl_vals[i]+dl_errs[i]+0.01, f'{dl_vals[i]:.3f}',
                ha='center', fontsize=8)
    ax.set_ylim(0, 1.1)

# ── Panel B: WHU + MIMIC comorbidity R² — normalized ratio (R²_group / R²_overall) ──
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'B')

# Sort categories by WHU R² descending
sorted_cats = sorted(common_cats, key=lambda c: r2_table[c]['WHU'], reverse=True)

n_cats = len(sorted_cats)
x = np.arange(n_cats)
w = 0.35

whu_ratios = [r2_table[cat]['WHU'] / whu_overall_r2 for cat in sorted_cats]
mimic_ratios = [r2_table[cat]['MIMIC'] / mimic_overall_r2 for cat in sorted_cats]

ax.bar(x - w/2, whu_ratios, w, color=COLORS['primary'], alpha=0.85,
       label=f'WHU (overall $R^2$={whu_overall_r2:.3f})', edgecolor='white')
ax.bar(x + w/2, mimic_ratios, w, color=COLORS['accent1'], alpha=0.85,
       label=f'MIMIC (overall $R^2$={mimic_overall_r2:.3f})', edgecolor='white')
ax.axhline(1.0, color='grey', linestyle='--', linewidth=1, alpha=0.5, label='Overall baseline')
ax.set_xticks(x)
ax.set_xticklabels(sorted_cats, rotation=30, ha='right', fontsize=7.5)
ax.set_ylabel('Normalized $R^2$ (group / overall)')
ax.set_title('Comorbidity Predictability: WHU vs MIMIC-IV')
ax.legend(frameon=True, fontsize=6.5, loc='lower left')
ax.set_ylim(0, max(max(whu_ratios), max(mimic_ratios)) * 1.15)

# Annotate Spearman
sp_info = spearman.get('WHU_vs_MIMIC', {})
if sp_info:
    rho_val = sp_info['rho']
    p_val = sp_info['p_value']
    ax.text(0.97, 0.95, f'Spearman $\\rho$={rho_val:.3f}\n$p$={p_val:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ── Panel C: SCH treatment-stratified R² ──
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'C')

sch_overall_r2 = xd['SCH']['overall']['r2']
tx_sorted = sorted(sch_tx.items(), key=lambda x: x[1]['r2'])  # ascending for horizontal bars
tx_names = [k for k, v in tx_sorted]
tx_r2 = [v['r2'] for k, v in tx_sorted]
tx_r2_std = [v.get('r2_std', 0) for k, v in tx_sorted]
tx_n = [v['n'] for k, v in tx_sorted]

y_pos = np.arange(len(tx_names))
bar_colors = [COLORS['accent2'] if r > sch_overall_r2 else COLORS['secondary'] for r in tx_r2]

bars = ax.barh(y_pos, tx_r2, xerr=tx_r2_std, height=0.6, color=bar_colors,
               alpha=0.85, edgecolor='white', capsize=3)
ax.axvline(sch_overall_r2, color='grey', linestyle='--', linewidth=1, alpha=0.6)
ax.text(sch_overall_r2 + 0.005, len(tx_names) - 0.5, f'Overall\n$R^2$={sch_overall_r2:.3f}',
        fontsize=7, va='top', color='grey')

ax.set_yticks(y_pos)
ax.set_yticklabels(tx_names, fontsize=8)
ax.set_xlabel('Gap $R^2$ Score')
ax.set_title('SCH Treatment-Stratified Gap $R^2$')

# Add n labels
for i, (r2, n) in enumerate(zip(tx_r2, tx_n)):
    ax.text(r2 + tx_r2_std[i] + 0.01, i, f'n={n:,}', va='center', fontsize=7)

# ── Panel D: WHU↔MIMIC rank concordance (bump chart) ──
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'D')

ds_pair = ['WHU', 'MIMIC-IV']
rankings = {}
for di, ds_key in enumerate(['WHU', 'MIMIC']):
    vals = [(cat, r2_table[cat][ds_key]) for cat in common_cats]
    vals.sort(key=lambda t: t[1], reverse=True)
    rankings[ds_pair[di]] = {cat: rank + 1 for rank, (cat, _) in enumerate(vals)}

cat_colors = {cat: PAL[i % len(PAL)] for i, cat in enumerate(common_cats)}
ds_x = np.array([0, 1])
for cat in common_cats:
    ranks = [rankings[ds][cat] for ds in ds_pair]
    ax.plot(ds_x, ranks, '-o', color=cat_colors[cat], linewidth=2.5, markersize=10,
            alpha=0.85, label=cat, zorder=3)
    # Labels at left and right
    ax.text(-0.12, ranks[0], f'{cat}', va='center', ha='right', fontsize=7.5,
            color=cat_colors[cat], fontweight='bold')
    ax.text(1.12, ranks[1], f'{cat}', va='center', ha='left', fontsize=7.5,
            color=cat_colors[cat], fontweight='bold')

ax.set_xticks(ds_x)
ax.set_xticklabels(ds_pair, fontsize=11, fontweight='bold')
ax.set_ylabel('Rank (1 = highest $R^2$)')
ax.set_title('Comorbidity Predictability Rank')
ax.set_ylim(n_cats + 0.8, 0.2)  # invert so rank 1 is on top
ax.set_xlim(-0.55, 1.55)
ax.grid(axis='y', alpha=0.2)

# ── Panel E: Top 20 features (log10 x-axis) ──
ax = fig.add_subplot(gs[2, 0])
panel_label(ax, 'E')

y_pos_f = np.arange(len(top20_names))
gap_vals = [gap_imp[n] for n in top20_names]
los_vals = [los_imp.get(n, 0) for n in top20_names]
w_f = 0.35
ax.barh(y_pos_f + w_f/2, gap_vals, w_f, color=COLORS['primary'], alpha=0.85, label='Gap Days')
ax.barh(y_pos_f - w_f/2, los_vals, w_f, color=COLORS['accent1'], alpha=0.85, label='LOS')
ax.set_xscale('log')
ax.set_yticks(y_pos_f)
ax.set_yticklabels([get_readable_name(n) for n in top20_names], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Gain, log scale)')
ax.set_title('Top 20 Predictive Features')
ax.legend(frameon=True, loc='lower right', fontsize=7)

# ── Panel F: Feature importance heatmap by comorbidity ──
ax = fig.add_subplot(gs[2, 1])
panel_label(ax, 'F')

all_adm = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
comorbidity_kw = {
    'HTN': ['hypertension', 'hypertensive', '\u9ad8\u8840\u538b'],
    'DM': ['diabetes', 'diabetic', '\u7cd6\u5c3f\u75c5'],
    'CVD': ['coronary', 'cardiac', 'heart failure', '\u51a0\u5fc3', '\u5fc3\u529b\u8870\u7aed'],
    'Resp': ['copd', 'asthma', 'pneumonia', '\u80ba\u708e', '\u54ee\u5598'],
    'Liver': ['hepatitis', 'cirrhosis', 'liver', '\u809d\u708e', '\u809d\u786c\u5316'],
    'Renal': ['kidney', 'renal', 'nephro', '\u80be'],
}

text_cols = [c for c in ['EMR_\u521d\u6b65\u8bca\u65ad', 'EMR_\u65e2\u5f80\u53f2', 'EMR_\u51fa\u9662\u8bb0\u5f55', 'EMR_\u75c5\u4f8b\u6458\u8981', '\u8bca\u65ad\u6587\u672c'] if c in all_adm.columns]
all_adm['_diag_combined'] = all_adm[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()

pid_col = '\u75c5\u6848\u53f7' if '\u75c5\u6848\u53f7' in all_adm.columns else 'patient_id'
patient_comorbidity = {}
for pid, txt in zip(all_adm[pid_col], all_adm['_diag_combined']):
    if pd.isna(pid):
        continue
    if pid not in patient_comorbidity:
        patient_comorbidity[pid] = set()
    for cat, kws in comorbidity_kw.items():
        if any(k in txt for k in kws):
            patient_comorbidity[pid].add(cat)

top10 = top20_names[:10]
heat_data = np.zeros((len(top10), len(comorbidity_kw)))

for ci, (cat, _) in enumerate(comorbidity_kw.items()):
    if 'patient_id' in features.columns:
        mask = features['patient_id'].map(lambda p, c=cat: c in patient_comorbidity.get(p, set()))
    else:
        mask = pd.Series([False] * len(features))
    subset = features[mask & (features['visit_order'] >= 20)].copy()
    subset['target_gap_days'] = subset['target_gap_days'].clip(upper=10)
    subset = subset[subset['target_gap_days'].notna()]
    if len(subset) < 20:
        heat_data[:, ci] = 0
        continue
    X = subset[feat_cols].values.astype(np.float32)
    y = subset['target_gap_days'].values.astype(np.float32)
    cm_val = np.nanmean(X, 0); cm_val = np.where(np.isnan(cm_val), 0, cm_val)
    for j in range(X.shape[1]):
        mk = np.isnan(X[:, j]); X[mk, j] = cm_val[j]
    m = xgb.XGBRegressor(**gap_params, device='cuda', random_state=42, n_jobs=-1)
    m.fit(X, y, verbose=False)
    imp_dict = dict(zip(feat_cols, m.feature_importances_))
    for fi, fn in enumerate(top10):
        heat_data[fi, ci] = imp_dict.get(fn, 0)

row_max = heat_data.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
heat_norm = heat_data / row_max

im = ax.imshow(heat_norm, aspect='auto', cmap=SPRING_CMAP, vmin=0, vmax=1)
ax.set_xticks(np.arange(len(comorbidity_kw)))
ax.set_xticklabels(list(comorbidity_kw.keys()), fontsize=8, rotation=30, ha='right')
ax.set_yticks(np.arange(len(top10)))
ax.set_yticklabels([get_readable_name(n) for n in top10], fontsize=7.5)
ax.set_title('Feature Importance by Comorbidity')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Normalized Importance', fontsize=8)

plt.tight_layout()
save_figure(fig, 'Figure7_disease_stratified')
print("  Done!")
