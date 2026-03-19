"""
Figure 5: Model Generalizability (2×2 merged)
  A) Gap scatter (actual vs predicted gap days)
  B) R² vs visit_order threshold (monotonic improvement)
  C) ML vs DL comparison across 3 cohorts (grouped bars)
  D) Comorbidity-stratified R² normalized ratio (WHU vs MIMIC)
Run: python gen_fig_new5.py
"""
import sys, io, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 50)
print("Figure 5: Model Generalizability")
print("=" * 50)

# ══════════════════════════════════════════════════════
# Data loading for Panels A & B
# ══════════════════════════════════════════════════════
features = pd.read_csv(os.path.join(HISTORY_DIR, 'history_features.csv'))
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), low_memory=False)
with open(os.path.join(V5_DIR, 'results.json'), 'r') as f:
    results = json.load(f)

features['target_gap_days'] = td['target_gap_days'].values
features['target_next_los'] = td['target_next_los'].values

# ── Enhanced features (same as V5) ──
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

# ── XGBoost cross-validation helper ──
def run_xgb(task, min_vo, cap, params):
    df = features[features['visit_order'] >= min_vo].copy()
    df[task] = df[task].clip(upper=cap)
    df = df[df[task].notna()]
    X = df[feat_cols].values.astype(np.float32)
    y = df[task].values.astype(np.float32)
    col_m = np.nanmean(X, axis=0); col_m = np.where(np.isnan(col_m), 0, col_m)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j]); X[mask, j] = col_m[j]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred, fold_metrics = [], [], []
    for tr, va in kf.split(X):
        m = xgb.XGBRegressor(**params, device='cuda', random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        p = m.predict(X[va])
        all_true.extend(y[va]); all_pred.extend(p)
        fold_metrics.append({'r2': r2_score(y[va], p), 'mae': mean_absolute_error(y[va], p)})
    return np.array(all_true), np.array(all_pred), fold_metrics

# ── Panel A data: Gap predictions ──
gap_params = results['gap']['params']
los_params = results['los']['params']

print("  Running Gap predictions...")
g_true, g_pred, g_folds = run_xgb('target_gap_days', 20, 10, gap_params)

# ── Panel B data: R² vs visit_order sweep ──
print("  Computing R² vs visit_order thresholds...")
vo_thresholds = [3, 5, 8, 10, 12, 15, 18, 20, 25]
gap_r2_by_vo, los_r2_by_vo = [], []

for vo in vo_thresholds:
    df_g = features[features['visit_order'] >= vo].copy()
    df_g['target_gap_days'] = df_g['target_gap_days'].clip(upper=10)
    df_g = df_g[df_g['target_gap_days'].notna()]
    if len(df_g) < 30:
        gap_r2_by_vo.append(np.nan); continue
    Xg = df_g[feat_cols].values.astype(np.float32)
    yg = df_g['target_gap_days'].values.astype(np.float32)
    cm = np.nanmean(Xg, 0); cm = np.where(np.isnan(cm), 0, cm)
    for j in range(Xg.shape[1]):
        mk = np.isnan(Xg[:, j]); Xg[mk, j] = cm[j]
    kf = KFold(n_splits=min(5, max(2, len(Xg) // 20)), shuffle=True, random_state=42)
    r2s = []
    for tr, va in kf.split(Xg):
        m = xgb.XGBRegressor(**gap_params, device='cuda', random_state=42, n_jobs=-1)
        m.fit(Xg[tr], yg[tr], verbose=False)
        r2s.append(r2_score(yg[va], m.predict(Xg[va])))
    gap_r2_by_vo.append(np.mean(r2s))

for vo in vo_thresholds:
    df_l = features[features['visit_order'] >= vo].copy()
    df_l['target_next_los'] = df_l['target_next_los'].clip(upper=7)
    df_l = df_l[df_l['target_next_los'].notna()]
    if len(df_l) < 30:
        los_r2_by_vo.append(np.nan); continue
    Xl = df_l[feat_cols].values.astype(np.float32)
    yl = df_l['target_next_los'].values.astype(np.float32)
    cm = np.nanmean(Xl, 0); cm = np.where(np.isnan(cm), 0, cm)
    for j in range(Xl.shape[1]):
        mk = np.isnan(Xl[:, j]); Xl[mk, j] = cm[j]
    kf = KFold(n_splits=min(5, max(2, len(Xl) // 20)), shuffle=True, random_state=42)
    r2s = []
    for tr, va in kf.split(Xl):
        m = xgb.XGBRegressor(**los_params, device='cuda', random_state=42, n_jobs=-1)
        m.fit(Xl[tr], yl[tr], verbose=False)
        r2s.append(r2_score(yl[va], m.predict(Xl[va])))
    los_r2_by_vo.append(np.mean(r2s))

# ══════════════════════════════════════════════════════
# Data loading for Panels C & D
# ══════════════════════════════════════════════════════
xd_path = os.path.join(OUTPUT_DIR, 'cross_dataset_disease_results.json')
with open(xd_path, 'r') as f:
    xd = json.load(f)

common_cats = xd['cross_dataset']['common_categories']
r2_table = xd['cross_dataset']['r2_table']
spearman = xd['cross_dataset'].get('spearman', {})
whu_overall_r2 = xd['WHU']['overall']['r2']
mimic_overall_r2 = xd['MIMIC']['overall']['r2']

# Panel C: multicohort ML vs DL
mc_csv = os.path.join(OUTPUT_DIR, 'model_comparison', 'multicohort_comparison.csv')
mc_df = pd.read_csv(mc_csv) if os.path.exists(mc_csv) else None

# ══════════════════════════════════════════════════════
# Create Figure 5 (2×2)
# ══════════════════════════════════════════════════════
print("  Generating Figure 5...")
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

# ── Panel A: Gap scatter (actual vs predicted) ──
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, 'A')
ax.scatter(g_true, g_pred, alpha=0.5, s=20, c=COLORS['primary'],
           edgecolors='white', linewidth=0.3)
lims = [min(g_true.min(), g_pred.min()) - 0.5,
        max(g_true.max(), g_pred.max()) + 0.5]
ax.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=1.5, alpha=0.8)
ax.set_xlabel('Actual Gap Days')
ax.set_ylabel('Predicted Gap Days')
ax.set_title('Gap Days Prediction')
r2_g = r2_score(g_true, g_pred)
mae_g = mean_absolute_error(g_true, g_pred)
ax.text(0.05, 0.95, f'$R^2$={r2_g:.3f}\nMAE={mae_g:.3f}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['grid']))

# ── Panel B: R² vs visit_order threshold ──
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'B')
ax.plot(vo_thresholds, gap_r2_by_vo, '-o', color=COLORS['primary'],
        markersize=5, linewidth=2, label='Gap Days')
ax.plot(vo_thresholds, los_r2_by_vo, '-s', color=COLORS['accent1'],
        markersize=5, linewidth=2, label='LOS')
ax.axhline(y=0.9, color=COLORS['grid'], linestyle=':', alpha=0.5)
ax.set_xlabel('Minimum Visit Order Threshold')
ax.set_ylabel('$R^2$ Score')
ax.set_title('Predictability vs History Depth')
ax.legend(frameon=True, facecolor='white')

# ── Panel C: ML vs DL comparison (3 cohorts) ──
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'C')

if mc_df is not None:
    ds_names = ['WHU', 'SCH', 'MIMIC-IV']
    gap_data = mc_df[mc_df['Task'].str.lower().str.strip() == 'gap']

    ml_r2s, dl_r2s, ml_errs, dl_errs = [], [], [], []
    for ds in ds_names:
        ds_gap = gap_data[gap_data['Dataset'] == ds]
        if len(ds_gap) == 0:
            ml_r2s.append(0); dl_r2s.append(0)
            ml_errs.append(0); dl_errs.append(0)
            continue
        ml_rows = ds_gap[ds_gap['Category'] == 'ML'].sort_values('R2', ascending=False)
        dl_rows = ds_gap[ds_gap['Category'] == 'Deep Learning'].sort_values('R2', ascending=False)
        if len(ml_rows) > 0:
            ml_r2s.append(ml_rows.iloc[0]['R2'])
            ml_errs.append(ml_rows.iloc[0].get('R2_std', 0))
        else:
            ml_r2s.append(0); ml_errs.append(0)
        if len(dl_rows) > 0:
            dl_r2s.append(dl_rows.iloc[0]['R2'])
            dl_errs.append(dl_rows.iloc[0].get('R2_std', 0))
        else:
            dl_r2s.append(0); dl_errs.append(0)

    x = np.arange(len(ds_names))
    w = 0.3
    ax.bar(x - w / 2, ml_r2s, w, yerr=ml_errs, color=COLORS['primary'], alpha=0.85,
           label='Best ML', edgecolor='white', capsize=4)
    ax.bar(x + w / 2, dl_r2s, w, yerr=dl_errs, color=COLORS['accent3'], alpha=0.85,
           label='Best DL', edgecolor='white', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=10)
    ax.set_ylabel('Gap $R^2$ Score')
    ax.set_title('ML vs Deep Learning (3 Cohorts)')
    ax.legend(frameon=True, fontsize=7, loc='upper right')
    for i in range(len(ds_names)):
        if ml_r2s[i] > 0:
            ax.text(x[i] - w / 2, ml_r2s[i] + ml_errs[i] + 0.01, f'{ml_r2s[i]:.3f}',
                    ha='center', fontsize=7, fontweight='bold')
        if dl_r2s[i] > 0:
            ax.text(x[i] + w / 2, dl_r2s[i] + dl_errs[i] + 0.01, f'{dl_r2s[i]:.3f}',
                    ha='center', fontsize=7)
    ax.set_ylim(0, max(max(ml_r2s), max(dl_r2s)) * 1.2 if max(ml_r2s) > 0 else 1.0)
else:
    ax.text(0.5, 0.5, 'multicohort_comparison.csv\nnot found',
            transform=ax.transAxes, ha='center', va='center', fontsize=10)
    ax.set_title('ML vs Deep Learning (3 Cohorts)')

# ── Panel D: Comorbidity R² normalized ratio (WHU vs MIMIC) ──
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'D')

sorted_cats = sorted(common_cats, key=lambda c: r2_table[c]['WHU'], reverse=True)
n_cats = len(sorted_cats)
x = np.arange(n_cats)
w = 0.35

whu_ratios = [r2_table[cat]['WHU'] / whu_overall_r2 for cat in sorted_cats]
mimic_ratios = [r2_table[cat]['MIMIC'] / mimic_overall_r2 for cat in sorted_cats]

ax.bar(x - w / 2, whu_ratios, w, color=COLORS['primary'], alpha=0.85,
       label=f'WHU (overall $R^2$={whu_overall_r2:.3f})', edgecolor='white')
ax.bar(x + w / 2, mimic_ratios, w, color=COLORS['accent1'], alpha=0.85,
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

plt.tight_layout()
save_figure(fig, 'Figure5_model_generalizability')
print("  Done!")
