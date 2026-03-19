"""
Supplementary Figure S11: Model Performance Extended Analysis (4×2)
  A) Gap actual vs predicted scatter
  B) LOS actual vs predicted scatter
  C) Gap residual distribution
  D) LOS residual distribution
  E) 5-fold CV stability (R² + MAE)
  F) R² vs visit_order threshold
  G) Top 20 feature importance (Gap)
  H) Top 20 feature importance (LOS)

Run: python gen_supp_model_extended.py
"""
import sys, io, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure S11: Model Performance Extended Analysis")
print("=" * 60)

# ── Load data ──
print("[1] Loading data...")
features = pd.read_csv(os.path.join(HISTORY_DIR, 'history_features.csv'))
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), low_memory=False)
with open(os.path.join(V5_DIR, 'results.json'), 'r') as f:
    results = json.load(f)

features['target_gap_days'] = td['target_gap_days'].values
features['target_next_los'] = td['target_next_los'].values

# ── Add enhanced features (same as main pipeline) ──
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
    features['gap_range'] = features.get('gap_max_prev', pd.Series(dtype=float)).values - \
                            features.get('gap_min_prev', pd.Series(dtype=float)).values
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
feat_cols = [c for c in features.columns if c not in exclude and
             features[c].dtype in ['float64', 'int64', 'float32', 'int32']]

# ── Run predictions ──
def run_xgb(task, min_vo, cap, params):
    df = features[features['visit_order'] >= min_vo].copy()
    df[task] = df[task].clip(upper=cap)
    df = df[df[task].notna()]
    X = df[feat_cols].values.astype(np.float32)
    y = df[task].values.astype(np.float32)
    col_m = np.nanmean(X, axis=0)
    col_m = np.where(np.isnan(col_m), 0, col_m)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_m[j]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred, fold_metrics, fold_importances = [], [], [], []
    for tr, va in kf.split(X):
        m = xgb.XGBRegressor(**params, device='cuda', random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        p = m.predict(X[va])
        all_true.extend(y[va])
        all_pred.extend(p)
        fold_metrics.append({
            'r2': r2_score(y[va], p),
            'mae': mean_absolute_error(y[va], p)
        })
        fold_importances.append(dict(zip(feat_cols, m.feature_importances_)))
    return np.array(all_true), np.array(all_pred), fold_metrics, fold_importances

gap_params = results['gap']['params']
los_params = results['los']['params']

print("[2] Running Gap predictions (5-fold CV)...")
g_true, g_pred, g_folds, g_imps = run_xgb('target_gap_days', 20, 10, gap_params)
print(f"  Gap: R²={r2_score(g_true, g_pred):.4f}, MAE={mean_absolute_error(g_true, g_pred):.4f}")

print("[3] Running LOS predictions (5-fold CV)...")
l_true, l_pred, l_folds, l_imps = run_xgb('target_next_los', 5, 7, los_params)
print(f"  LOS: R²={r2_score(l_true, l_pred):.4f}, MAE={mean_absolute_error(l_true, l_pred):.4f}")

# ── R² vs visit_order threshold ──
print("[4] Computing R² vs visit_order thresholds...")
vo_thresholds = [3, 5, 8, 10, 12, 15, 18, 20, 25]
gap_r2_by_vo, los_r2_by_vo = [], []

for vo in vo_thresholds:
    for task, cap, params_t, out_list in [
        ('target_gap_days', 10, gap_params, gap_r2_by_vo),
        ('target_next_los', 7, los_params, los_r2_by_vo),
    ]:
        df_t = features[features['visit_order'] >= vo].copy()
        df_t[task] = df_t[task].clip(upper=cap)
        df_t = df_t[df_t[task].notna()]
        if len(df_t) < 30:
            out_list.append(np.nan)
            continue
        X = df_t[feat_cols].values.astype(np.float32)
        y = df_t[task].values.astype(np.float32)
        cm = np.nanmean(X, 0)
        cm = np.where(np.isnan(cm), 0, cm)
        for j in range(X.shape[1]):
            mk = np.isnan(X[:, j])
            X[mk, j] = cm[j]
        kf = KFold(n_splits=min(5, max(2, len(X) // 20)), shuffle=True, random_state=42)
        r2s = []
        for tr, va in kf.split(X):
            m = xgb.XGBRegressor(**params_t, device='cuda', random_state=42, n_jobs=-1)
            m.fit(X[tr], y[tr], verbose=False)
            r2s.append(r2_score(y[va], m.predict(X[va])))
        out_list.append(np.mean(r2s))

# ── Feature importances ──
print("[5] Computing feature importances...")
gap_imp_avg = {}
for fc in feat_cols:
    gap_imp_avg[fc] = np.mean([imp.get(fc, 0) for imp in g_imps])
los_imp_avg = {}
for fc in feat_cols:
    los_imp_avg[fc] = np.mean([imp.get(fc, 0) for imp in l_imps])

gap_top20 = sorted(gap_imp_avg.items(), key=lambda x: x[1], reverse=True)[:20]
los_top20 = sorted(los_imp_avg.items(), key=lambda x: x[1], reverse=True)[:20]

def get_name(col):
    return FEATURE_NAME_MAP.get(col, col)

# ═══════════════════════════════════════════════════════════════
# Generate Figure S11 — 4×2 grid
# ═══════════════════════════════════════════════════════════════
print("\n[6] Generating FigureS11_model_extended (4×2)...")
fig = plt.figure(figsize=(14, 18))
gs = gridspec.GridSpec(4, 2, hspace=0.38, wspace=0.32,
                       left=0.08, right=0.96, top=0.96, bottom=0.04)

# ─── Panel A: Gap scatter ───
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, 'A')
ax.scatter(g_true, g_pred, alpha=0.4, s=12, c=COLORS['primary'],
           edgecolors='white', linewidth=0.2)
lims = [min(g_true.min(), g_pred.min()) - 0.5, max(g_true.max(), g_pred.max()) + 0.5]
ax.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=1.5, alpha=0.8)
ax.set_xlabel('Actual Gap Days')
ax.set_ylabel('Predicted Gap Days')
ax.set_title('Gap Days: Actual vs Predicted')
r2_g = r2_score(g_true, g_pred)
mae_g = mean_absolute_error(g_true, g_pred)
ax.text(0.05, 0.95, f'$R^2$={r2_g:.3f}\nMAE={mae_g:.3f}\nn={len(g_true):,}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['grid']))

# ─── Panel B: LOS scatter ───
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'B')
ax.scatter(l_true, l_pred, alpha=0.35, s=10, c=COLORS['accent1'],
           edgecolors='white', linewidth=0.2)
lims = [min(l_true.min(), l_pred.min()) - 0.5, max(l_true.max(), l_pred.max()) + 0.5]
ax.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=1.5, alpha=0.8)
ax.set_xlabel('Actual LOS (days)')
ax.set_ylabel('Predicted LOS (days)')
ax.set_title('Length of Stay: Actual vs Predicted')
r2_l = r2_score(l_true, l_pred)
mae_l = mean_absolute_error(l_true, l_pred)
ax.text(0.05, 0.95, f'$R^2$={r2_l:.3f}\nMAE={mae_l:.3f}\nn={len(l_true):,}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['grid']))

# ─── Panel C: Gap residual distribution ───
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'C')
residuals_g = g_true - g_pred
ax.hist(residuals_g, bins=40, color=COLORS['primary'], alpha=0.7,
        edgecolor='white', linewidth=0.5, density=True)
ax.axvline(0, color=COLORS['secondary'], linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual (days)')
ax.set_ylabel('Density')
ax.set_title('Gap Days Residual Distribution')
ax.text(0.95, 0.95, f'Mean={residuals_g.mean():.3f}\nStd={residuals_g.std():.3f}\n'
        f'Skew={pd.Series(residuals_g).skew():.3f}',
        transform=ax.transAxes, va='top', ha='right', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['grid']))

# ─── Panel D: LOS residual distribution ───
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'D')
residuals_l = l_true - l_pred
ax.hist(residuals_l, bins=40, color=COLORS['accent1'], alpha=0.7,
        edgecolor='white', linewidth=0.5, density=True)
ax.axvline(0, color=COLORS['secondary'], linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual (days)')
ax.set_ylabel('Density')
ax.set_title('LOS Residual Distribution')
ax.text(0.95, 0.95, f'Mean={residuals_l.mean():.3f}\nStd={residuals_l.std():.3f}\n'
        f'Skew={pd.Series(residuals_l).skew():.3f}',
        transform=ax.transAxes, va='top', ha='right', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['grid']))

# ─── Panel E: 5-fold CV stability ───
ax = fig.add_subplot(gs[2, 0])
panel_label(ax, 'E')
x_f = np.arange(5)
w = 0.35
gap_r2s = [f['r2'] for f in g_folds]
los_r2s = [f['r2'] for f in l_folds]
ax.bar(x_f - w / 2, gap_r2s, w, color=COLORS['primary'], alpha=0.85, label='Gap $R^2$')
ax.bar(x_f + w / 2, los_r2s, w, color=COLORS['accent1'], alpha=0.85, label='LOS $R^2$')
ax.axhline(np.mean(gap_r2s), color=COLORS['primary'], linestyle='--', alpha=0.5)
ax.axhline(np.mean(los_r2s), color=COLORS['accent1'], linestyle='--', alpha=0.5)
ax.set_xlabel('Fold')
ax.set_ylabel('$R^2$ Score')
ax.set_title('Cross-Validation Stability')
ax.set_xticks(x_f)
ax.set_xticklabels([f'Fold {i + 1}' for i in range(5)])
ax.legend(frameon=True, facecolor='white', fontsize=8)
ax.set_ylim(0, 1.05)
# Add MAE on secondary axis
ax2 = ax.twinx()
gap_maes = [f['mae'] for f in g_folds]
los_maes = [f['mae'] for f in l_folds]
ax2.plot(x_f, gap_maes, 's--', color=COLORS['primary'], alpha=0.6, markersize=5, label='Gap MAE')
ax2.plot(x_f, los_maes, 'o--', color=COLORS['accent1'], alpha=0.6, markersize=5, label='LOS MAE')
ax2.set_ylabel('MAE (days)', fontsize=8)
ax2.legend(loc='center right', fontsize=7, framealpha=0.9)

# ─── Panel F: R² vs visit_order threshold ───
ax = fig.add_subplot(gs[2, 1])
panel_label(ax, 'F')
ax.plot(vo_thresholds, gap_r2_by_vo, '-o', color=COLORS['primary'],
        markersize=6, linewidth=2, label='Gap Days')
ax.plot(vo_thresholds, los_r2_by_vo, '-s', color=COLORS['accent1'],
        markersize=6, linewidth=2, label='LOS')
ax.axhline(y=0.9, color=COLORS['grid'], linestyle=':', alpha=0.5, label='$R^2$=0.9')
ax.set_xlabel('Minimum Visit Order Threshold')
ax.set_ylabel('$R^2$ Score')
ax.set_title('Predictability vs History Depth')
ax.legend(frameon=True, facecolor='white', fontsize=8)
ax.set_ylim(0, 1.05)

# ─── Panel G: Top 20 feature importance (Gap) ───
ax = fig.add_subplot(gs[3, 0])
panel_label(ax, 'G')
g_names = [x[0] for x in gap_top20]
g_vals = [x[1] for x in gap_top20]
y_pos = np.arange(len(g_names))
ax.barh(y_pos, g_vals, color=COLORS['primary'], alpha=0.85, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels([get_name(n) for n in g_names], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Top 20 Features — Gap Days')

# ─── Panel H: Top 20 feature importance (LOS) ───
ax = fig.add_subplot(gs[3, 1])
panel_label(ax, 'H')
l_names = [x[0] for x in los_top20]
l_vals = [x[1] for x in los_top20]
y_pos = np.arange(len(l_names))
ax.barh(y_pos, l_vals, color=COLORS['accent1'], alpha=0.85, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels([get_name(n) for n in l_names], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Top 20 Features — LOS')

save_figure(fig, 'FigureS11_model_extended')
print("  FigureS11_model_extended saved.")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
