"""
Figure 1: Epidemiology & Behavioral — 2×2 merged figure.
Panels: A) Monthly admission timeline (respiratory highlighted)
        B) Respiratory admission proportion by month across years
        C) Respiratory gap+LOS residual coupling scatter (cross-validated)
        D) Cross-year cosine similarity heatmap
Run: python gen_fig_new1.py
"""
import sys, io, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure 1: Epidemiology & Behavioral (merged)")
print("=" * 60)

# ═══════════════════════════════════════════════════
# Data loading — admissions (shared by Panel A, B, D)
# ═══════════════════════════════════════════════════
adm = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
adm['admit_dt'] = pd.to_datetime(adm['入院日期'], errors='coerce')
adm = adm.dropna(subset=['admit_dt'])

resp_kws = ['肺炎', '哮喘', '慢阻肺', '慢性阻塞性肺', '支气管', '肺部感染',
            '呼吸', '肺气肿', '肺结核', '上呼吸道', '肺癌']
diag_text = adm['EMR_初步诊断'].fillna('').astype(str).str.lower()
adm['is_respiratory'] = diag_text.apply(lambda t: any(k in t for k in resp_kws))

adm = adm[adm['admit_dt'].dt.year.isin([2016, 2017, 2018, 2019, 2020])]
print(f"  Admissions 2016-2020: {len(adm):,}")
print(f"  Respiratory: {adm['is_respiratory'].sum():,}")

# ═══════════════════════════════════════════════════
# Respiratory proportion matrix (Panel D)
# ═══════════════════════════════════════════════════
years = [2016, 2017, 2018, 2019, 2020]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
resp_matrix = np.full((len(years), 12), np.nan)
for yi, year in enumerate(years):
    yr_data = adm[adm['admit_dt'].dt.year == year]
    for m in range(12):
        m_data = yr_data[yr_data['admit_dt'].dt.month == (m + 1)]
        if len(m_data) >= 10:
            resp_matrix[yi, m] = m_data['is_respiratory'].sum() / len(m_data) * 100

# ═══════════════════════════════════════════════════
# Data loading — features + residuals (Panel C)
# ═══════════════════════════════════════════════════
features = pd.read_csv(os.path.join(HISTORY_DIR, 'history_features.csv'))
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), low_memory=False)
with open(os.path.join(V5_DIR, 'results.json')) as f:
    v5_results = json.load(f)

features['target_gap_days'] = td['target_gap_days'].values
features['target_next_los'] = td['target_next_los'].values

# Enhanced features (same as V5)
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
    features['gap_range'] = (features.get('gap_max_prev', pd.Series(dtype=float)).values
                             - features.get('gap_min_prev', pd.Series(dtype=float)).values)
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
feat_cols = [c for c in features.columns
             if c not in exclude and features[c].dtype in ['float64', 'int64', 'float32', 'int32']]

# Link features to respiratory status via train_data diagnosis
td_diag = td['EMR_初步诊断'].fillna('').astype(str).str.lower()
features['is_respiratory'] = td_diag.apply(lambda t: any(k in t for k in resp_kws)).values

# Admission dates for COVID period labeling
if '入院日期' in td.columns:
    features['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce').values
else:
    features['admit_dt'] = pd.NaT
features['is_covid_period'] = features['admit_dt'].dt.year == 2020

# ── Cross-validated gap residuals ──
gap_params = v5_results['gap']['params']
los_params = v5_results['los']['params']

df_gap = features[features['visit_order'] >= 5].copy()
df_gap['target_gap_days'] = df_gap['target_gap_days'].clip(upper=10)
df_gap = df_gap[df_gap['target_gap_days'].notna()].copy()

X_g = df_gap[feat_cols].values.astype(np.float32)
y_g = df_gap['target_gap_days'].values.astype(np.float32)
cm = np.nanmean(X_g, 0)
cm = np.where(np.isnan(cm), 0, cm)
for j in range(X_g.shape[1]):
    mk = np.isnan(X_g[:, j])
    X_g[mk, j] = cm[j]

df_gap['target_next_los_clipped'] = df_gap['target_next_los'].clip(upper=7)

print("  Computing cross-validated gap residuals...")
gap_residuals = np.full(len(X_g), np.nan)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in kf.split(X_g):
    m = xgb.XGBRegressor(**gap_params, device='cuda', random_state=42, n_jobs=-1)
    m.fit(X_g[tr], y_g[tr], verbose=False)
    gap_residuals[va] = y_g[va] - m.predict(X_g[va])

# ── Cross-validated LOS residuals (same rows) ──
print("  Computing cross-validated LOS residuals...")
y_l = df_gap['target_next_los_clipped'].values.astype(np.float32)
los_residuals = np.full(len(X_g), np.nan)
valid_los = ~np.isnan(y_l)
if valid_los.sum() > 100:
    X_l_sub = X_g[valid_los]
    y_l_sub = y_l[valid_los]
    los_res_sub = np.full(len(X_l_sub), np.nan)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in kf2.split(X_l_sub):
        m = xgb.XGBRegressor(**los_params, device='cuda', random_state=42, n_jobs=-1)
        m.fit(X_l_sub[tr], y_l_sub[tr], verbose=False)
        los_res_sub[va] = y_l_sub[va] - m.predict(X_l_sub[va])
    los_residuals[valid_los] = los_res_sub

df_gap['gap_residual'] = gap_residuals
df_gap['los_residual'] = los_residuals
df_residual = df_gap[df_gap['gap_residual'].notna() & df_gap['los_residual'].notna()].copy()
resp_mask = df_residual['is_respiratory'].values
covid_mask = df_residual['is_covid_period'].values
print(f"  Residual analysis: n={len(df_residual)}, respiratory={resp_mask.sum()}, "
      f"COVID period={covid_mask.sum()}")

# ═══════════════════════════════════════════════════
# Create Figure (2×2)
# ═══════════════════════════════════════════════════
print("  Generating Figure 1 (merged)...")
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ── Panel A: Monthly admission timeline ──
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, 'A')
monthly_total = adm.groupby(adm['admit_dt'].dt.to_period('M')).size()
monthly_resp = adm[adm['is_respiratory']].groupby(
    adm[adm['is_respiratory']]['admit_dt'].dt.to_period('M')).size()
monthly_resp_aligned = monthly_resp.reindex(monthly_total.index, fill_value=0)
monthly_non_resp = monthly_total - monthly_resp_aligned
months_ts = monthly_total.index.to_timestamp()

ax.bar(months_ts, monthly_non_resp.values, width=25, color=COLORS['primary'], alpha=0.6,
       label='Non-respiratory', edgecolor='white', linewidth=0.3)
ax.bar(months_ts, monthly_resp_aligned.values, width=25, color=COLORS['secondary'], alpha=0.85,
       label='Respiratory', edgecolor='white', linewidth=0.3, bottom=monthly_non_resp.values)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.set_xlabel('Year')
ax.set_ylabel('Monthly Admissions')
ax.set_title('Hospital Admissions Timeline')
ax.legend(frameon=True, fontsize=7)
covid_start = pd.Timestamp('2020-01-01')
ax.axvline(covid_start, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.text(covid_start, ax.get_ylim()[1] * 0.95, ' COVID-19', color='red', fontsize=7, va='top')

# ── Panel B: Respiratory proportion by month (each year as a line) ──
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'B')
year_colors = {2016: PAL[0], 2017: PAL[1], 2018: PAL[2], 2019: PAL[4], 2020: COLORS['secondary']}

for year in years:
    yr_data = adm[adm['admit_dt'].dt.year == year]
    if len(yr_data) < 50:
        continue
    monthly_pct = []
    for m in range(1, 13):
        m_data = yr_data[yr_data['admit_dt'].dt.month == m]
        if len(m_data) > 0:
            pct = m_data['is_respiratory'].sum() / len(m_data) * 100
        else:
            pct = np.nan
        monthly_pct.append(pct)
    lw = 2.5 if year == 2020 else 1.5
    marker = 'o' if year == 2020 else 's'
    ms = 5 if year == 2020 else 3
    ax.plot(range(12), monthly_pct, f'-{marker}', color=year_colors[year],
            linewidth=lw, markersize=ms, alpha=0.85, label=str(year))

ax.set_xticks(range(12))
ax.set_xticklabels(month_labels, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Respiratory Admission (%)')
ax.set_title('Monthly Respiratory Proportion by Year')
ax.legend(frameon=True, fontsize=7, ncol=2)
ax.axvspan(-0.5, 1.5, alpha=0.08, color=COLORS['secondary'])
ax.axvspan(10.5, 11.5, alpha=0.08, color=COLORS['secondary'])
ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Flu\nSeason', ha='center', fontsize=6,
        color=COLORS['secondary'])

# ── Panel C: Respiratory gap+LOS residual coupling scatter ──
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'C')

non_resp = df_residual[~df_residual['is_respiratory']]
resp_only = df_residual[df_residual['is_respiratory']]

ax.scatter(non_resp['gap_residual'], non_resp['los_residual'],
           c=COLORS['primary'], alpha=0.15, s=8,
           label=f'Non-respiratory (n={len(non_resp):,})',
           edgecolors='none', rasterized=True)
if len(resp_only) > 0:
    ax.scatter(resp_only['gap_residual'], resp_only['los_residual'],
               c=COLORS['secondary'], alpha=0.5, s=20,
               label=f'Respiratory (n={len(resp_only):,})',
               edgecolors='white', linewidth=0.3)

ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

if len(resp_only) > 10:
    rho, p_val = stats.spearmanr(resp_only['gap_residual'], resp_only['los_residual'])
    ax.text(0.03, 0.97, f'Respiratory:\n$\\rho$={rho:.3f}, p={p_val:.3f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Gap Residual (Actual - Predicted, days)')
ax.set_ylabel('LOS Residual (Actual - Predicted, days)')
ax.set_title('Disease-Specific Residual Coupling')
ax.legend(frameon=True, fontsize=6.5, loc='lower right')

# ── Panel D: Cross-year cosine similarity heatmap ──
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'D')
valid_cols = ~np.any(np.isnan(resp_matrix), axis=0)
if valid_cols.sum() >= 3:
    mat_valid = resp_matrix[:, valid_cols]
    sim = cosine_similarity(mat_valid)
    im = ax.imshow(sim, cmap=SPRING_CMAP, aspect='auto', vmin=0.9, vmax=1.0)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, fontsize=8)
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years, fontsize=8)
    for i in range(len(years)):
        for j in range(len(years)):
            ax.text(j, i, f'{sim[i, j]:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if sim[i, j] < 0.97 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Cross-Year Pattern Similarity')
else:
    ax.text(0.5, 0.5, 'Insufficient monthly data', ha='center', va='center',
            transform=ax.transAxes)

plt.tight_layout()
save_figure(fig, 'Figure1_epidemiology_behavioral')
print("  Done!")
