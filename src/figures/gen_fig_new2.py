"""
Figure 2: Sentinel Discovery — Behavioral Early Warning
2×2 merged figure from gen_fig6.py (behavioral early warning).

Panels:
  A: 2019 monthly behavioral similarity heatmap (comorbidity × month)
  B: Multi-year quarterly similarity time series
  C: Bootstrap confidence intervals (originally Panel D)
  D: Lead-time analysis (originally Panel E)

Panel C from gen_fig6 (seasonal confounding Q4 permutation test) → supplementary S8.
"""

import sys, io, warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure 2: Sentinel Discovery — Behavioral Early Warning")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("[1] Loading data...")
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
covid = pd.read_csv(os.path.join(OUTPUT_DIR, 'covid_test_results.csv'), encoding='utf-8-sig')

td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month
td['quarter'] = td['admit_dt'].dt.quarter

print(f"  Admissions: {len(td):,} rows, {td['住院流水号'].nunique():,} patients")
print(f"  COVID tests: {len(covid)} records")

# ═══════════════════════════════════════════════════════════════
# 2. Comorbidity extraction
# ═══════════════════════════════════════════════════════════════
print("[2] Extracting comorbidities...")
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

# ═══════════════════════════════════════════════════════════════
# 3. COVID status classification
# ═══════════════════════════════════════════════════════════════
print("[3] Classifying COVID status...")
positive_pids = set(covid[covid['status'] == 'positive']['patient_id'].astype(str).unique())
tested_pids = set(covid['patient_id'].astype(str).unique())
td['pid_str'] = td['住院流水号'].astype(str)

td['covid_status'] = 'Not Tested'
td.loc[td['pid_str'].isin(tested_pids), 'covid_status'] = 'COVID-'
td.loc[td['pid_str'].isin(positive_pids), 'covid_status'] = 'COVID+'

covid_pos_2020 = td[(td['covid_status'] == 'COVID+') & (td['year'] == 2020)]
n_pos = len(covid_pos_2020)
print(f"  COVID+ admissions in 2020: n={n_pos}")

# ═══════════════════════════════════════════════════════════════
# 4. Behavioral z-score profiles
# ═══════════════════════════════════════════════════════════════
print("[4] Computing behavioral profiles...")
behavior_cols = ['实际住院天数', '检验项目数', '医嘱数量', '药品医嘱', '检查数量']
lab_cols = ['lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']
all_metrics = behavior_cols + lab_cols

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

covid_pos_profile = zscore_profile(covid_pos_2020)
vec_pos = profile_to_vec(covid_pos_profile)
print(f"  COVID+ reference vector norm: {np.linalg.norm(vec_pos):.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. Compute all similarities
# ═══════════════════════════════════════════════════════════════
top_comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                     'Cerebrovascular', 'Renal', 'Respiratory']
years = [2016, 2017, 2018, 2019]
quarters = [1, 2, 3, 4]

# 5a. Quarterly similarities (Panel B)
print("[5] Computing quarterly similarities...")
quarterly_sim = {c: {} for c in top_comorbidities}
quarterly_n = {c: {} for c in top_comorbidities}

for comor in top_comorbidities:
    for yr in years:
        for q in quarters:
            sub = td[(td[comor] == 1) & (td['year'] == yr) & (td['quarter'] == q)]
            quarterly_n[comor][(yr, q)] = len(sub)
            if len(sub) >= 5:
                prof = zscore_profile(sub)
                v = profile_to_vec(prof)
                if np.linalg.norm(vec_pos) > 0 and np.linalg.norm(v) > 0:
                    quarterly_sim[comor][(yr, q)] = 1 - cosine(vec_pos, v)
                else:
                    quarterly_sim[comor][(yr, q)] = np.nan
            else:
                quarterly_sim[comor][(yr, q)] = np.nan

# 5b. Monthly similarities for 2019 heatmap (Panel A)
print("  Computing 2019 monthly heatmap...")
months_2019 = list(range(1, 13))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
td_2019 = td[td['year'] == 2019]

sim_matrix = np.full((len(top_comorbidities), 12), np.nan)
n_matrix = np.zeros((len(top_comorbidities), 12), dtype=int)

for i, comor in enumerate(top_comorbidities):
    for j, m in enumerate(months_2019):
        sub = td_2019[(td_2019[comor] == 1) & (td_2019['month'] == m)]
        n_matrix[i, j] = len(sub)
        if len(sub) >= 5:
            prof = zscore_profile(sub)
            v = profile_to_vec(prof)
            if np.linalg.norm(vec_pos) > 0 and np.linalg.norm(v) > 0:
                sim_matrix[i, j] = 1 - cosine(vec_pos, v)

# ═══════════════════════════════════════════════════════════════
# 6. Bootstrap confidence intervals (Panel C)
# ═══════════════════════════════════════════════════════════════
print("[6] Computing bootstrap confidence intervals...")
N_BOOT = 2000
np.random.seed(42)

def bootstrap_similarity(sub_df, n_boot=N_BOOT):
    if len(sub_df) < 5:
        return np.nan, np.nan, np.nan, np.nan
    sims = []
    for _ in range(n_boot):
        boot_idx = np.random.choice(len(sub_df), size=len(sub_df), replace=True)
        boot_df = sub_df.iloc[boot_idx]
        prof = zscore_profile(boot_df)
        v = profile_to_vec(prof)
        if np.linalg.norm(vec_pos) > 0 and np.linalg.norm(v) > 0:
            sims.append(1 - cosine(vec_pos, v))
    if len(sims) < 10:
        return np.nan, np.nan, np.nan, np.nan
    sims = np.array(sims)
    return np.mean(sims), np.percentile(sims, 2.5), np.percentile(sims, 97.5), np.std(sims)

boot_results = {}
print("  Bootstrapping Respiratory Q4 across years...")
for yr in years:
    sub = td[(td['Respiratory'] == 1) & (td['year'] == yr) & (td['quarter'] == 4)]
    mean_s, lo, hi, sd = bootstrap_similarity(sub)
    boot_results[('Respiratory', yr, 4)] = {'mean': mean_s, 'ci_lo': lo, 'ci_hi': hi, 'std': sd, 'n': len(sub)}
    if not np.isnan(mean_s):
        print(f"    {yr} Q4: sim={mean_s:.3f} [{lo:.3f}, {hi:.3f}] n={len(sub)}")

print("  Bootstrapping all comorbidities Q4 2019...")
for comor in top_comorbidities:
    sub = td[(td[comor] == 1) & (td['year'] == 2019) & (td['quarter'] == 4)]
    mean_s, lo, hi, sd = bootstrap_similarity(sub)
    boot_results[(comor, 2019, 4)] = {'mean': mean_s, 'ci_lo': lo, 'ci_hi': hi, 'std': sd, 'n': len(sub)}
    if not np.isnan(mean_s):
        print(f"    {comor} 2019Q4: sim={mean_s:.3f} [{lo:.3f}, {hi:.3f}] n={len(sub)}")

print("  Bootstrapping Respiratory 2019 Q1-Q3...")
for q in [1, 2, 3]:
    sub = td[(td['Respiratory'] == 1) & (td['year'] == 2019) & (td['quarter'] == q)]
    mean_s, lo, hi, sd = bootstrap_similarity(sub)
    boot_results[('Respiratory', 2019, q)] = {'mean': mean_s, 'ci_lo': lo, 'ci_hi': hi, 'std': sd, 'n': len(sub)}

# ═══════════════════════════════════════════════════════════════
# 7. COVID testing timeline for lead-time panel (Panel D)
# ═══════════════════════════════════════════════════════════════
print("[7] Preparing COVID testing timeline...")
covid['date_dt'] = pd.to_datetime(covid['admission_date'], errors='coerce')
covid['yearmonth'] = covid['date_dt'].dt.to_period('M')

monthly_tests = covid.groupby('yearmonth').agg(
    total=('status', 'count'),
    positive=('status', lambda x: (x == 'positive').sum()),
).reset_index()
monthly_tests['pos_rate'] = monthly_tests['positive'] / monthly_tests['total'] * 100
monthly_tests['ym_str'] = monthly_tests['yearmonth'].astype(str)

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — 2×2 layout
# gs[0,0]: Panel A heatmap (2019 monthly)
# gs[0,1]: Panel B quarterly time series
# gs[1,0]: Panel C bootstrap CIs
# gs[1,1]: Panel D lead-time analysis
# ═══════════════════════════════════════════════════════════════
print("\n[Figure 2] Generating 2x2 publication figure...")
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35,
                       left=0.08, right=0.97, top=0.94, bottom=0.06)

# Comorbidity color mapping
comor_colors = {
    'Respiratory': COLORS['secondary'],
    'Cardiovascular': COLORS['primary'],
    'Hypertension': COLORS['accent1'],
    'Diabetes': COLORS['accent2'],
    'Cerebrovascular': COLORS['accent3'],
    'Renal': COLORS['accent5'],
}

# ─── Panel A: 2019 Monthly Behavioral Similarity Heatmap ────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

im = ax_a.imshow(sim_matrix, cmap=SPRING_CMAP, aspect='auto', vmin=-0.5, vmax=1)

for i in range(sim_matrix.shape[0]):
    for j in range(sim_matrix.shape[1]):
        v = sim_matrix[i, j]
        n = n_matrix[i, j]
        if not np.isnan(v):
            tc = 'white' if v > 0.75 or v < -0.2 else COLORS['text']
            ax_a.text(j, i, f'{v:.2f}\n({n})', ha='center', va='center',
                      fontsize=6.5, color=tc, fontweight='bold')
        else:
            ax_a.text(j, i, f'n={n}', ha='center', va='center',
                      fontsize=6.5, color='gray')

ax_a.set_xticks(range(12))
ax_a.set_xticklabels(month_names, fontsize=8)
ax_a.set_yticks(range(len(top_comorbidities)))
ax_a.set_yticklabels(top_comorbidities, fontsize=9)
ax_a.set_title('2019 Monthly Behavioral Similarity to COVID+ Profile', fontsize=10)

cbar = plt.colorbar(im, ax=ax_a, fraction=0.025, pad=0.03, shrink=0.85)
cbar.set_label('Cosine Similarity', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Highlight top-3 cells
flat_sim = []
for i in range(sim_matrix.shape[0]):
    for j in range(sim_matrix.shape[1]):
        if not np.isnan(sim_matrix[i, j]):
            flat_sim.append((sim_matrix[i, j], i, j))
flat_sim.sort(reverse=True)

for rank, (val, ri, rj) in enumerate(flat_sim[:3]):
    rect = plt.Rectangle((rj - 0.5, ri - 0.5), 1, 1, fill=False,
                          edgecolor=COLORS['secondary'] if rank == 0 else COLORS['accent2'],
                          linewidth=2.0 if rank == 0 else 1.3, linestyle='-' if rank == 0 else '--')
    ax_a.add_patch(rect)

if flat_sim:
    best_val, best_i, best_j = flat_sim[0]
    best_comor = top_comorbidities[best_i]
    best_month = month_names[best_j]
    ax_a.annotate(f'Peak: {best_comor}\n{best_month} 2019 (sim={best_val:.3f})',
                  xy=(best_j, best_i),
                  xytext=(0.02, 0.96), textcoords='axes fraction',
                  fontsize=7.5, va='top',
                  arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.3),
                  bbox=dict(boxstyle='round,pad=0.3', fc='white',
                            ec=COLORS['secondary'], alpha=0.95))

# ─── Panel B: Multi-year quarterly similarity time series ────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
x_ticks = []
x_labels_b = []
for yr in years:
    for q in quarters:
        x_ticks.append(len(x_ticks))
        x_labels_b.append(f"{yr}\n{quarter_labels[q-1]}")

for comor in top_comorbidities:
    vals = []
    valid_x = []
    for xi, (yr, q) in enumerate([(y, q) for y in years for q in quarters]):
        s = quarterly_sim[comor].get((yr, q), np.nan)
        if not np.isnan(s):
            vals.append(s)
            valid_x.append(xi)
    lw = 2.5 if comor == 'Respiratory' else 1.2
    alpha = 1.0 if comor == 'Respiratory' else 0.55
    ms = 6 if comor == 'Respiratory' else 3.5
    ax_b.plot(valid_x, vals, '-o', color=comor_colors[comor], linewidth=lw,
              markersize=ms, alpha=alpha, label=comor, zorder=10 if comor == 'Respiratory' else 5)

# Highlight Q4 2019
idx_2019q4 = 15
resp_2019q4_val = quarterly_sim['Respiratory'].get((2019, 4), np.nan)
if not np.isnan(resp_2019q4_val):
    ax_b.annotate(f'Resp. 2019Q4\nsim={resp_2019q4_val:.3f}',
                  xy=(idx_2019q4, resp_2019q4_val),
                  xytext=(idx_2019q4 - 4, resp_2019q4_val + 0.08),
                  fontsize=7.5, fontweight='bold', color=COLORS['secondary'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.3),
                  bbox=dict(boxstyle='round,pad=0.2', fc='white',
                            ec=COLORS['secondary'], alpha=0.95))

# Spearman trend test
resp_q_vals = []
for yi, yr in enumerate(years):
    for q in quarters:
        s = quarterly_sim['Respiratory'].get((yr, q), np.nan)
        if not np.isnan(s):
            resp_q_vals.append((yi * 4 + q, s))
if len(resp_q_vals) >= 4:
    xv, yv = zip(*resp_q_vals)
    rho_trend, p_trend = stats.spearmanr(xv, yv)
    ax_b.text(0.02, 0.02, f'Respiratory trend: $\\rho$={rho_trend:.3f}, p={p_trend:.4f}',
              transform=ax_b.transAxes, fontsize=7, ha='left', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['grid'], alpha=0.9))

for yr_idx in range(len(years)):
    q4_x = yr_idx * 4 + 3
    ax_b.axvspan(q4_x - 0.4, q4_x + 0.4, alpha=0.06, color='grey')

ax_b.set_xticks(x_ticks)
ax_b.set_xticklabels(x_labels_b, fontsize=6)
ax_b.set_ylabel('Cosine Similarity to COVID+ Profile')
ax_b.set_title('Quarterly Similarity to COVID+ (2016-2019)')
ax_b.legend(fontsize=6.5, ncol=2, loc='lower left', framealpha=0.9)
ax_b.set_ylim(-0.3, 1.05)
ax_b.axhline(0, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)

# ─── Panel C: Bootstrap CI comparison ────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, 'C')

plot_entries = []
for yr in years:
    key = ('Respiratory', yr, 4)
    if key in boot_results and not np.isnan(boot_results[key]['mean']):
        br = boot_results[key]
        plot_entries.append({
            'label': f'Resp. {yr}Q4', 'mean': br['mean'],
            'ci_lo': br['ci_lo'], 'ci_hi': br['ci_hi'],
            'n': br['n'], 'color': COLORS['secondary'], 'section': 'year'
        })

plot_entries.append({'label': '', 'mean': np.nan, 'section': 'sep'})

for comor in top_comorbidities:
    key = (comor, 2019, 4)
    if key in boot_results and not np.isnan(boot_results[key]['mean']):
        br = boot_results[key]
        plot_entries.append({
            'label': f'{comor}\n2019Q4', 'mean': br['mean'],
            'ci_lo': br['ci_lo'], 'ci_hi': br['ci_hi'],
            'n': br['n'], 'color': comor_colors[comor], 'section': 'comor'
        })

for i, entry in enumerate(plot_entries):
    if entry['section'] == 'sep':
        ax_c.axhline(i, color=COLORS['grid'], linewidth=1, linestyle='--', alpha=0.5)
        continue
    ax_c.errorbar(entry['mean'], i,
                  xerr=[[entry['mean'] - entry['ci_lo']], [entry['ci_hi'] - entry['mean']]],
                  fmt='o', color=entry['color'], markersize=7, capsize=3.5, capthick=1.3,
                  elinewidth=1.3, markeredgecolor='white', markeredgewidth=0.5)
    ax_c.text(entry['ci_hi'] + 0.02, i, f"n={entry['n']}",
              va='center', fontsize=6.5, color='grey')

ax_c.set_yticks([i for i, e in enumerate(plot_entries) if e['section'] != 'sep'])
ax_c.set_yticklabels([e['label'] for e in plot_entries if e['section'] != 'sep'], fontsize=7)
ax_c.set_xlabel('Cosine Similarity (bootstrap 95% CI)')
ax_c.set_title('Bootstrap Confidence Intervals')
ax_c.axvline(0, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax_c.invert_yaxis()

yr_entries = [i for i, e in enumerate(plot_entries) if e.get('section') == 'year']
comor_entries = [i for i, e in enumerate(plot_entries) if e.get('section') == 'comor']
n_items = len(plot_entries)
if yr_entries:
    yr_frac = 1.0 - (np.mean(yr_entries) + 0.5) / n_items  # inverted y-axis
    ax_c.text(-0.08, yr_frac, 'Respiratory\nacross years', fontsize=6.5,
              ha='center', va='center', fontstyle='italic', color=COLORS['text'],
              transform=ax_c.transAxes)
if comor_entries:
    comor_frac = 1.0 - (np.mean(comor_entries) + 0.5) / n_items
    ax_c.text(-0.08, comor_frac, 'All groups\n2019 Q4', fontsize=6.5,
              ha='center', va='center', fontstyle='italic', color=COLORS['text'],
              transform=ax_c.transAxes)

# ─── Panel D: Lead-time analysis ─────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, 'D')

month_labels_d = []
resp_monthly_sim = []
for m in range(1, 13):
    sub = td[(td['Respiratory'] == 1) & (td['year'] == 2019) & (td['month'] == m)]
    if len(sub) >= 5:
        prof = zscore_profile(sub)
        v = profile_to_vec(prof)
        if np.linalg.norm(vec_pos) > 0 and np.linalg.norm(v) > 0:
            resp_monthly_sim.append(1 - cosine(vec_pos, v))
        else:
            resp_monthly_sim.append(np.nan)
    else:
        resp_monthly_sim.append(np.nan)
    month_labels_d.append(f"2019-{m:02d}")

for m in range(1, 7):
    sub = td[(td['Respiratory'] == 1) & (td['year'] == 2020) & (td['month'] == m)]
    if len(sub) >= 5:
        prof = zscore_profile(sub)
        v = profile_to_vec(prof)
        if np.linalg.norm(vec_pos) > 0 and np.linalg.norm(v) > 0:
            resp_monthly_sim.append(1 - cosine(vec_pos, v))
        else:
            resp_monthly_sim.append(np.nan)
    else:
        resp_monthly_sim.append(np.nan)
    month_labels_d.append(f"2020-{m:02d}")

x_d = np.arange(len(month_labels_d))
valid_mask = [not np.isnan(v) for v in resp_monthly_sim]
valid_x = [x for x, v in zip(x_d, valid_mask) if v]
valid_y = [y for y, v in zip(resp_monthly_sim, valid_mask) if v]

ax_d.plot(valid_x, valid_y, '-o', color=COLORS['secondary'], linewidth=2,
          markersize=5, label='Respiratory similarity\nto COVID+', zorder=5)
ax_d.fill_between(valid_x, 0, valid_y, alpha=0.12, color=COLORS['secondary'])

# COVID testing positive rate overlay
ax_d2 = ax_d.twinx()
covid_rate_x, covid_rate_y = [], []
for mi, ml in enumerate(month_labels_d):
    matches = monthly_tests[monthly_tests['ym_str'] == ml]
    if len(matches) > 0:
        covid_rate_x.append(mi)
        covid_rate_y.append(matches.iloc[0]['pos_rate'])

if covid_rate_x:
    ax_d2.bar(covid_rate_x, covid_rate_y, width=0.5, alpha=0.35,
              color=COLORS['accent2'], label='COVID+ rate (%)', zorder=2)
    ax_d2.set_ylabel('COVID+ Rate (%)', color=COLORS['accent2'], fontsize=8)
    ax_d2.tick_params(axis='y', labelcolor=COLORS['accent2'])
    ax_d2.set_ylim(0, max(covid_rate_y) * 1.5 if covid_rate_y else 30)
    ax_d2.legend(loc='upper right', fontsize=6.5, framealpha=0.9)

# Lead-time annotation
scan_months = list(range(6, 12))
scan_vals = [(resp_monthly_sim[m] if not np.isnan(resp_monthly_sim[m]) else -1) for m in scan_months]
peak_idx = scan_months[np.argmax(scan_vals)]
peak_val = resp_monthly_sim[peak_idx]

earliest_elevated_idx = None
for m in scan_months:
    v = resp_monthly_sim[m] if not np.isnan(resp_monthly_sim[m]) else 0
    if v >= 0.75:
        earliest_elevated_idx = m
        break

covid_detect_idx = 12  # 2020-01

if not np.isnan(peak_val):
    lead_start = earliest_elevated_idx if earliest_elevated_idx is not None else peak_idx
    lead_months = covid_detect_idx - lead_start
    arrow_y = 0.35
    ax_d.annotate('', xy=(covid_detect_idx, arrow_y), xytext=(lead_start, arrow_y),
                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.8))
    ax_d.text((lead_start + covid_detect_idx) / 2, arrow_y + 0.06,
              f'Lead time: {lead_months} months',
              ha='center', fontsize=7.5, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='grey', alpha=0.9))

    ax_d.annotate(f'Peak: {month_labels_d[peak_idx]}\nsim={peak_val:.3f}',
                  xy=(peak_idx, peak_val),
                  xytext=(max(0, peak_idx - 3), min(peak_val + 0.13, 1.1)),
                  fontsize=7, fontweight='bold', color=COLORS['secondary'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1),
                  bbox=dict(boxstyle='round,pad=0.2', fc='white',
                            ec=COLORS['secondary'], alpha=0.9))

# H2 2019 elevated signal shading
ax_d.axvspan(6 - 0.4, 11 + 0.4, alpha=0.05, color=COLORS['secondary'])
ax_d.text(8.5, -0.15, 'H2 2019: Elevated signal', ha='center', fontsize=6.5,
          color=COLORS['secondary'], fontstyle='italic')
ax_d.axvspan(12 - 0.4, 14 + 0.4, alpha=0.07, color=COLORS['accent2'])
ax_d.text(13, -0.15, 'COVID-19\nDetection', ha='center', fontsize=6.5,
          color=COLORS['accent2'], fontstyle='italic')
ax_d.axvline(12, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax_d.text(12.1, 1.05, 'Wuhan\nLockdown', fontsize=6, color='grey', va='top')

ax_d.set_xticks(x_d[::2])
ax_d.set_xticklabels([month_labels_d[i] for i in range(0, len(month_labels_d), 2)],
                     fontsize=6.5, rotation=45, ha='right')
ax_d.set_ylabel('Cosine Similarity')
ax_d.set_title('Lead-Time: Behavioral Signal vs COVID Detection')
ax_d.legend(loc='upper left', fontsize=6.5, framealpha=0.9)

# ─── Save figure ─────────────────────────────────────────────
plt.tight_layout()
save_figure(fig, 'Figure2_sentinel_discovery')

# ═══════════════════════════════════════════════════════════════
# Save numerical results
# ═══════════════════════════════════════════════════════════════
results = {
    'quarterly_similarity': {},
    'bootstrap_ci': {},
    'lead_time_months': int(covid_detect_idx - (earliest_elevated_idx if earliest_elevated_idx is not None else peak_idx)) if not np.isnan(peak_val) else None,
    'peak_signal': {
        'month': month_labels_d[peak_idx] if not np.isnan(peak_val) else None,
        'similarity': float(peak_val) if not np.isnan(peak_val) else None,
    },
    'earliest_elevated_signal': {
        'month': month_labels_d[earliest_elevated_idx] if earliest_elevated_idx is not None else None,
        'similarity': float(resp_monthly_sim[earliest_elevated_idx]) if earliest_elevated_idx is not None else None,
    },
    'covid_pos_n': int(n_pos),
}

for comor in top_comorbidities:
    results['quarterly_similarity'][comor] = {}
    for yr in years:
        for q in quarters:
            s = quarterly_sim[comor].get((yr, q), np.nan)
            if not np.isnan(s):
                results['quarterly_similarity'][comor][f'{yr}Q{q}'] = round(float(s), 4)

for key, val in boot_results.items():
    if not np.isnan(val['mean']):
        label = f'{key[0]}_{key[1]}Q{key[2]}'
        results['bootstrap_ci'][label] = {
            'mean': round(val['mean'], 4),
            'ci_lo': round(val['ci_lo'], 4),
            'ci_hi': round(val['ci_hi'], 4),
            'n': val['n'],
        }

out_json = os.path.join(OUTPUT_DIR, 'fig2_sentinel_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n  Results saved: {out_json}")

print("\n" + "=" * 60)
print("FIGURE 2 SUMMARY")
print("=" * 60)
print(f"  COVID+ reference: n={n_pos} admissions in 2020")
print(f"  Feature dimensions: {len(all_metrics)} (5 utilization + 8 lab)")
if not np.isnan(resp_2019q4_val):
    print(f"  Respiratory 2019 Q4 similarity: {resp_2019q4_val:.3f}")
if not np.isnan(peak_val):
    print(f"  Peak monthly signal: {month_labels_d[peak_idx]}, sim={peak_val:.3f}")
    lead_mo = covid_detect_idx - (earliest_elevated_idx if earliest_elevated_idx is not None else peak_idx)
    print(f"  Earliest elevated signal: {month_labels_d[earliest_elevated_idx] if earliest_elevated_idx is not None else 'N/A'}")
    print(f"  Lead time: up to {lead_mo} months")
print("  Output: Figure2_sentinel_discovery.png / .pdf / .tif")
print("  Done!")
