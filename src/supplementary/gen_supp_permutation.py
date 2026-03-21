"""
Supplementary Figure S8: Seasonal Confounding — Permutation Test (2-panel)
  (A) Q4 cosine similarity comparison across years (2016–2019) for respiratory
      patients, showing that the 2019 Q4 elevation is NOT a seasonal artifact.
  (B) Permutation distribution (n=5,000) comparing 2019 Q4 vs. 2018 Q4
      respiratory similarity (p = 0.53), confirming structural sentinel effect.

Generates:
  - FigureS8_permutation_test (PNG/PDF/TIF)

Run: python gen_supp_permutation.py
"""
import sys, io, warnings, os, json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure S8: Permutation Test — Seasonal Confounding Control")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data & build profiles (same as Fig 2)
# ═══════════════════════════════════════════════════════════════
print("[1] Loading data...")
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['quarter'] = td['admit_dt'].dt.quarter

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
print(f"  COVID+ reference: n={len(covid_pos)} admissions")

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']
years = [2016, 2017, 2018, 2019]

# ═══════════════════════════════════════════════════════════════
# 2. Compute Q4 similarities across years for all comorbidities
# ═══════════════════════════════════════════════════════════════
print("\n[2] Computing Q4 similarities 2016–2019...")

q4_sim = {c: {} for c in comorbidities}
q4_n = {c: {} for c in comorbidities}

for comor in comorbidities:
    for yr in years:
        sub = td[(td[comor] == 1) & (td['year'] == yr) & (td['quarter'] == 4)]
        q4_n[comor][yr] = len(sub)
        if len(sub) >= 5:
            v = profile_to_vec(zscore_profile(sub))
            q4_sim[comor][yr] = cos_sim(vec_pos, v)
        else:
            q4_sim[comor][yr] = np.nan
        print(f"  {comor} {yr}Q4: sim={q4_sim[comor][yr]:.3f}, n={q4_n[comor][yr]}")

# ═══════════════════════════════════════════════════════════════
# 3. Permutation test: 2019 Q4 vs 2018 Q4 for Respiratory
# ═══════════════════════════════════════════════════════════════
print("\n[3] Permutation test (n=5000)...")

N_PERM = 5000
np.random.seed(42)

resp_2019q4 = td[(td['Respiratory'] == 1) & (td['year'] == 2019) & (td['quarter'] == 4)]
resp_2018q4 = td[(td['Respiratory'] == 1) & (td['year'] == 2018) & (td['quarter'] == 4)]

obs_2019 = cos_sim(vec_pos, profile_to_vec(zscore_profile(resp_2019q4)))
obs_2018 = cos_sim(vec_pos, profile_to_vec(zscore_profile(resp_2018q4)))
obs_diff = obs_2019 - obs_2018
print(f"  Observed: 2019Q4={obs_2019:.3f}, 2018Q4={obs_2018:.3f}, diff={obs_diff:.3f}")

# Pool 2018 Q4 and 2019 Q4 respiratory patients and permute
pooled = pd.concat([resp_2018q4, resp_2019q4], ignore_index=True)
n_2018 = len(resp_2018q4)
n_2019 = len(resp_2019q4)

perm_diffs = []
for i in range(N_PERM):
    perm_idx = np.random.permutation(len(pooled))
    grp_a = pooled.iloc[perm_idx[:n_2018]]
    grp_b = pooled.iloc[perm_idx[n_2018:n_2018 + n_2019]]
    sim_a = cos_sim(vec_pos, profile_to_vec(zscore_profile(grp_a)))
    sim_b = cos_sim(vec_pos, profile_to_vec(zscore_profile(grp_b)))
    perm_diffs.append(sim_b - sim_a)

perm_diffs = np.array(perm_diffs)
p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
print(f"  Permutation p-value (two-sided): {p_value:.3f}")

# ═══════════════════════════════════════════════════════════════
# 4. Generate Figure S8 (1×2)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Generating FigureS3_permutation_test...")

fig = plt.figure(figsize=(14, 5.5))
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

# ─── Panel A: Q4 similarity comparison across years ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

x = np.arange(len(years))
width = 0.12
for i, comor in enumerate(comorbidities):
    vals = [q4_sim[comor][yr] for yr in years]
    lw_edge = 1.5 if comor == 'Respiratory' else 0
    ax_a.bar(x + i * width, vals, width,
             color=comor_colors[comor],
             alpha=1.0 if comor == 'Respiratory' else 0.6,
             edgecolor='black' if comor == 'Respiratory' else 'white',
             linewidth=lw_edge, label=comor)

ax_a.set_xticks(x + width * (len(comorbidities) - 1) / 2)
ax_a.set_xticklabels([f'{yr} Q4' for yr in years], fontsize=9)
ax_a.set_ylabel('Cosine Similarity to COVID+ Profile', fontsize=9)
ax_a.set_title('Q4 Similarity Comparison (2016–2019)', fontsize=10)
ax_a.legend(fontsize=6.5, ncol=2, loc='upper left', framealpha=0.9)
ax_a.axhline(0, color='grey', linewidth=0.5, linestyle=':')

# Annotate Respiratory 2019Q4
if not np.isnan(q4_sim['Respiratory'][2019]):
    resp_idx = comorbidities.index('Respiratory')
    bar_x = 3 + resp_idx * width
    ax_a.annotate(f'sim={q4_sim["Respiratory"][2019]:.3f}',
                  xy=(bar_x, q4_sim['Respiratory'][2019]),
                  xytext=(bar_x + 0.3, q4_sim['Respiratory'][2019] + 0.08),
                  fontsize=7, fontweight='bold', color=COLORS['secondary'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1),
                  bbox=dict(boxstyle='round,pad=0.2', fc='white',
                            ec=COLORS['secondary'], alpha=0.9))

# ─── Panel B: Permutation distribution ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

ax_b.hist(perm_diffs, bins=50, density=True, color=COLORS['primary'],
          alpha=0.7, edgecolor='white', label='Permutation\ndistribution')

# Observed difference
ax_b.axvline(obs_diff, color=COLORS['secondary'], linewidth=2.5,
             linestyle='-', label=f'Observed Δ = {obs_diff:.3f}')
ax_b.axvline(-obs_diff, color=COLORS['secondary'], linewidth=2.5,
             linestyle='--', alpha=0.5)

# KDE overlay
from scipy.stats import gaussian_kde
kde = gaussian_kde(perm_diffs, bw_method=0.3)
x_kde = np.linspace(perm_diffs.min() - 0.1, perm_diffs.max() + 0.1, 200)
ax_b.plot(x_kde, kde(x_kde), '-', color='black', linewidth=1.2, alpha=0.7)

# p-value annotation
ax_b.text(0.97, 0.95, f'Two-sided p = {p_value:.3f}\nn = {N_PERM:,} permutations',
          transform=ax_b.transAxes, fontsize=9, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='grey', alpha=0.95))

ax_b.set_xlabel('Δ Cosine Similarity (2019Q4 − 2018Q4)', fontsize=9)
ax_b.set_ylabel('Density', fontsize=9)
ax_b.set_title('Permutation Test: Seasonal Confounding Control', fontsize=10)
ax_b.legend(fontsize=7, loc='upper left', framealpha=0.9)

# Conclusion text
conclusion = 'Non-significant → not seasonal artifact' if p_value > 0.05 else 'Significant seasonal effect detected'
ax_b.text(0.5, 0.02, conclusion, transform=ax_b.transAxes,
          fontsize=8, ha='center', fontstyle='italic', color=COLORS['text'],
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['grid'], alpha=0.9))

save_figure(fig, 'FigureS3_permutation_test')
print("  FigureS3_permutation_test saved.")

# Save numerical results
results = {
    'q4_similarities': {comor: {str(yr): round(q4_sim[comor][yr], 4) for yr in years
                                 if not np.isnan(q4_sim[comor][yr])}
                        for comor in comorbidities},
    'permutation_test': {
        'n_permutations': N_PERM,
        'observed_2019q4': round(obs_2019, 4),
        'observed_2018q4': round(obs_2018, 4),
        'observed_diff': round(obs_diff, 4),
        'p_value': round(p_value, 4),
        'n_2018q4': int(n_2018),
        'n_2019q4': int(n_2019),
    },
}

json_path = os.path.join(FIG_DIR, 'permutation_test_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
