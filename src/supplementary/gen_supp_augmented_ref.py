"""
Supplementary Analysis: Expanded Pandemic-Positive Reference
  + Synthetic augmentation to ≥100 cases
  + Strict non-respiratory references
  + Rerun sentinel analyses

Generates:
  - FigureS13_augmented_sentinel (PNG/PDF/TIF)
  - augmented_sentinel_results.json

Run: python gen_supp_augmented_ref.py
"""
import sys, io, warnings, os, json
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
print("Augmented Pandemic-Positive Reference + Strict Non-Resp")
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

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']

# ═══════════════════════════════════════════════════════════════
# 2. Build original + augmented pandemic-positive references
# ═══════════════════════════════════════════════════════════════
print("\n[2] Building reference vectors...")

covid_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'covid_test_results.csv'), encoding='utf-8-sig')
positive_pids = set(covid_test[covid_test['status'] == 'positive']['patient_id'].astype(str).unique())
td['pid_str'] = td['住院流水号'].astype(str)

# Original 2020 reference (n=19)
covid_pos_2020 = td[(td['pid_str'].isin(positive_pids)) & (td['year'] == 2020)]
vec_original = profile_to_vec(zscore_profile(covid_pos_2020))
n_original = len(covid_pos_2020)
print(f"  Original COVID+ reference: n={n_original}")

# Strategy A: Combine with peak respiratory-season admissions (Jan–Feb flu peaks)
# These share organ-system behavioral burden without requiring pandemic labels
flu_peak = td[(td['year'].isin([2017, 2018, 2019])) &
              (td['month'].isin([1, 2])) &
              (td['Respiratory'] == 1)]
print(f"  Respiratory flu-peak admissions (2017-19 Jan/Feb): n={len(flu_peak)}")

# Strategy B: Combine original COVID+ with flu-peak respiratory admissions
combined_respiratory_burden = pd.concat([covid_pos_2020, flu_peak], ignore_index=True)
vec_combined = profile_to_vec(zscore_profile(combined_respiratory_burden))
n_combined = len(combined_respiratory_burden)
print(f"  Combined respiratory-burden reference: n={n_combined}")

# Strategy C: Bootstrap augmentation of COVID+ to ≥100 pseudo-admissions
# Combine COVID+ with respiratory flu-peak admissions and resample to TARGET_N
TARGET_N = 100
rng = np.random.default_rng(42)

# Pool all respiratory-burden admissions: COVID+ (2020) + flu-peak respiratory
# + any respiratory admissions from pandemic months
pool = pd.concat([covid_pos_2020, flu_peak], ignore_index=True)

# Bootstrap resample the pool to TARGET_N admissions
boot_indices = rng.choice(len(pool), size=TARGET_N, replace=True)
augmented_pool = pool.iloc[boot_indices]
vec_augmented = profile_to_vec(zscore_profile(augmented_pool))
print(f"  Bootstrap-augmented reference: n={TARGET_N} (resampled from pool of {len(pool)})")

# Strategy D: Strict non-respiratory COVID+ reference
covid_pos_nonresp = covid_pos_2020[covid_pos_2020['Respiratory'] == 0]
n_nonresp = len(covid_pos_nonresp)
vec_nonresp = profile_to_vec(zscore_profile(covid_pos_nonresp)) if n_nonresp >= 3 else None
print(f"  Non-respiratory COVID+ reference: n={n_nonresp}")

# ═══════════════════════════════════════════════════════════════
# 3. Sentinel analysis with each reference
# ═══════════════════════════════════════════════════════════════
print("\n[3] Sentinel analysis across reference strategies...")

q4_2019 = td[(td['year'] == 2019) & (td['month'].isin([10, 11, 12]))]

references = {
    f'Original COVID+ (n={n_original})': vec_original,
    f'Combined resp-burden (n={n_combined})': vec_combined,
    f'Bootstrap-augmented (n={TARGET_N})': vec_augmented,
}
if vec_nonresp is not None:
    references[f'Non-resp COVID+ (n={n_nonresp})'] = vec_nonresp

all_results = {}
for ref_name, ref_vec in references.items():
    sims = {}
    for comor in comorbidities:
        sub = q4_2019[q4_2019[comor] == 1]
        if len(sub) >= 5:
            v = profile_to_vec(zscore_profile(sub))
            sims[comor] = cos_sim(ref_vec, v)
        else:
            sims[comor] = np.nan

    sorted_c = sorted([(c, s) for c, s in sims.items() if not np.isnan(s)],
                       key=lambda x: x[1], reverse=True)
    resp_rank = next((i + 1 for i, (c, _) in enumerate(sorted_c) if c == 'Respiratory'),
                     len(comorbidities))

    all_results[ref_name] = {
        'similarities': sims,
        'ranking': [(c, float(s)) for c, s in sorted_c],
        'respiratory_rank': resp_rank,
        'respiratory_sim': sims.get('Respiratory', np.nan),
    }
    print(f"\n  {ref_name}:")
    for c, s in sorted_c:
        marker = " <<<" if c == 'Respiratory' else ""
        print(f"    {c:<20s}: {s:.3f}{marker}")

# ═══════════════════════════════════════════════════════════════
# 4. Bootstrap CIs for each reference strategy
# ═══════════════════════════════════════════════════════════════
print("\n[4] Bootstrap CIs (n=2000) for each reference...")

N_BOOT = 2000
boot_ci_results = {}

for ref_name, ref_vec in references.items():
    ci_data = {}
    for comor in comorbidities:
        sub = q4_2019[q4_2019[comor] == 1]
        if len(sub) < 5:
            ci_data[comor] = {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
            continue
        boot_sims = []
        for _ in range(N_BOOT):
            idx = rng.choice(len(sub), size=len(sub), replace=True)
            boot_sub = sub.iloc[idx]
            v = profile_to_vec(zscore_profile(boot_sub))
            boot_sims.append(cos_sim(ref_vec, v))
        ci_data[comor] = {
            'mean': float(np.mean(boot_sims)),
            'ci_lo': float(np.percentile(boot_sims, 2.5)),
            'ci_hi': float(np.percentile(boot_sims, 97.5)),
        }
    boot_ci_results[ref_name] = ci_data
    resp_ci = ci_data.get('Respiratory', {})
    print(f"  {ref_name}: Resp sim = {resp_ci.get('mean', 0):.3f} "
          f"[{resp_ci.get('ci_lo', 0):.3f}–{resp_ci.get('ci_hi', 0):.3f}]")

# ═══════════════════════════════════════════════════════════════
# 5. Negative control with strict non-respiratory reference
# ═══════════════════════════════════════════════════════════════
print("\n[5] Negative control: organ-system specificity...")

# Heart and diabetes references (same as negative_control.py)
heart_2020 = td[(td['year'] == 2020) & (td['Cardiovascular'] == 1)]
diab_2020 = td[(td['year'] == 2020) & (td['Diabetes'] == 1)]
vec_heart = profile_to_vec(zscore_profile(heart_2020))
vec_diab = profile_to_vec(zscore_profile(diab_2020))

neg_ctrl_refs = {
    f'COVID+ original (n={n_original})': vec_original,
    f'Heart Disease (n={len(heart_2020)})': vec_heart,
    f'Diabetes (n={len(diab_2020)})': vec_diab,
}
if vec_nonresp is not None:
    neg_ctrl_refs[f'Non-resp COVID+ (n={n_nonresp})'] = vec_nonresp

neg_ctrl_results = {}
for ref_name, ref_vec in neg_ctrl_refs.items():
    sims = {}
    for comor in comorbidities:
        sub = q4_2019[q4_2019[comor] == 1]
        if len(sub) >= 5:
            v = profile_to_vec(zscore_profile(sub))
            sims[comor] = cos_sim(ref_vec, v)
    sorted_c = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    neg_ctrl_results[ref_name] = {
        'top_group': sorted_c[0][0] if sorted_c else 'N/A',
        'top_sim': sorted_c[0][1] if sorted_c else 0,
        'all_sims': {c: float(s) for c, s in sims.items()},
    }
    print(f"  {ref_name}: Top = {sorted_c[0][0]} ({sorted_c[0][1]:.3f})")

# ═══════════════════════════════════════════════════════════════
# 6. Generate Figure S13: Augmented Reference Analysis (2×2)
# ═══════════════════════════════════════════════════════════════
print("\n[6] Generating FigureS13_augmented_sentinel...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, wspace=0.30, hspace=0.35,
                       left=0.08, right=0.96, top=0.94, bottom=0.06)

# ─── Panel A: Similarity by reference strategy ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

ref_names_short = {
    f'Original COVID+ (n={n_original})': f'Original\n(n={n_original})',
    f'Combined resp-burden (n={n_combined})': f'Combined\n(n={n_combined})',
    f'Bootstrap-augmented (n={TARGET_N})': f'Augmented\n(n={TARGET_N})',
}
if vec_nonresp is not None:
    ref_names_short[f'Non-resp COVID+ (n={n_nonresp})'] = f'Non-resp\n(n={n_nonresp})'

x = np.arange(len(comorbidities))
width = 0.8 / len(references)
comor_short = {'Cardiovascular': 'Cardio.', 'Hypertension': 'Hypert.',
               'Diabetes': 'Diabetes', 'Cerebrovascular': 'Cerebro.',
               'Renal': 'Renal', 'Respiratory': 'Resp.'}

for i, (rname, rdata) in enumerate(all_results.items()):
    vals = [rdata['similarities'].get(c, 0) for c in comorbidities]
    short = ref_names_short.get(rname, rname[:15])
    ax_a.bar(x + (i - len(references) / 2 + 0.5) * width, vals, width,
             label=short, color=PAL[i], edgecolor='white', alpha=0.85)

ax_a.set_xticks(x)
ax_a.set_xticklabels([comor_short[c] for c in comorbidities], fontsize=8)
ax_a.set_ylabel('Cosine Similarity to Reference', fontsize=9)
ax_a.set_title('Sentinel Similarity: Multiple Reference Strategies (Q4 2019)', fontsize=10)
ax_a.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.9)

# ─── Panel B: Bootstrap CIs per reference ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

y_pos = 0
y_labels = []
y_ticks = []
for rname, ci_data in boot_ci_results.items():
    resp_ci = ci_data.get('Respiratory', {})
    mean_val = resp_ci.get('mean', 0)
    lo = resp_ci.get('ci_lo', 0)
    hi = resp_ci.get('ci_hi', 0)
    short = ref_names_short.get(rname, rname[:20])
    ax_b.barh(y_pos, mean_val, height=0.6, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax_b.errorbar(mean_val, y_pos, xerr=[[mean_val - lo], [hi - mean_val]],
                  fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    ax_b.text(hi + 0.02, y_pos, f'{mean_val:.3f}\n[{lo:.3f}–{hi:.3f}]',
              va='center', fontsize=7)
    y_labels.append(short.replace('\n', ' '))
    y_ticks.append(y_pos)
    y_pos += 1

ax_b.set_yticks(y_ticks)
ax_b.set_yticklabels(y_labels, fontsize=8)
ax_b.set_xlabel('Respiratory Cosine Similarity (95% CI)', fontsize=9)
ax_b.set_title('Bootstrap CIs: Respiratory Sentinel Signal', fontsize=10)
ax_b.invert_yaxis()

# ─── Panel C: Negative control heatmap ───
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, 'C')

ref_order = list(neg_ctrl_results.keys())
heatmap_data = np.zeros((len(ref_order), len(comorbidities)))
for i, rname in enumerate(ref_order):
    for j, comor in enumerate(comorbidities):
        heatmap_data[i, j] = neg_ctrl_results[rname]['all_sims'].get(comor, 0)

im = ax_c.imshow(heatmap_data, cmap=SPRING_CMAP, aspect='auto', vmin=-0.2, vmax=1.0)
ax_c.set_xticks(range(len(comorbidities)))
ax_c.set_xticklabels([comor_short[c] for c in comorbidities], fontsize=8, rotation=30, ha='right')
ax_c.set_yticks(range(len(ref_order)))
ax_c.set_yticklabels([r[:30] for r in ref_order], fontsize=7)
ax_c.set_title('Negative Control: Organ-System Specificity', fontsize=10)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax_c.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.colorbar(im, ax=ax_c, shrink=0.8, label='Cosine Similarity')

# ─── Panel D: Augmentation stability ───
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, 'D')

# Run augmentation at different sample sizes from the pool
sample_sizes = [20, 50, 100, 200, 500]
n_trials = 50

stability_means = np.zeros(len(sample_sizes))
stability_stds = np.zeros(len(sample_sizes))
for si, n_samp in enumerate(sample_sizes):
    sims_trial = []
    for trial in range(n_trials):
        boot_idx = rng.choice(len(pool), size=min(n_samp, len(pool)), replace=True)
        ref_aug = profile_to_vec(zscore_profile(pool.iloc[boot_idx]))
        resp_sub = q4_2019[q4_2019['Respiratory'] == 1]
        if len(resp_sub) >= 5:
            v = profile_to_vec(zscore_profile(resp_sub))
            sims_trial.append(cos_sim(ref_aug, v))
    stability_means[si] = np.mean(sims_trial) if sims_trial else 0
    stability_stds[si] = np.std(sims_trial) if sims_trial else 0

ax_d.bar(range(len(sample_sizes)), stability_means, color=COLORS['primary'],
         edgecolor='white', alpha=0.8)
ax_d.errorbar(range(len(sample_sizes)), stability_means, yerr=stability_stds,
              fmt='none', ecolor='black', capsize=4, linewidth=1.2)
ax_d.set_xticks(range(len(sample_sizes)))
ax_d.set_xticklabels([str(s) for s in sample_sizes], fontsize=8)
ax_d.set_xlabel('Bootstrap Sample Size', fontsize=9)
ax_d.set_ylabel('Respiratory Similarity (mean ± SD)', fontsize=9)
ax_d.set_title('Augmentation Stability by Sample Size', fontsize=10)
for i, (m, s) in enumerate(zip(stability_means, stability_stds)):
    ax_d.text(i, m + s + 0.02, f'{m:.3f}\n±{s:.3f}', ha='center', fontsize=7)

save_figure(fig, 'FigureS13_augmented_sentinel')
print("  FigureS13_augmented_sentinel saved.")

# ═══════════════════════════════════════════════════════════════
# 7. Save results
# ═══════════════════════════════════════════════════════════════
results_out = {
    'reference_strategies': {},
    'bootstrap_cis': {},
    'negative_control': {},
}

for rname, rdata in all_results.items():
    results_out['reference_strategies'][rname] = {
        'respiratory_sim': float(rdata['respiratory_sim']) if not np.isnan(rdata['respiratory_sim']) else None,
        'respiratory_rank': rdata['respiratory_rank'],
        'all_similarities': {c: float(s) if not np.isnan(s) else None
                             for c, s in rdata['similarities'].items()},
    }

for rname, ci_data in boot_ci_results.items():
    results_out['bootstrap_cis'][rname] = {
        comor: {k: float(v) if not np.isnan(v) else None for k, v in vals.items()}
        for comor, vals in ci_data.items()
    }

for rname, ndata in neg_ctrl_results.items():
    results_out['negative_control'][rname] = {
        'top_group': ndata['top_group'],
        'top_sim': float(ndata['top_sim']),
        'all_sims': {c: float(s) for c, s in ndata['all_sims'].items()},
    }

json_path = os.path.join(FIG_DIR, 'augmented_sentinel_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_out, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {json_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
