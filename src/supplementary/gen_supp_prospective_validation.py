"""
Supplementary Analysis: Prospective Validation Cohort (20K patients)
=====================================================================
Uses final_preprocessed_data_new.csv as an independent 20,169-patient
prospective cohort from a cardiac-specialty hospital (2011-2024).

Data availability:
  - 2012-2019 H1: pre-pandemic era (2019 data ends June)
  - 2020 May-Dec: post-initial-lockdown recovery
  - 2021-2024: continuous data including COVID reopening wave (2022 Q4)
  - COVID+ cases: 113 admissions from 18 patients (2022-2024)

Key validation objectives:
  1. Replicate sentinel population discovery with independent COVID+ cases
  2. Pre-pandemic (H1 2019) vs pandemic-wave (2022 Q4) comparison
  3. Monthly behavioral similarity profiling (2016-2023)
  4. RDI (Respiratory Dominance Index) temporal tracking
  5. Bootstrap confidence intervals for sentinel ranking

Generates:
  - FigureS14_prospective_validation (PNG/PDF/TIF)
  - prospective_validation_results.json

Run: python gen_supp_prospective_validation.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures'))
from fig_config import *

print("=" * 70)
print("Prospective Validation Cohort: 20K-Patient Sentinel Analysis")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("\n[1] Loading prospective validation data...")
VALIDATION_PATH = os.path.join(FIG_DIR, 'final_preprocessed_data_new.csv')
if not os.path.exists(VALIDATION_PATH):
    VALIDATION_PATH = r'D:\LDH_cancer\files\healthline\readmission_output\figures_v2\merge\final_preprocessed_data_new.csv'

td = pd.read_csv(VALIDATION_PATH, low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院时间'], errors='coerce')
td['discharge_dt'] = pd.to_datetime(td['出院时间'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month
td['quarter'] = td['admit_dt'].dt.quarter

# Compute LOS from admission/discharge dates
td['los_days'] = (td['discharge_dt'] - td['admit_dt']).dt.days

print(f"  Total admissions: {len(td):,}")
print(f"  Unique patients:  {td['病案号'].nunique():,}")
print(f"  Date range:       {td['admit_dt'].min().date()} to {td['admit_dt'].max().date()}")

# ═══════════════════════════════════════════════════════════════
# 2. Comorbidity extraction (adapted for cardiac hospital)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Extracting comorbidities from diagnosis text...")

COMORBIDITY_PATTERNS = {
    'Cardiovascular': r'冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病',
    'Hypertension':   r'高血压',
    'Diabetes':       r'糖尿病|血糖',
    'Cerebrovascular': r'脑梗|脑出血|脑血管|脑卒中|中风|腔隙性',
    'Renal':          r'肾功能|肾病|肾切除|肾衰|氮质血症|透析|肾脏',
    'Respiratory':    r'肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭|肺部感染',
}

diag_text = td['主要诊断'].fillna('').astype(str)
for name, pattern in COMORBIDITY_PATTERNS.items():
    td[name] = diag_text.str.contains(pattern, na=False).astype(int)
    print(f"    {name}: n={td[name].sum():,} ({td[name].mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# 3. Identify COVID+ patients (2022-2024 pandemic wave)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Identifying pandemic-positive patients...")

# Identify from primary diagnosis
covid_mask_primary = diag_text.str.contains(r'新型冠状病毒|冠状病毒感染|冠状病毒肺炎', na=False)

# Also check previous-visit diagnosis for COVID history
prev_diag = td['上次诊断'].fillna('').astype(str)
covid_mask_prev = prev_diag.str.contains(r'新型冠状病毒|冠状病毒感染|冠状病毒肺炎', na=False)

td['is_covid_positive'] = (covid_mask_primary | covid_mask_prev).astype(int)

covid_pos = td[td['is_covid_positive'] == 1]
n_covid_admissions = len(covid_pos)
n_covid_patients = covid_pos['病案号'].nunique()
print(f"  Pandemic-positive admissions: {n_covid_admissions}")
print(f"  Pandemic-positive patients:   {n_covid_patients}")
print(f"  Year distribution:")
print(f"    {covid_pos.groupby('year').size().to_dict()}")

# ═══════════════════════════════════════════════════════════════
# 4. Define behavioral metrics & z-score baseline
# ═══════════════════════════════════════════════════════════════
print("\n[4] Computing behavioral profiles...")

# Column mapping: new dataset → analysis framework
# The new dataset uses Chinese lab names directly (already z-scored or raw values)
BEHAVIOR_COLS = ['los_days']   # LOS computed from dates
LAB_COLS_MAP = {
    '白细胞':     'lab_WBC',
    '超敏C反应蛋白': 'lab_CRP',
    '血红蛋白':   'lab_HGB',
    '白蛋白':     'lab_ALB',
    '肌酐':       'lab_CREA',
    '空腹血糖':   'lab_GLU',
    '钾':         'lab_K',
    '钠':         'lab_Na',
}

# Rename lab columns to standardized names
for cn_name, en_name in LAB_COLS_MAP.items():
    if cn_name in td.columns:
        td[en_name] = pd.to_numeric(td[cn_name], errors='coerce')
    else:
        td[en_name] = np.nan

# Define the behavioral metric vector (aligned with original study)
all_metrics = ['los_days', 'lab_WBC', 'lab_CRP', 'lab_HGB', 'lab_ALB',
               'lab_CREA', 'lab_GLU', 'lab_K', 'lab_Na']

# Z-score baseline: 2016-2018 (pre-pandemic)
baseline = td[(td['year'] >= 2016) & (td['year'] <= 2018)]
ref_stats = {}
for col in all_metrics:
    vals = pd.to_numeric(baseline[col], errors='coerce').dropna()
    ref_stats[col] = {
        'mean': float(vals.mean()) if len(vals) > 0 else 0.0,
        'std': float(max(vals.std(), 1e-6)) if len(vals) > 0 else 1.0,
        'n': int(len(vals))
    }
    print(f"    {col}: mean={ref_stats[col]['mean']:.3f}, std={ref_stats[col]['std']:.3f}, n={ref_stats[col]['n']:,}")

# ─── Core functions ───
def zscore_profile(sub_df):
    """Compute mean z-score per metric for a patient subgroup."""
    p = {}
    for col in all_metrics:
        vals = pd.to_numeric(sub_df[col], errors='coerce').dropna()
        if len(vals) > 0:
            p[col] = (vals.mean() - ref_stats[col]['mean']) / ref_stats[col]['std']
        else:
            p[col] = 0.0
    return p

def profile_to_vec(prof):
    """Convert profile dict → numpy vector."""
    v = np.array([prof.get(k, 0) for k in all_metrics], dtype=float)
    return np.nan_to_num(v, 0)

def cos_sim(v1, v2):
    """Cosine similarity."""
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        return 1 - cosine(v1, v2)
    return 0.0

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']

# ═══════════════════════════════════════════════════════════════
# 5. Build pandemic-positive reference vector
# ═══════════════════════════════════════════════════════════════
print("\n[5] Building pandemic-positive reference vectors...")

# Primary reference: all COVID+ admissions (2022-2024)
covid_pos_all = td[td['is_covid_positive'] == 1]
vec_covid = profile_to_vec(zscore_profile(covid_pos_all))
print(f"  Primary COVID+ reference: n={len(covid_pos_all)} admissions")

# Alternative: COVID+ admissions from peak wave only (2022)
covid_2022 = td[(td['is_covid_positive'] == 1) & (td['year'] == 2022)]
vec_covid_2022 = profile_to_vec(zscore_profile(covid_2022)) if len(covid_2022) >= 5 else vec_covid
print(f"  2022 wave COVID+ reference: n={len(covid_2022)} admissions")

# Cross-reference: use respiratory flu-peak as augmented reference
flu_peak = td[(td['year'].isin([2017, 2018, 2019])) &
              (td['month'].isin([1, 2])) &
              (td['Respiratory'] == 1)]
print(f"  Respiratory flu-peak pool (2017-2019 Jan/Feb): n={len(flu_peak)}")

# Bootstrap-augmented reference (COVID+ combined with flu-peak respiratory)
TARGET_N = 100
rng = np.random.default_rng(42)
pool = pd.concat([covid_pos_all, flu_peak], ignore_index=True)
if len(pool) >= TARGET_N:
    boot_indices = rng.choice(len(pool), size=TARGET_N, replace=True)
else:
    boot_indices = rng.choice(len(pool), size=TARGET_N, replace=True)
augmented_pool = pool.iloc[boot_indices]
vec_augmented = profile_to_vec(zscore_profile(augmented_pool))
print(f"  Bootstrap-augmented reference: n={TARGET_N} (from pool of {len(pool)})")

# ═══════════════════════════════════════════════════════════════
# 6. Sentinel analysis: multiple time periods
# ═══════════════════════════════════════════════════════════════
print("\n[6] Sentinel population discovery...")
print("  Note: This cohort has data gap Jul 2019 – Apr 2020.")
print("        Using H1 2019 as pre-pandemic reference period.")

def sentinel_analysis(test_df, ref_vec, label=""):
    """Run sentinel analysis on test period data against reference."""
    sims = {}
    for comor in comorbidities:
        sub = test_df[test_df[comor] == 1]
        if len(sub) >= 5:
            v = profile_to_vec(zscore_profile(sub))
            sims[comor] = cos_sim(ref_vec, v)
        else:
            sims[comor] = np.nan
    valid_ranking = sorted([(c, s) for c, s in sims.items() if not np.isnan(s)],
                           key=lambda x: x[1], reverse=True)
    resp_rank = next((i + 1 for i, (c, _) in enumerate(valid_ranking) if c == 'Respiratory'),
                     len(valid_ranking) + 1)
    if label:
        print(f"\n  [{label}] (n={len(test_df):,})")
        for c, s in valid_ranking:
            marker = " ← SENTINEL" if c == 'Respiratory' else ""
            print(f"    {c:20s}: sim={s:.4f}{marker}")
        print(f"    Respiratory rank: #{resp_rank}")
    return sims, resp_rank

# Analysis A: Pre-pandemic H1 2019 (last pre-pandemic period with data)
print("\n  --- Analysis A: Pre-pandemic H1 2019 ---")
h1_2019 = td[(td['year'] == 2019) & (td['month'].isin([1, 2, 3, 4, 5, 6]))]
sims_h1_2019, rank_h1_2019 = sentinel_analysis(h1_2019, vec_covid, "H1 2019 vs COVID+ ref")

# Analysis B: Pre-pandemic H1 2018 (seasonal control)
print("\n  --- Analysis B: Pre-pandemic H1 2018 (seasonal control) ---")
h1_2018 = td[(td['year'] == 2018) & (td['month'].isin([1, 2, 3, 4, 5, 6]))]
sims_h1_2018, rank_h1_2018 = sentinel_analysis(h1_2018, vec_covid, "H1 2018 vs COVID+ ref")

# Analysis C: Post-lockdown recovery 2020 H2
print("\n  --- Analysis C: Post-lockdown 2020 H2 ---")
h2_2020 = td[(td['year'] == 2020) & (td['month'].isin([7, 8, 9, 10, 11, 12]))]
sims_h2_2020, rank_h2_2020 = sentinel_analysis(h2_2020, vec_covid, "H2 2020 vs COVID+ ref")

# Analysis D: Pandemic wave 2022 Q4 (China reopening, peak COVID+ cases)
print("\n  --- Analysis D: Pandemic wave 2022 Q4 ---")
q4_2022 = td[(td['year'] == 2022) & (td['month'].isin([10, 11, 12])) & (td['is_covid_positive'] == 0)]
sims_q4_2022, rank_q4_2022 = sentinel_analysis(q4_2022, vec_covid, "Q4 2022 non-COVID vs COVID+ ref")

# Analysis E: Full year 2022 (excluding COVID+ patients)
print("\n  --- Analysis E: Full year 2022 ---")
full_2022 = td[(td['year'] == 2022) & (td['is_covid_positive'] == 0)]
sims_2022, rank_2022 = sentinel_analysis(full_2022, vec_covid, "2022 non-COVID vs COVID+ ref")

# Analysis F: Post-pandemic 2023
print("\n  --- Analysis F: Post-pandemic 2023 ---")
post_2023 = td[td['year'] == 2023]
sims_2023, rank_2023 = sentinel_analysis(post_2023, vec_covid, "2023 vs COVID+ ref")

# Keep backward-compatible variable names for downstream code
sims_q4_2019 = sims_h1_2019   # best available pre-pandemic
rank_q4_2019 = rank_h1_2019
sims_2019 = sims_h1_2019

# ═══════════════════════════════════════════════════════════════
# 7. Monthly behavioral similarity tracking (2016-2023)
# ═══════════════════════════════════════════════════════════════
print("\n[7] Monthly behavioral similarity tracking...")

monthly_sims = {comor: {} for comor in comorbidities}
years_range = range(2016, 2024)

for year in years_range:
    for month in range(1, 13):
        period = td[(td['year'] == year) & (td['month'] == month)]
        if len(period) < 10:
            continue
        label = f"{year}-{month:02d}"
        for comor in comorbidities:
            sub = period[period[comor] == 1]
            if len(sub) >= 5:
                v = profile_to_vec(zscore_profile(sub))
                monthly_sims[comor][label] = cos_sim(vec_covid, v)

print(f"  Computed monthly similarities for {len(monthly_sims['Respiratory'])} month-periods")

# ═══════════════════════════════════════════════════════════════
# 8. RDI (Respiratory Dominance Index) quarterly tracking
# ═══════════════════════════════════════════════════════════════
print("\n[8] RDI quarterly tracking...")

rdi_records = []
for year in years_range:
    for quarter in range(1, 5):
        sub_q = td[(td['year'] == year) & (td['quarter'] == quarter)]
        if len(sub_q) < 10:
            continue
        sims_q = {}
        n_per_comor = {}
        for comor in comorbidities:
            csub = sub_q[sub_q[comor] == 1]
            n_per_comor[comor] = len(csub)
            if len(csub) >= 3:
                v = profile_to_vec(zscore_profile(csub))
                sims_q[comor] = cos_sim(vec_covid, v)

        resp_sim = sims_q.get('Respiratory', np.nan)
        other_sims = [s for c, s in sims_q.items() if c != 'Respiratory' and not np.isnan(s)]
        mean_other = np.mean(other_sims) if other_sims else np.nan
        rdi = resp_sim - mean_other if not np.isnan(resp_sim) and not np.isnan(mean_other) else np.nan

        rdi_records.append({
            'year': year, 'quarter': quarter,
            'label': f"{year}Q{quarter}",
            'n_admissions': len(sub_q),
            'resp_sim': resp_sim,
            'mean_other': mean_other,
            'rdi': rdi,
            **{f'sim_{c}': sims_q.get(c, np.nan) for c in comorbidities},
            **{f'n_{c}': n_per_comor.get(c, 0) for c in comorbidities},
        })

df_rdi = pd.DataFrame(rdi_records)
print(f"  RDI records: {len(df_rdi)}")
print(f"\n  RDI summary (key quarters):")
for _, row in df_rdi.iterrows():
    if row['year'] in [2019, 2020, 2022, 2023] and not np.isnan(row['rdi']):
        print(f"    {row['label']}: RDI={row['rdi']:.3f}, Resp sim={row['resp_sim']:.3f}, "
              f"Other mean={row['mean_other']:.3f}, n={int(row['n_admissions'])}")

# ═══════════════════════════════════════════════════════════════
# 9. Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════
print("\n[9] Bootstrap confidence intervals for sentinel ranking...")

N_BOOT = 2000
N_PERM = 5000  # define early for permutation test
rng = np.random.default_rng(42)

# Test on H1 2019 (best available pre-pandemic period)
boot_results = {comor: [] for comor in comorbidities}
h1_data = td[(td['year'] == 2019) & (td['month'].isin([1, 2, 3, 4, 5, 6]))]

for _ in range(N_BOOT):
    for comor in comorbidities:
        sub = h1_data[h1_data[comor] == 1]
        if len(sub) >= 5:
            idx = rng.choice(len(sub), size=len(sub), replace=True)
            boot_sub = sub.iloc[idx]
            v = profile_to_vec(zscore_profile(boot_sub))
            boot_results[comor].append(cos_sim(vec_covid, v))
        else:
            boot_results[comor].append(np.nan)

boot_ci = {}
for comor in comorbidities:
    vals = [x for x in boot_results[comor] if not np.isnan(x)]
    if vals:
        boot_ci[comor] = {
            'mean': float(np.mean(vals)),
            'ci_lo': float(np.percentile(vals, 2.5)),
            'ci_hi': float(np.percentile(vals, 97.5)),
            'std': float(np.std(vals)),
        }
        print(f"    {comor:20s}: {boot_ci[comor]['mean']:.3f} "
              f"[{boot_ci[comor]['ci_lo']:.3f}, {boot_ci[comor]['ci_hi']:.3f}]")

# ═══════════════════════════════════════════════════════════════
# 10. Permutation test: H1 2019 vs H1 2018
# ═══════════════════════════════════════════════════════════════
print("\n[10] Permutation test: H1 2019 vs H1 2018 respiratory similarity...")

resp_h1_2019 = h1_2019[h1_2019['Respiratory'] == 1]
resp_h1_2018 = h1_2018[h1_2018['Respiratory'] == 1]

if len(resp_h1_2019) >= 5 and len(resp_h1_2018) >= 5:
    sim_period_a = cos_sim(vec_covid, profile_to_vec(zscore_profile(resp_h1_2019)))
    sim_period_b = cos_sim(vec_covid, profile_to_vec(zscore_profile(resp_h1_2018)))
    observed_diff = sim_period_a - sim_period_b

    combined = pd.concat([resp_h1_2019, resp_h1_2018], ignore_index=True)
    n_a = len(resp_h1_2019)
    perm_diffs = []
    for _ in range(N_PERM):
        idx = rng.permutation(len(combined))
        perm_a = combined.iloc[idx[:n_a]]
        perm_b = combined.iloc[idx[n_a:]]
        va = profile_to_vec(zscore_profile(perm_a))
        vb = profile_to_vec(zscore_profile(perm_b))
        perm_diffs.append(cos_sim(vec_covid, va) - cos_sim(vec_covid, vb))

    perm_p = np.mean([abs(d) >= abs(observed_diff) for d in perm_diffs])
    print(f"  Respiratory sim H1 2019: {sim_period_a:.4f}")
    print(f"  Respiratory sim H1 2018: {sim_period_b:.4f}")
    print(f"  Observed difference:     {observed_diff:.4f}")
    print(f"  Permutation p-value:     {perm_p:.4f} (n_perm={N_PERM})")
else:
    sim_period_a = cos_sim(vec_covid, profile_to_vec(zscore_profile(resp_h1_2019))) if len(resp_h1_2019) >= 5 else np.nan
    sim_period_b = cos_sim(vec_covid, profile_to_vec(zscore_profile(resp_h1_2018))) if len(resp_h1_2018) >= 5 else np.nan
    observed_diff = np.nan
    perm_p = np.nan
    print(f"  Respiratory H1 2019 n={len(resp_h1_2019)}, H1 2018 n={len(resp_h1_2018)}")
    print("  Insufficient respiratory samples for permutation test")

# ═══════════════════════════════════════════════════════════════
# 11. Cross-reference with original WHU results
# ═══════════════════════════════════════════════════════════════
print("\n[11] Cross-reference with original study results...")

# Load original augmented sentinel results if available
orig_results_path = os.path.join(FIG_DIR, 'augmented_sentinel_results.json')
orig_results = None
if os.path.exists(orig_results_path):
    with open(orig_results_path, 'r') as f:
        orig_results = json.load(f)
    print("  Loaded original WHU sentinel results for comparison")
    if 'reference_strategies' in orig_results:
        orig_resp = orig_results['reference_strategies'].get(
            'Original COVID+ (n=19)', {}).get('respiratory_sim', None)
        if orig_resp is not None:
            val_resp = sims_h1_2019.get('Respiratory', np.nan)
            print(f"  WHU resp similarity (Q4 2019):        {orig_resp:.4f}")
            print(f"  Validation resp similarity (H1 2019): {val_resp:.4f}" if not np.isnan(val_resp)
                  else "  Validation resp similarity (H1 2019): N/A")

# ═══════════════════════════════════════════════════════════════
# 12. Generate Figure S14: Prospective Validation (2×3 layout)
# ═════════════════════════════════════════════════════════════
print("\n[12] Generating Figure S8...")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                       left=0.07, right=0.95, top=0.95, bottom=0.05)

# ─── Panel A: Sentinel bar chart (H1 2019) ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

comor_order = sorted(sims_h1_2019.keys(),
                     key=lambda c: sims_h1_2019.get(c, 0) if not np.isnan(sims_h1_2019.get(c, 0)) else -999,
                     reverse=True)
# Filter out NaN comorbidities
comor_order = [c for c in comor_order if not np.isnan(sims_h1_2019.get(c, np.nan))]
bar_vals = [sims_h1_2019.get(c, 0) for c in comor_order]
bar_colors = [COLORS['secondary'] if c == 'Respiratory' else COLORS['primary']
              for c in comor_order]

# Add CI error bars
bar_errors = []
for c in comor_order:
    if c in boot_ci:
        bar_errors.append([
            boot_ci[c]['mean'] - boot_ci[c]['ci_lo'],
            boot_ci[c]['ci_hi'] - boot_ci[c]['mean']
        ])
    else:
        bar_errors.append([0, 0])
bar_errors = np.array(bar_errors).T

bars = ax_a.bar(range(len(comor_order)), bar_vals, color=bar_colors, edgecolor='white',
                linewidth=0.5, yerr=bar_errors, capsize=3, error_kw={'linewidth': 0.8})
ax_a.set_xticks(range(len(comor_order)))
ax_a.set_xticklabels([c[:12] for c in comor_order], rotation=30, ha='right')
ax_a.set_ylabel('Cosine Similarity to COVID+')
ax_a.set_title('Sentinel Discovery (H1 2019)', fontweight='bold')
ax_a.axhline(y=0.5, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

# ─── Panel B: Multi-period comparison ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

periods = {
    'H1 2019\n(Pre-pandemic)': sims_h1_2019,
    'H2 2020\n(Post-lockdown)': sims_h2_2020,
    'Q4 2022\n(Reopening wave)': sims_q4_2022,
    '2023\n(Post-pandemic)': sims_2023,
}

x_pos = np.arange(len(comorbidities))
width = 0.18
period_colors = [PAL[0], PAL[1], PAL[2], PAL[3]]

for i, (period_name, period_sims) in enumerate(periods.items()):
    vals = [period_sims.get(c, 0) for c in comorbidities]
    ax_b.bar(x_pos + i * width, vals, width, label=period_name,
             color=period_colors[i], edgecolor='white', linewidth=0.3)

ax_b.set_xticks(x_pos + width * 1.5)
ax_b.set_xticklabels([c[:10] for c in comorbidities], rotation=30, ha='right')
ax_b.set_ylabel('Cosine Similarity')
ax_b.set_title('Cross-Period Sentinel Consistency', fontweight='bold')
ax_b.legend(fontsize=7, loc='upper right')

# ─── Panel C: Monthly respiratory similarity timeline ───
ax_c = fig.add_subplot(gs[1, :])
panel_label(ax_c, 'C')

resp_monthly = monthly_sims.get('Respiratory', {})
months_sorted = sorted(resp_monthly.keys())
resp_vals = [resp_monthly[m] for m in months_sorted]

# Also plot cardiovascular as contrast
cv_monthly = monthly_sims.get('Cardiovascular', {})
cv_months = sorted(cv_monthly.keys())
cv_vals = [cv_monthly[m] for m in cv_months]

ax_c.plot(range(len(months_sorted)), resp_vals, 'o-',
          color=COLORS['secondary'], markersize=3, linewidth=1.2,
          label='Respiratory')
if cv_vals:
    # Align cardiovascular to same x-axis
    cv_x = [months_sorted.index(m) for m in cv_months if m in months_sorted]
    cv_y = [cv_monthly[m] for m in cv_months if m in months_sorted]
    ax_c.plot(cv_x, cv_y, 's-',
              color=COLORS['primary'], markersize=2, linewidth=0.8,
              alpha=0.7, label='Cardiovascular')

# Mark key events
for i, m in enumerate(months_sorted):
    if m == '2020-01':
        ax_c.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_c.text(i, ax_c.get_ylim()[1] * 0.95, 'COVID-19\nonset',
                  fontsize=7, ha='center', color='red')
    elif m == '2022-12':
        ax_c.axvline(x=i, color='orange', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_c.text(i, ax_c.get_ylim()[1] * 0.90, 'Reopening\nwave',
                  fontsize=7, ha='center', color='orange')

# Reduce x-tick density
tick_step = max(1, len(months_sorted) // 20)
ax_c.set_xticks(range(0, len(months_sorted), tick_step))
ax_c.set_xticklabels([months_sorted[i] for i in range(0, len(months_sorted), tick_step)],
                      rotation=45, ha='right', fontsize=7)
ax_c.set_ylabel('Cosine Similarity to COVID+')
ax_c.set_title('Monthly Respiratory Behavioral Similarity (2016-2023)', fontweight='bold')
ax_c.legend(fontsize=8)
ax_c.axhline(y=0.75, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

# ─── Panel D: RDI quarterly tracking ───
ax_d = fig.add_subplot(gs[2, 0])
panel_label(ax_d, 'D')

rdi_vals = df_rdi['rdi'].values
rdi_labels = df_rdi['label'].values
valid_mask = ~np.isnan(rdi_vals)

ax_d.bar(np.where(valid_mask)[0], rdi_vals[valid_mask],
         color=[COLORS['secondary'] if r > 0.2 else COLORS['primary'] for r in rdi_vals[valid_mask]],
         edgecolor='white', linewidth=0.3)
tick_step_rdi = max(1, len(rdi_labels) // 12)
ax_d.set_xticks(range(0, len(rdi_labels), tick_step_rdi))
ax_d.set_xticklabels(rdi_labels[::tick_step_rdi], rotation=45, ha='right', fontsize=7)
ax_d.set_ylabel('RDI')
ax_d.set_title('Respiratory Dominance Index (Quarterly)', fontweight='bold')
ax_d.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

# ─── Panel E: WHU vs Validation comparison ───
ax_e = fig.add_subplot(gs[2, 1])
panel_label(ax_e, 'E')

# Compare validation sentinel sims with original WHU results
val_sims_list = [sims_h1_2019.get(c, 0) if not np.isnan(sims_h1_2019.get(c, 0)) else 0 for c in comorbidities]
if orig_results and 'reference_strategies' in orig_results:
    orig_all = orig_results['reference_strategies'].get(
        'Original COVID+ (n=19)', {}).get('all_similarities', {})
    whu_sims_list = [orig_all.get(c, 0) for c in comorbidities]
else:
    whu_sims_list = [0] * len(comorbidities)

x_pos2 = np.arange(len(comorbidities))
width2 = 0.35
ax_e.bar(x_pos2 - width2/2, whu_sims_list, width2,
         label='WHU (Original)', color=COLORS['primary'], edgecolor='white', linewidth=0.3)
ax_e.bar(x_pos2 + width2/2, val_sims_list, width2,
         label='Validation (20K)', color=COLORS['secondary'], edgecolor='white', linewidth=0.3)
ax_e.set_xticks(x_pos2)
ax_e.set_xticklabels([c[:10] for c in comorbidities], rotation=30, ha='right')
ax_e.set_ylabel('Cosine Similarity')
ax_e.set_title('WHU vs Validation Cohort', fontweight='bold')
ax_e.legend(fontsize=8)

# ─── Save figure ───
save_figure(fig, 'Figure5_event_driven_sentinel')

# ═══════════════════════════════════════════════════════════════
# 13. Save results JSON
# ═══════════════════════════════════════════════════════════════
print("\n[13] Saving results...")

output = {
    'cohort_summary': {
        'total_admissions': int(len(td)),
        'unique_patients': int(td['病案号'].nunique()),
        'date_range': f"{td['admit_dt'].min().date()} to {td['admit_dt'].max().date()}",
        'covid_positive_admissions': int(n_covid_admissions),
        'covid_positive_patients': int(n_covid_patients),
        'comorbidity_counts': {c: int(td[c].sum()) for c in comorbidities},
    },
    'baseline_stats': {col: {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                             for k, v in stats_dict.items()}
                       for col, stats_dict in ref_stats.items()},
    'sentinel_analysis': {
        'H1_2019_prepandemic': {
            'similarities': {k: float(v) if not np.isnan(v) else None for k, v in sims_h1_2019.items()},
            'respiratory_rank': int(rank_h1_2019),
        },
        'H1_2018_seasonal_ctrl': {
            'similarities': {k: float(v) if not np.isnan(v) else None for k, v in sims_h1_2018.items()},
            'respiratory_rank': int(rank_h1_2018),
        },
        'H2_2020_postlockdown': {
            'similarities': {k: float(v) if not np.isnan(v) else None for k, v in sims_h2_2020.items()},
            'respiratory_rank': int(rank_h2_2020),
        },
        'Q4_2022_reopening': {
            'similarities': {k: float(v) if not np.isnan(v) else None for k, v in sims_q4_2022.items()},
            'respiratory_rank': int(rank_q4_2022),
        },
        'Full_2022': {
            'similarities': {k: float(v) if not np.isnan(v) else None for k, v in sims_2022.items()},
            'respiratory_rank': int(rank_2022),
        },
        'Post_2023': {
            'similarities': {k: float(v) if not np.isnan(v) else None for k, v in sims_2023.items()},
            'respiratory_rank': int(rank_2023),
        },
    },
    'bootstrap_ci': {k: v for k, v in boot_ci.items()},
    'permutation_test': {
        'sim_h1_2019': float(sim_period_a) if not np.isnan(sim_period_a) else None,
        'sim_h1_2018': float(sim_period_b) if not np.isnan(sim_period_b) else None,
        'observed_diff': float(observed_diff) if not np.isnan(observed_diff) else None,
        'p_value': float(perm_p) if not np.isnan(perm_p) else None,
        'n_permutations': N_PERM,
    },
    'rdi_quarterly': df_rdi.replace({np.nan: None}).to_dict(orient='records'),
    'monthly_respiratory_similarity': {m: float(v) for m, v in sorted(resp_monthly.items())},
}

results_path = os.path.join(FIG_DIR, 'prospective_validation_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {results_path}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROSPECTIVE VALIDATION SUMMARY")
print("=" * 70)
print(f"  Cohort: {td['病案号'].nunique():,} patients, {len(td):,} admissions (2011-2024)")
print(f"  COVID+ reference: {n_covid_admissions} admissions from {n_covid_patients} patients (2022-2024)")
print(f"  Data gap: Jul 2019 – Apr 2020 (using H1 2019 as pre-pandemic period)")
print(f"\n  Sentinel Analysis (H1 2019, pre-pandemic):")
for c in comor_order:
    v = sims_h1_2019.get(c, 0)
    ci = boot_ci.get(c, {})
    ci_str = f" [{ci.get('ci_lo', 0):.3f}, {ci.get('ci_hi', 0):.3f}]" if ci else ""
    marker = " *** SENTINEL" if c == 'Respiratory' else ""
    print(f"    {c:20s}: {v:.4f}{ci_str}{marker}")
print(f"  Respiratory rank: #{rank_h1_2019}")
print(f"\n  Sentinel Analysis (Q4 2022, reopening wave):")
q4_order = sorted(sims_q4_2022.keys(),
                  key=lambda c: sims_q4_2022.get(c, 0) if not np.isnan(sims_q4_2022.get(c, 0)) else -999,
                  reverse=True)
for c in q4_order:
    v = sims_q4_2022.get(c, np.nan)
    if not np.isnan(v):
        marker = " *** SENTINEL" if c == 'Respiratory' else ""
        print(f"    {c:20s}: {v:.4f}{marker}")
print(f"  Respiratory rank: #{rank_q4_2022}")
print(f"\n  Permutation test (H1 2019 vs H1 2018):")
print(f"    p = {perm_p:.4f}" if not np.isnan(perm_p) else "    p = N/A")
rdi_valid = df_rdi.dropna(subset=['rdi'])
if len(rdi_valid) > 0:
    peak_idx = rdi_valid['rdi'].idxmax()
    print(f"\n  RDI peak: {rdi_valid.loc[peak_idx, 'label']} (RDI = {rdi_valid.loc[peak_idx, 'rdi']:.3f})")
print("=" * 70)
print("Done!")
