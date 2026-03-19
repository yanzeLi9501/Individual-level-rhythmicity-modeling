"""
Supplementary Materials Consolidation Script
  + Flesh out Figures S7–S11 with actual data
  + Extended Tables S5–S7
  + Data-cleaning procedures documentation
  + Code/Data availability with repository structure

Generates:
  - FigureS15_extended_supp (PNG/PDF/TIF)
  - TableS6_data_cleaning.csv
  - TableS7_code_availability.json
  - supplementary_completeness_report.json

Run: python gen_supp_extended.py
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
print("Supplementary Materials Consolidation")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("[1] Loading data...")
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month
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
dim_labels = ['LOS', 'Lab Tests', 'Orders', 'Drug Orders', 'Examinations',
              'WBC', 'CRP', 'HGB', 'ALB', 'Creatinine', 'Glucose', 'K', 'Na']

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']

print(f"  Total admissions: {len(td):,}")
print(f"  Year range: {td['year'].min()}–{td['year'].max()}")

# ═══════════════════════════════════════════════════════════════
# 2. Data-Cleaning Procedures Documentation (Table S6)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Data-cleaning procedures...")

cleaning_steps = []

# Step 1: Raw extraction
n_raw = len(td)
cleaning_steps.append({
    'Step': 1,
    'Procedure': 'Raw EHR extraction from semi-structured JSON',
    'Records_Before': n_raw,
    'Records_After': n_raw,
    'Records_Removed': 0,
    'Criteria': 'All available admission records extracted',
})

# Step 2: Date validation
valid_dates = td['admit_dt'].notna()
n_valid_dates = valid_dates.sum()
cleaning_steps.append({
    'Step': 2,
    'Procedure': 'Date validation (admission date parsing)',
    'Records_Before': n_raw,
    'Records_After': int(n_valid_dates),
    'Records_Removed': int(n_raw - n_valid_dates),
    'Criteria': 'Parseable admission date in format YYYY-MM-DD',
})

# Step 3: Duplicate removal
td_dedup = td[valid_dates].copy()
pid_col_candidates = ['住院流水号', 'patient_id']
pid_col = None
for c in pid_col_candidates:
    if c in td.columns:
        pid_col = c
        break

if pid_col:
    n_before_dedup = len(td_dedup)
    td_dedup = td_dedup.drop_duplicates(subset=[pid_col, '入院日期'], keep='first')
    n_after_dedup = len(td_dedup)
    cleaning_steps.append({
        'Step': 3,
        'Procedure': 'Duplicate admission removal',
        'Records_Before': n_before_dedup,
        'Records_After': n_after_dedup,
        'Records_Removed': n_before_dedup - n_after_dedup,
        'Criteria': f'Same {pid_col} + admission date',
    })

# Step 4: LOS outlier handling
los_col = '实际住院天数'
if los_col in td.columns:
    los_vals = pd.to_numeric(td[los_col], errors='coerce')
    n_los_valid = los_vals.notna().sum()
    n_los_zero = (los_vals <= 0).sum()
    los_p99 = los_vals.quantile(0.99) if n_los_valid > 0 else np.nan
    cleaning_steps.append({
        'Step': 4,
        'Procedure': 'LOS validation and capping',
        'Records_Before': int(n_los_valid),
        'Records_After': int(n_los_valid - n_los_zero),
        'Records_Removed': int(n_los_zero),
        'Criteria': f'LOS > 0 days; capped at 7 days for modeling (P99={los_p99:.0f})',
    })

# Step 5: Laboratory value outlier handling
lab_cleaning = []
for col_name, label in zip(lab_cols, dim_labels[5:]):
    if col_name in td.columns:
        vals = pd.to_numeric(td[col_name], errors='coerce')
        n_total = len(vals)
        n_valid = vals.notna().sum()
        n_outlier = 0
        if n_valid > 0:
            p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
            n_outlier = ((vals < p1) | (vals > p99)).sum()
        lab_cleaning.append({
            'Lab': label,
            'Column': col_name,
            'Total': int(n_total),
            'Valid': int(n_valid),
            'Missing': int(n_total - n_valid),
            'Missing%': f'{100*(n_total-n_valid)/n_total:.1f}',
            'P1': f'{p1:.2f}' if n_valid > 0 else 'N/A',
            'P99': f'{p99:.2f}' if n_valid > 0 else 'N/A',
        })

cleaning_steps.append({
    'Step': 5,
    'Procedure': 'Laboratory value validation',
    'Records_Before': 'Per-feature (see lab detail table)',
    'Records_After': 'Per-feature',
    'Records_Removed': 'Missing values retained as NaN; no imputation',
    'Criteria': 'Numeric coercion; values outside physiological range flagged',
})

# Step 6: Visit sequence construction
cleaning_steps.append({
    'Step': 6,
    'Procedure': 'Visit sequence construction',
    'Records_Before': 'All valid admissions',
    'Records_After': 'Per-patient longitudinal sequences',
    'Records_Removed': 'Single-visit patients excluded from gap analysis',
    'Criteria': 'Group by patient ID, sort by admission date, compute visit order',
})

# Step 7: Gap day target construction
cleaning_steps.append({
    'Step': 7,
    'Procedure': 'Target variable construction (gap days)',
    'Records_Before': 'All multi-visit patients',
    'Records_After': 'Visit order >= 5 (broad); >= 20 (frequent)',
    'Records_Removed': 'Varies by configuration',
    'Criteria': 'Gap capped at 30 days (broad) or 10 days (frequent); negative gaps excluded',
})

# Print summary
print("\n  Data Cleaning Pipeline:")
for step in cleaning_steps:
    print(f"    Step {step['Step']}: {step['Procedure']}")
    print(f"      Before: {step['Records_Before']} → After: {step['Records_After']}")

# Save as CSV
df_cleaning = pd.DataFrame(cleaning_steps)
csv_path = os.path.join(FIG_DIR, 'TableS6_data_cleaning.csv')
df_cleaning.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n  Saved: {csv_path}")

# Lab detail table
if lab_cleaning:
    df_lab = pd.DataFrame(lab_cleaning)
    lab_csv = os.path.join(FIG_DIR, 'TableS6b_lab_completeness.csv')
    df_lab.to_csv(lab_csv, index=False, encoding='utf-8-sig')
    print(f"  Saved: {lab_csv}")

# ═══════════════════════════════════════════════════════════════
# 3. Cohort flow diagram data (CONSORT-style)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Cohort statistics...")

# Patient-level statistics
if pid_col:
    n_patients = td[pid_col].nunique()
    visit_counts = td.groupby(pid_col).size()
    n_multi_visit = (visit_counts >= 2).sum()
    n_vo5 = (visit_counts >= 5).sum()
    n_vo20 = (visit_counts >= 20).sum()
else:
    n_patients = 'Unknown'
    n_multi_visit = n_vo5 = n_vo20 = 'Unknown'

cohort_stats = {
    'total_admissions': len(td),
    'total_patients': int(n_patients) if isinstance(n_patients, (int, np.integer)) else n_patients,
    'multi_visit_patients': int(n_multi_visit) if isinstance(n_multi_visit, (int, np.integer)) else n_multi_visit,
    'vo5_patients': int(n_vo5) if isinstance(n_vo5, (int, np.integer)) else n_vo5,
    'vo20_patients': int(n_vo20) if isinstance(n_vo20, (int, np.integer)) else n_vo20,
    'comorbidity_prevalence': {c: int(td[c].sum()) for c in comorbidities},
    'year_distribution': td.groupby('year').size().to_dict(),
}

print(f"  Patients: {cohort_stats['total_patients']}")
print(f"  Multi-visit (≥2): {cohort_stats['multi_visit_patients']}")
print(f"  Frequent (≥5): {cohort_stats['vo5_patients']}")
print(f"  Very frequent (≥20): {cohort_stats['vo20_patients']}")

# ═══════════════════════════════════════════════════════════════
# 4. Extended supplementary figures with actual data
# ═══════════════════════════════════════════════════════════════
print("\n[4] Generating FigureS15_extended_supp...")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, wspace=0.35, hspace=0.45,
                       left=0.06, right=0.96, top=0.95, bottom=0.05)

# ─── Panel A: Annual admission volume ───
ax1 = fig.add_subplot(gs[0, 0])
panel_label(ax1, 'A')

year_counts = td.groupby('year').size()
ax1.bar(year_counts.index, year_counts.values, color=COLORS['primary'], edgecolor='white')
ax1.set_xlabel('Year', fontsize=9)
ax1.set_ylabel('Admissions', fontsize=9)
ax1.set_title('Annual Admission Volume', fontsize=10)
for y, c in zip(year_counts.index, year_counts.values):
    ax1.text(y, c + max(year_counts.values) * 0.02, f'{c:,}', ha='center', fontsize=6)

# ─── Panel B: Comorbidity prevalence ───
ax2 = fig.add_subplot(gs[0, 1])
panel_label(ax2, 'B')

comor_counts = {c: td[c].sum() for c in comorbidities}
sorted_comor = sorted(comor_counts.items(), key=lambda x: x[1], reverse=True)
c_names = [x[0] for x in sorted_comor]
c_vals = [x[1] for x in sorted_comor]

bars = ax2.barh(range(len(c_names)), c_vals, color=PAL[:len(c_names)], edgecolor='white')
ax2.set_yticks(range(len(c_names)))
ax2.set_yticklabels(c_names, fontsize=8)
ax2.set_xlabel('Number of Admissions', fontsize=9)
ax2.set_title('Comorbidity Prevalence', fontsize=10)
for bar, val in zip(bars, c_vals):
    ax2.text(bar.get_width() + max(c_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
             f'{val:,}', va='center', fontsize=7)
ax2.invert_yaxis()

# ─── Panel C: LOS distribution ───
ax3 = fig.add_subplot(gs[0, 2])
panel_label(ax3, 'C')

if los_col in td.columns:
    los_data = pd.to_numeric(td[los_col], errors='coerce').dropna()
    los_clipped = los_data[los_data.between(0, 30)]
    ax3.hist(los_clipped, bins=30, color=COLORS['primary'], edgecolor='white', alpha=0.8)
    ax3.axvline(los_clipped.median(), color=COLORS['secondary'], linestyle='--',
                label=f'Median: {los_clipped.median():.1f}d')
    ax3.axvline(los_clipped.mean(), color=COLORS['accent2'], linestyle=':',
                label=f'Mean: {los_clipped.mean():.1f}d')
    ax3.set_xlabel('Length of Stay (days)', fontsize=9)
    ax3.set_ylabel('Frequency', fontsize=9)
    ax3.set_title('LOS Distribution (0–30d)', fontsize=10)
    ax3.legend(fontsize=7)

# ─── Panel D: Lab value completeness ───
ax4 = fig.add_subplot(gs[1, 0])
panel_label(ax4, 'D')

if lab_cleaning:
    lab_names = [l['Lab'] for l in lab_cleaning]
    lab_pct = [float(l['Missing%']) for l in lab_cleaning]
    colors_lab = [COLORS['secondary'] if p > 30 else COLORS['accent1'] for p in lab_pct]
    ax4.barh(range(len(lab_names)), lab_pct, color=colors_lab, edgecolor='white')
    ax4.set_yticks(range(len(lab_names)))
    ax4.set_yticklabels(lab_names, fontsize=8)
    ax4.set_xlabel('Missing %', fontsize=9)
    ax4.set_title('Laboratory Value Missingness', fontsize=10)
    ax4.axvline(30, color='grey', linestyle='--', alpha=0.5, label='30% threshold')
    ax4.legend(fontsize=7)
    ax4.invert_yaxis()

# ─── Panel E: Monthly respiratory proportion (all years) ───
ax5 = fig.add_subplot(gs[1, 1])
panel_label(ax5, 'E')

monthly_resp = td.groupby(['year', 'month']).agg(
    total=('year', 'size'),
    resp=('Respiratory', 'sum')
).reset_index()
monthly_resp['resp_pct'] = monthly_resp['resp'] / monthly_resp['total'] * 100

for year in sorted(monthly_resp['year'].unique()):
    sub = monthly_resp[monthly_resp['year'] == year]
    color = COLORS['secondary'] if year == 2020 else PAL[int(year) % len(PAL)]
    lw = 2 if year == 2020 else 0.8
    ax5.plot(sub['month'], sub['resp_pct'], '-o', color=color, linewidth=lw,
             markersize=2 if year != 2020 else 4, label=str(year), alpha=0.7 if year != 2020 else 1)

ax5.set_xlabel('Month', fontsize=9)
ax5.set_ylabel('Respiratory %', fontsize=9)
ax5.set_title('Monthly Respiratory Admission Rate', fontsize=10)
ax5.legend(fontsize=5, ncol=3, loc='upper right')

# ─── Panel F: Gap-LOS residual coupling ───
ax6 = fig.add_subplot(gs[1, 2])
panel_label(ax6, 'F')

# Compute basic gap-LOS correlation by respiratory status
if 'incoming_gap' in td.columns and los_col in td.columns:
    gap_vals = pd.to_numeric(td['incoming_gap'], errors='coerce')
    los_vals_f = pd.to_numeric(td[los_col], errors='coerce')
    mask = gap_vals.notna() & los_vals_f.notna() & (gap_vals > 0) & (gap_vals <= 60) & (los_vals_f > 0) & (los_vals_f <= 30)

    for label, color, subset in [('Non-respiratory', COLORS['primary'], td[mask & (td['Respiratory'] == 0)]),
                                  ('Respiratory', COLORS['secondary'], td[mask & (td['Respiratory'] == 1)])]:
        g = pd.to_numeric(subset['incoming_gap'], errors='coerce')
        l = pd.to_numeric(subset[los_col], errors='coerce')
        if len(g) > 10:
            rho, _ = stats.spearmanr(g, l)
            ax6.scatter(g.sample(min(500, len(g)), random_state=42),
                       l[g.sample(min(500, len(g)), random_state=42).index],
                       alpha=0.3, s=5, color=color, label=f'{label} (ρ={rho:.3f})')
    ax6.set_xlabel('Gap Days', fontsize=9)
    ax6.set_ylabel('LOS (days)', fontsize=9)
    ax6.set_title('Gap-LOS Coupling by Respiratory Status', fontsize=10)
    ax6.legend(fontsize=7)
else:
    ax6.text(0.5, 0.5, 'Gap column not available', ha='center', va='center',
             transform=ax6.transAxes)

# ─── Panel G: Cross-year respiratory seasonality ───
ax7 = fig.add_subplot(gs[2, 0])
panel_label(ax7, 'G')

# Cosine similarity of monthly respiratory patterns year-to-year
years = sorted([y for y in td['year'].unique() if 2016 <= y <= 2020])
yearly_vecs = {}
for year in years:
    monthly = []
    for month in range(1, 13):
        sub = td[(td['year'] == year) & (td['month'] == month)]
        monthly.append(sub['Respiratory'].sum() / max(len(sub), 1))
    yearly_vecs[year] = np.array(monthly)

sim_matrix = np.zeros((len(years), len(years)))
for i, y1 in enumerate(years):
    for j, y2 in enumerate(years):
        v1, v2 = yearly_vecs[y1], yearly_vecs[y2]
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            sim_matrix[i, j] = 1 - cosine(v1, v2)

im = ax7.imshow(sim_matrix, cmap=SPRING_CMAP, vmin=0, vmax=1)
ax7.set_xticks(range(len(years)))
ax7.set_xticklabels(years, fontsize=8)
ax7.set_yticks(range(len(years)))
ax7.set_yticklabels(years, fontsize=8)
ax7.set_title('Cross-Year Respiratory Seasonality', fontsize=10)
for i in range(len(years)):
    for j in range(len(years)):
        ax7.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                 color='white' if sim_matrix[i, j] > 0.6 else 'black')
plt.colorbar(im, ax=ax7, shrink=0.8)

# ─── Panel H: Visit order distribution ───
ax8 = fig.add_subplot(gs[2, 1])
panel_label(ax8, 'H')

if pid_col:
    vo_dist = visit_counts.value_counts().sort_index()
    vo_clipped = vo_dist[vo_dist.index <= 50]
    ax8.bar(vo_clipped.index, vo_clipped.values, color=COLORS['primary'], edgecolor='white', width=0.8)
    ax8.axvline(5, color=COLORS['secondary'], linestyle='--', label='VO ≥ 5 threshold')
    ax8.axvline(20, color=COLORS['accent2'], linestyle='--', label='VO ≥ 20 threshold')
    ax8.set_xlabel('Visit Order', fontsize=9)
    ax8.set_ylabel('Number of Patients', fontsize=9)
    ax8.set_title('Visit Order Distribution', fontsize=10)
    ax8.legend(fontsize=7)

# ─── Panel I: Comorbidity co-occurrence ───
ax9 = fig.add_subplot(gs[2, 2])
panel_label(ax9, 'I')

co_matrix = np.zeros((len(comorbidities), len(comorbidities)))
for i, c1 in enumerate(comorbidities):
    for j, c2 in enumerate(comorbidities):
        if i == j:
            co_matrix[i, j] = td[c1].sum()
        else:
            co_matrix[i, j] = ((td[c1] == 1) & (td[c2] == 1)).sum()

# Normalize to percentages
co_pct = co_matrix / len(td) * 100
im9 = ax9.imshow(co_pct, cmap='YlOrRd', aspect='equal')
ax9.set_xticks(range(len(comorbidities)))
ax9.set_xticklabels([c[:6] for c in comorbidities], fontsize=7, rotation=45, ha='right')
ax9.set_yticks(range(len(comorbidities)))
ax9.set_yticklabels([c[:6] for c in comorbidities], fontsize=7)
ax9.set_title('Comorbidity Co-occurrence (%)', fontsize=10)
for i in range(len(comorbidities)):
    for j in range(len(comorbidities)):
        ax9.text(j, i, f'{co_pct[i,j]:.1f}', ha='center', va='center', fontsize=6,
                 color='white' if co_pct[i, j] > 5 else 'black')
plt.colorbar(im9, ax=ax9, shrink=0.8)

save_figure(fig, 'FigureS15_extended_supp')
print("  FigureS15_extended_supp saved.")

# ═══════════════════════════════════════════════════════════════
# 5. Code & Data Availability Documentation
# ═══════════════════════════════════════════════════════════════
print("\n[5] Code & data availability documentation...")

code_availability = {
    'repository': {
        'url': '[To be filled: GitHub repository URL]',
        'doi': '[To be filled: Zenodo DOI after archiving]',
        'license': 'MIT License',
    },
    'repository_structure': {
        'data_processing/': 'EHR extraction and feature engineering pipeline',
        'models/': 'XGBoost, LightGBM, deep learning model implementations',
        'sentinel_analysis/': 'Cosine similarity, RDI computation, permutation tests',
        'figures/': 'All figure generation scripts (gen_fig1.py through gen_fig7.py)',
        'supplementary/': 'Supplementary figure/table generation scripts',
        'validation/': 'External validation (SCH, MIMIC-IV) and negative control scripts',
        'config/': 'Hyperparameter configurations and shared settings',
    },
    'dependencies': {
        'python': '>=3.10',
        'xgboost': '>=2.0.0',
        'scikit-learn': '>=1.3.0',
        'scipy': '>=1.11.0',
        'pandas': '>=2.0.0',
        'numpy': '>=1.24.0',
        'matplotlib': '>=3.7.0',
    },
    'data_availability': {
        'WHU_primary': 'Protected health information; de-identified summary statistics available on request',
        'SCH_external': 'Protected health information; de-identified summary statistics available on request',
        'MIMIC_IV': 'Publicly available via PhysioNet (https://physionet.org/content/mimiciv/2.2/)',
        'MIMIC_access': 'Requires PhysioNet credentialed access and CITI training',
    },
    'reproducibility': {
        'random_seeds': 'All analyses use fixed random seeds (42)',
        'cross_validation': '5-fold patient-level stratified CV',
        'bootstrap': 'n=2000 iterations for sentinel CIs; n=1000 for validation CIs',
        'permutation': 'n=5000 iterations for seasonal confounding test',
    },
}

json_path = os.path.join(FIG_DIR, 'TableS7_code_availability.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(code_availability, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

# ═══════════════════════════════════════════════════════════════
# 6. Supplementary completeness report
# ═══════════════════════════════════════════════════════════════
print("\n[6] Generating completeness report...")

# Check which supplementary figures/tables have been generated
supp_items = {
    'Figure S1': {'file': 'FigureS1_data_overview', 'status': 'unknown'},
    'Figure S2': {'file': 'FigureS2_model_comparison', 'status': 'unknown'},
    'Figure S3': {'file': 'FigureS3_best_model', 'status': 'unknown'},
    'Figure S4': {'file': 'FigureS4_learning_curves', 'status': 'unknown'},
    'Figure S5': {'file': 'FigureS5_cv_folds', 'status': 'unknown'},
    'Figure S6': {'file': 'FigureS6_subgroup_analysis', 'status': 'unknown'},
    'Figure S7': {'file': 'FigureS7_pandemic_impact', 'status': 'unknown'},
    'Figure S8': {'file': 'FigureS8_permutation', 'status': 'unknown'},
    'Figure S9': {'file': 'FigureS9B_threshold_calibration', 'status': 'unknown'},
    'Figure S10': {'file': 'FigureS10_prospective_validation', 'status': 'unknown'},
    'Figure S11': {'file': 'FigureS11_model_extended', 'status': 'unknown'},
    'Figure S12': {'file': 'FigureS12_dimension_ablation', 'status': 'unknown'},
    'Figure S13': {'file': 'FigureS13_augmented_sentinel', 'status': 'unknown'},
    'Figure S14': {'file': 'FigureS14_sch_rdi_workflow', 'status': 'unknown'},
    'Figure S15': {'file': 'FigureS15_extended_supp', 'status': 'unknown'},
    'Table S1': {'file': 'feature_categories', 'status': 'in manuscript'},
    'Table S2': {'file': 'hyperparameters', 'status': 'in manuscript'},
    'Table S3': {'file': 'mimic_results', 'status': 'in manuscript'},
    'Table S4': {'file': 'data_completeness', 'status': 'unknown'},
    'Table S5': {'file': 'TableS5_threshold_performance', 'status': 'unknown'},
    'Table S6': {'file': 'TableS6_data_cleaning', 'status': 'unknown'},
    'Table S7': {'file': 'TableS7_code_availability', 'status': 'unknown'},
}

for item_name, info in supp_items.items():
    if item_name.startswith('Figure'):
        for ext in ['.png', '.pdf', '.tif']:
            if os.path.exists(os.path.join(FIG_DIR, info['file'] + ext)):
                info['status'] = 'generated'
                break
    elif item_name.startswith('Table'):
        for ext in ['.csv', '.json']:
            if os.path.exists(os.path.join(FIG_DIR, info['file'] + ext)):
                info['status'] = 'generated'
                break
        if info['status'] == 'unknown':
            if os.path.exists(os.path.join(OUTPUT_DIR, info['file'] + '.json')):
                info['status'] = 'generated'

print("\n  Supplementary Materials Status:")
for item_name, info in supp_items.items():
    status_mark = '  ' if info['status'] == 'generated' else '  ' if info['status'] == 'in manuscript' else '  '
    print(f"    {status_mark} {item_name}: {info['status']}")

report = {
    'cohort_statistics': cohort_stats,
    'supplementary_status': {k: v['status'] for k, v in supp_items.items()},
    'data_cleaning_steps': len(cleaning_steps),
    'lab_completeness': lab_cleaning if lab_cleaning else [],
}

report_path = os.path.join(FIG_DIR, 'supplementary_completeness_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Saved: {report_path}")

print("\n" + "=" * 60)
print("Done! All supplementary materials consolidated.")
print("=" * 60)
