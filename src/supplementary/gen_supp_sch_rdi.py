"""
Supplementary Analysis: RDI Workflow Applied to SCH
  + Policy-shock controls (2022 lockdown period)
  + Cross-center sentinel replication

Generates:
  - FigureS14_sch_rdi_workflow (PNG/PDF/TIF)
  - sch_rdi_results.json

Run: python gen_supp_sch_rdi.py
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
print("SCH RDI Workflow + Policy-Shock Controls")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load SCH data
# ═══════════════════════════════════════════════════════════════
print("[1] Loading SCH data...")
SCH_PATH = os.path.join(OUTPUT_DIR, '..', 'SCH_processed_data.csv')
if not os.path.exists(SCH_PATH):
    SCH_PATH = r'D:\LDH_cancer\files\healthline\SCH_processed_data.csv'

if not os.path.exists(SCH_PATH):
    print(f"  ! SCH data not found at {SCH_PATH}")
    print("  Searching alternative paths...")
    for candidate in [
        os.path.join(OUTPUT_DIR, 'SCH_processed_data.csv'),
        r'D:\LDH_cancer\files\healthline\readmission_output\SCH_processed_data.csv',
    ]:
        if os.path.exists(candidate):
            SCH_PATH = candidate
            break

print(f"  Loading from: {SCH_PATH}")
sch = pd.read_csv(SCH_PATH, low_memory=False)
print(f"  SCH records: {len(sch):,}")

# Standardize column names
# SCH uses different column names; adapt to the analysis framework
date_col = None
for c in ['入院时间', '检查时间', 'admission_time']:
    if c in sch.columns:
        date_col = c
        break

if date_col:
    sch['admit_dt'] = pd.to_datetime(sch[date_col], errors='coerce')
else:
    print("  ! No date column found. Attempting to infer...")
    for c in sch.columns:
        if '时间' in c or '日期' in c:
            sch['admit_dt'] = pd.to_datetime(sch[c], errors='coerce')
            date_col = c
            break

sch['year'] = sch['admit_dt'].dt.year
sch['month'] = sch['admit_dt'].dt.month
sch['quarter'] = sch['admit_dt'].dt.quarter

# Identify patient ID column
pid_col = None
for c in ['病案号', 'patient_id', '住院流水号']:
    if c in sch.columns:
        pid_col = c
        break

# Identify diagnosis column
diag_col = None
for c in ['出院主要诊断名称1', '门（急）诊诊断', 'diagnosis']:
    if c in sch.columns:
        diag_col = c
        break

print(f"  Date column: {date_col}")
print(f"  Patient ID column: {pid_col}")
print(f"  Diagnosis column: {diag_col}")
print(f"  Year range: {sch['year'].min()}–{sch['year'].max()}")

# ═══════════════════════════════════════════════════════════════
# 2. Comorbidity extraction for SCH
# ═══════════════════════════════════════════════════════════════
print("\n[2] Extracting comorbidities...")

# Cancer hospital: adapt comorbidity patterns
# SCH patients are predominantly cancer patients; we broaden patterns
# to capture cancer-site-based subgroups plus co-existing conditions
COMORBIDITY_PATTERNS = {
    'Cardiovascular': r'冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|高血压|心肌炎',
    'Lung_Cancer': r'肺恶性肿瘤|肺继发恶性|肺占位|肺原位癌|右肺.*恶性|左肺.*恶性',
    'GI_Cancer': r'直肠恶性肿瘤|食管.*恶性肿瘤|胃.*恶性肿瘤|贲门.*恶性|结肠恶性|乙状结肠恶性|升结肠恶性|胰腺恶性|十二指肠恶性|肝恶性肿瘤|肝继发恶性|肝细胞癌|肝内胆管癌|胆管恶性|胆囊恶性',
    'Breast_Cancer': r'乳腺恶性肿瘤|乳腺.*恶性|乳房恶性',
    'Renal_Urological': r'膀胱恶性肿瘤|肾恶性肿瘤|前列腺恶性|输尿管恶性',
    'Respiratory': r'肺炎|重症肺炎|间质性肺炎|放射性肺炎|新型冠状病毒|慢阻肺|COPD|哮喘|呼吸衰竭|呼吸道感染|胸腔积液|恶性胸腔积液|咯血|肺栓塞',
}

if diag_col:
    diag_text = sch[diag_col].fillna('').astype(str)
    for name, pattern in COMORBIDITY_PATTERNS.items():
        sch[name] = diag_text.str.contains(pattern, na=False).astype(int)
    for name in COMORBIDITY_PATTERNS:
        n = sch[name].sum()
        print(f"  {name}: n={n:,} ({100*n/len(sch):.1f}%)")
else:
    print("  ! No diagnosis column found; using empty comorbidities")
    for name in COMORBIDITY_PATTERNS:
        sch[name] = 0

comorbidities = list(COMORBIDITY_PATTERNS.keys())

# ═══════════════════════════════════════════════════════════════
# 3. Behavioral profile infrastructure (SCH-specific)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Setting up behavioral profiles...")

# Map SCH columns to behavioral metrics
# SCH may have different column names for LOS, lab values, etc.
sch_col_map = {}

# LOS
for c in ['实际住院天数', 'los_days', '住院天数']:
    if c in sch.columns:
        sch_col_map['LOS'] = c
        break

# Lab values - SCH uses asterisk prefix
lab_mapping = {
    'ALB': ['白蛋白', '★白蛋白', '*白蛋白', 'lab_ALB'],
    'CREA': ['肌酐', '★肌酐', '*肌酐', 'lab_CREA'],
    'GLU': ['葡萄糖', '★葡萄糖', '*葡萄糖', 'lab_GLU'],
    'K': ['钾', '★钾', '*钾', 'lab_K'],
    'Na': ['钠', '★钠', '*钠', 'lab_Na'],
    'WBC': ['白细胞计数', '★白细胞计数', '*白细胞计数'],
    'HGB': ['血红蛋白', '★血红蛋白', '*血红蛋白'],
}

for lab_name, candidates in lab_mapping.items():
    for c in candidates:
        if c in sch.columns:
            sch_col_map[lab_name] = c
            break

print(f"  Mapped columns: {sch_col_map}")

# Build available metrics list
sch_metrics = []
sch_metric_labels = []

if 'LOS' in sch_col_map:
    sch_metrics.append(sch_col_map['LOS'])
    sch_metric_labels.append('LOS')

for lab_name in ['ALB', 'CREA', 'GLU', 'K', 'Na', 'WBC', 'HGB']:
    if lab_name in sch_col_map:
        sch_metrics.append(sch_col_map[lab_name])
        sch_metric_labels.append(lab_name)

print(f"  Available metrics ({len(sch_metrics)}): {sch_metric_labels}")

# Baseline: first available full calendar year (pre-lockdown)
available_years = sorted(sch['year'].dropna().unique())
print(f"  Available years: {available_years}")

# Use 2021 as baseline (first full year in SCH data, pre-lockdown)
baseline_years = [y for y in available_years if y <= 2021]
if not baseline_years:
    baseline_years = available_years[:2]

sch_baseline = sch[sch['year'].isin(baseline_years)]
sch_ref_stats = {}
for col in sch_metrics:
    vals = pd.to_numeric(sch_baseline[col], errors='coerce').dropna()
    sch_ref_stats[col] = {'mean': vals.mean(), 'std': max(vals.std(), 1e-6)}

def sch_zscore_profile(sub_df):
    p = {}
    for col in sch_metrics:
        vals = pd.to_numeric(sub_df[col], errors='coerce').dropna()
        p[col] = (vals.mean() - sch_ref_stats[col]['mean']) / sch_ref_stats[col]['std'] if len(vals) > 0 else 0.0
    return p

def sch_profile_to_vec(prof):
    return np.nan_to_num(np.array([prof.get(k, 0) for k in sch_metrics], dtype=float), 0)

def cos_sim(v1, v2):
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        return 1 - cosine(v1, v2)
    return 0.0

# ═══════════════════════════════════════════════════════════════
# 4. Build respiratory-burden reference for SCH
# ═══════════════════════════════════════════════════════════════
print("\n[4] Building respiratory reference...")

# SCH: use respiratory comorbidity patients during known pandemic wave
# Dec 2022 (China reopening) as reference period
resp_wave = sch[(sch['Respiratory'] == 1) & (sch['year'] == 2022) & (sch['month'] == 12)]
if len(resp_wave) < 5:
    # Fallback: Q4 2022
    resp_wave = sch[(sch['Respiratory'] == 1) &
                    (sch['year'] == 2022) & (sch['quarter'] == 4)]
if len(resp_wave) < 5:
    # Broader fallback
    resp_wave = sch[(sch['Respiratory'] == 1) & (sch['year'] == 2022)]

vec_resp_ref = sch_profile_to_vec(sch_zscore_profile(resp_wave))
print(f"  Respiratory wave reference: n={len(resp_wave)} (from SCH)")

# ═══════════════════════════════════════════════════════════════
# 5. Compute monthly/quarterly RDI for SCH
# ═══════════════════════════════════════════════════════════════
print("\n[5] Computing quarterly RDI for SCH...")

quarterly_records = []
for year in available_years:
    for quarter in range(1, 5):
        sub_q = sch[(sch['year'] == year) & (sch['quarter'] == quarter)]
        if len(sub_q) < 10:
            continue
        sims = {}
        for comor in comorbidities:
            csub = sub_q[sub_q[comor] == 1]
            if len(csub) >= 3:
                v = sch_profile_to_vec(sch_zscore_profile(csub))
                sims[comor] = cos_sim(vec_resp_ref, v)
            else:
                sims[comor] = np.nan

        resp_sim = sims.get('Respiratory', np.nan)
        other_sims = [s for c, s in sims.items()
                      if c != 'Respiratory' and not np.isnan(s)]
        mean_other = np.mean(other_sims) if other_sims else np.nan
        rdi = resp_sim - mean_other if not np.isnan(resp_sim) and not np.isnan(mean_other) else np.nan

        # Respiratory admission rate
        n_resp = sub_q[sub_q['Respiratory'] == 1].shape[0]
        resp_rate = n_resp / len(sub_q)

        quarterly_records.append({
            'year': year, 'quarter': quarter,
            'yq': f"{year}Q{quarter}",
            'resp_sim': resp_sim,
            'mean_other': mean_other,
            'rdi': rdi,
            'resp_rate': resp_rate,
            'n_admissions': len(sub_q),
            **sims,
        })

df_quarterly = pd.DataFrame(quarterly_records)
print(f"  Computed {len(df_quarterly)} quarterly records")

# ═══════════════════════════════════════════════════════════════
# 6. Policy-shock detection & control
# ═══════════════════════════════════════════════════════════════
print("\n[6] Policy-shock analysis...")

# China lockdown: 2022 Q1-Q2 (Shanghai lockdown, regional restrictions)
# China reopening: 2022 Q4 (December wave)

# Compute visit gap statistics per quarter
if pid_col and date_col:
    sch_sorted = sch.sort_values([pid_col, 'admit_dt'])
    sch_sorted['prev_admit'] = sch_sorted.groupby(pid_col)['admit_dt'].shift(1)
    sch_sorted['gap_days'] = (sch_sorted['admit_dt'] - sch_sorted['prev_admit']).dt.days
    sch_sorted = sch_sorted[sch_sorted['gap_days'].notna() & (sch_sorted['gap_days'] > 0)]

    gap_stats = []
    for year in available_years:
        for quarter in range(1, 5):
            sub = sch_sorted[(sch_sorted['year'] == year) & (sch_sorted['quarter'] == quarter)]
            if len(sub) < 10:
                continue
            gaps = sub['gap_days'].dropna()
            gap_stats.append({
                'year': year, 'quarter': quarter,
                'yq': f"{year}Q{quarter}",
                'mean_gap': gaps.mean(),
                'median_gap': gaps.median(),
                'cv_gap': gaps.std() / gaps.mean() if gaps.mean() > 0 else np.nan,
                'n_visits': len(sub),
            })
    df_gap_stats = pd.DataFrame(gap_stats)
    print(f"  Gap statistics computed for {len(df_gap_stats)} quarters")

    # Identify lockdown quarters
    pre_lockdown = df_gap_stats[df_gap_stats['yq'].isin([f"{y}Q{q}" for y in [2021] for q in [1,2,3,4]])]
    lockdown = df_gap_stats[df_gap_stats['yq'].isin(['2022Q1', '2022Q2'])]
    post_lockdown = df_gap_stats[df_gap_stats['yq'].isin(['2022Q3', '2022Q4', '2023Q1'])]

    if len(pre_lockdown) > 0 and len(lockdown) > 0:
        print(f"  Pre-lockdown mean gap: {pre_lockdown['mean_gap'].mean():.1f} days")
        print(f"  Lockdown mean gap: {lockdown['mean_gap'].mean():.1f} days")
        print(f"  Gap increase during lockdown: "
              f"{(lockdown['mean_gap'].mean()/pre_lockdown['mean_gap'].mean()-1)*100:.1f}%")
        if len(post_lockdown) > 0:
            print(f"  Post-lockdown mean gap: {post_lockdown['mean_gap'].mean():.1f} days")
else:
    df_gap_stats = pd.DataFrame()
    print("  ! Cannot compute gap statistics (missing patient ID or date column)")

# ═══════════════════════════════════════════════════════════════
# 7. RDI with policy-shock covariate adjustment
# ═══════════════════════════════════════════════════════════════
print("\n[7] Policy-adjusted RDI...")

# Approach: compute RDI controlling for overall volume changes
# Adjust by normalizing RDI relative to admission volume deviation
if len(df_quarterly) > 0:
    baseline_volume = df_quarterly[df_quarterly['year'].isin(baseline_years)]['n_admissions'].mean()
    df_quarterly['volume_ratio'] = df_quarterly['n_admissions'] / baseline_volume
    df_quarterly['rdi_adjusted'] = df_quarterly['rdi'] / df_quarterly['volume_ratio'].replace(0, 1)

    # Flag lockdown quarters
    df_quarterly['period'] = 'Normal'
    df_quarterly.loc[(df_quarterly['year'] == 2022) &
                     (df_quarterly['quarter'].isin([1, 2])), 'period'] = 'Lockdown'
    df_quarterly.loc[(df_quarterly['year'] == 2022) &
                     (df_quarterly['quarter'] == 4), 'period'] = 'Reopening'

    print("  RDI summary by period:")
    for period in ['Normal', 'Lockdown', 'Reopening']:
        sub = df_quarterly[df_quarterly['period'] == period]
        if len(sub) > 0:
            print(f"    {period}: RDI={sub['rdi'].mean():.3f} (adj={sub['rdi_adjusted'].mean():.3f}), "
                  f"n_quarters={len(sub)}")

# ═══════════════════════════════════════════════════════════════
# 8. Generate Figure S14: SCH RDI Workflow (2×2)
# ═══════════════════════════════════════════════════════════════
print("\n[8] Generating FigureS14_sch_rdi_workflow...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, wspace=0.30, hspace=0.35,
                       left=0.08, right=0.96, top=0.93, bottom=0.08)

# ─── Panel A: Quarterly RDI timeline ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

if len(df_quarterly) > 0:
    x_labels = df_quarterly['yq'].values
    x_pos = np.arange(len(x_labels))

    # Color bars by period
    bar_colors = [COLORS['secondary'] if p == 'Lockdown'
                  else COLORS['accent1'] if p == 'Reopening'
                  else COLORS['primary']
                  for p in df_quarterly['period']]

    ax_a.bar(x_pos, df_quarterly['rdi'].values, color=bar_colors, edgecolor='white', alpha=0.8)
    ax_a.plot(x_pos, df_quarterly['rdi_adjusted'].values, 'k--', linewidth=1.2,
              marker='o', markersize=3, label='Volume-adjusted RDI')
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(x_labels, fontsize=6, rotation=45, ha='right')
    ax_a.set_ylabel('RDI', fontsize=9)
    ax_a.set_title('SCH Quarterly RDI Timeline', fontsize=11)
    ax_a.legend(fontsize=7)

    # Legend for periods
    from matplotlib.patches import Patch
    period_legend = [
        Patch(facecolor=COLORS['primary'], label='Normal'),
        Patch(facecolor=COLORS['secondary'], label='Lockdown'),
        Patch(facecolor=COLORS['accent1'], label='Reopening'),
    ]
    ax_a.legend(handles=period_legend, fontsize=7, loc='upper left', framealpha=0.9)
else:
    ax_a.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
              transform=ax_a.transAxes, fontsize=12)

# ─── Panel B: Comorbidity similarity heatmap ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

if len(df_quarterly) > 0:
    heatmap_data = np.zeros((len(comorbidities), len(df_quarterly)))
    for j, (_, row) in enumerate(df_quarterly.iterrows()):
        for i, comor in enumerate(comorbidities):
            val = row.get(comor, np.nan)
            # This is comorbidity count (0/1), we need similarity
            pass

    # Rebuild heatmap from similarity values
    sim_cols = [c for c in comorbidities if c in df_quarterly.columns]
    if sim_cols:
        hm = df_quarterly[sim_cols].values.T
        im = ax_b.imshow(hm, cmap=SPRING_CMAP, aspect='auto', vmin=-0.5, vmax=1.0)
        ax_b.set_yticks(range(len(sim_cols)))
        ax_b.set_yticklabels(sim_cols, fontsize=8)
        ax_b.set_xticks(range(len(df_quarterly)))
        ax_b.set_xticklabels(df_quarterly['yq'].values, fontsize=6, rotation=45, ha='right')
        ax_b.set_title('Comorbidity Similarity to Resp. Reference', fontsize=10)
        plt.colorbar(im, ax=ax_b, shrink=0.8)
    else:
        ax_b.text(0.5, 0.5, 'No similarity data', ha='center', va='center',
                  transform=ax_b.transAxes)

# ─── Panel C: Visit gap disruption ───
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, 'C')

if len(df_gap_stats) > 0:
    x_labels_c = df_gap_stats['yq'].values
    x_pos_c = np.arange(len(x_labels_c))

    ax_c.bar(x_pos_c, df_gap_stats['mean_gap'].values, color=COLORS['primary'],
             edgecolor='white', alpha=0.8, label='Mean gap')
    ax_c2 = ax_c.twinx()
    ax_c2.plot(x_pos_c, df_gap_stats['cv_gap'].values, 'o-', color=COLORS['secondary'],
               linewidth=1.5, markersize=4, label='Gap CV')

    ax_c.set_xticks(x_pos_c)
    ax_c.set_xticklabels(x_labels_c, fontsize=6, rotation=45, ha='right')
    ax_c.set_ylabel('Mean Visit Gap (days)', fontsize=9, color=COLORS['primary'])
    ax_c2.set_ylabel('Coefficient of Variation', fontsize=9, color=COLORS['secondary'])
    ax_c.set_title('SCH Visit Gap Disruption by Quarter', fontsize=11)

    # Mark lockdown
    for idx, yq in enumerate(x_labels_c):
        if yq in ['2022Q1', '2022Q2']:
            ax_c.axvspan(idx - 0.4, idx + 0.4, alpha=0.15, color=COLORS['secondary'])

    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
else:
    ax_c.text(0.5, 0.5, 'No gap data available', ha='center', va='center',
              transform=ax_c.transAxes, fontsize=12)

# ─── Panel D: Volume-normalized admission patterns ───
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, 'D')

if len(df_quarterly) > 0:
    ax_d.plot(range(len(df_quarterly)), df_quarterly['resp_rate'].values * 100,
              'o-', color=COLORS['secondary'], linewidth=1.5, markersize=4,
              label='Respiratory %')
    ax_d.plot(range(len(df_quarterly)), df_quarterly['volume_ratio'].values * 100,
              's--', color=COLORS['primary'], linewidth=1.2, markersize=3,
              label='Volume ratio (%)')
    ax_d.set_xticks(range(len(df_quarterly)))
    ax_d.set_xticklabels(df_quarterly['yq'].values, fontsize=6, rotation=45, ha='right')
    ax_d.set_ylabel('Percentage / Ratio (%)', fontsize=9)
    ax_d.set_title('SCH Respiratory Rate & Volume Changes', fontsize=11)
    ax_d.legend(fontsize=7)

    for idx, row in df_quarterly.iterrows():
        if row['period'] == 'Lockdown':
            pos = df_quarterly.index.get_loc(idx)
            ax_d.axvspan(pos - 0.4, pos + 0.4, alpha=0.1, color=COLORS['secondary'])
        elif row['period'] == 'Reopening':
            pos = df_quarterly.index.get_loc(idx)
            ax_d.axvspan(pos - 0.4, pos + 0.4, alpha=0.1, color=COLORS['accent1'])
else:
    ax_d.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
              transform=ax_d.transAxes, fontsize=12)

save_figure(fig, 'FigureS14_sch_rdi_workflow')
print("  FigureS14_sch_rdi_workflow saved.")

# ═══════════════════════════════════════════════════════════════
# 9. Save results
# ═══════════════════════════════════════════════════════════════
results = {
    'sch_summary': {
        'total_records': len(sch),
        'year_range': [int(sch['year'].min()), int(sch['year'].max())],
        'available_metrics': sch_metric_labels,
        'n_mapped_columns': len(sch_col_map),
    },
    'comorbidity_counts': {c: int(sch[c].sum()) for c in comorbidities},
    'quarterly_rdi': df_quarterly[['yq', 'rdi', 'rdi_adjusted', 'resp_sim',
                                    'resp_rate', 'n_admissions', 'period']].to_dict('records')
    if len(df_quarterly) > 0 else [],
    'gap_disruption': df_gap_stats.to_dict('records') if len(df_gap_stats) > 0 else [],
    'policy_shock_control': {
        'method': 'Volume-ratio normalization of RDI',
        'lockdown_quarters': ['2022Q1', '2022Q2'],
        'reopening_quarter': '2022Q4',
        'baseline_years': baseline_years,
    },
}

json_path = os.path.join(FIG_DIR, 'sch_rdi_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Saved: {json_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
