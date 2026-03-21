"""
Supplementary Figure S7: Pandemic Impact on Visit Regularity (3-panel)
  (A) SCH mean visit gap by quarter (2021–2025), lockdown highlight
  (B) Quarterly gap coefficient of variation (persistent irregularity)
  (C) WHU respiratory admission proportion change during pandemic onset

Generates:
  - FigureS7_pandemic_impact_extended (PNG/PDF/TIF)

Run: python gen_supp_pandemic_impact.py
"""
import sys, io, warnings, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure S2: Pandemic Impact on Visit Regularity")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load SCH data
# ═══════════════════════════════════════════════════════════════
print("[1] Loading SCH data...")

# Try loading from sch_rdi_results.json first (has pre-computed gap stats)
sch_rdi_path = os.path.join(FIG_DIR, 'sch_rdi_results.json')
sch_gap_data = None

if os.path.exists(sch_rdi_path):
    with open(sch_rdi_path, encoding='utf-8') as f:
        sch_rdi = json.load(f)
    if sch_rdi.get('gap_disruption'):
        sch_gap_data = pd.DataFrame(sch_rdi['gap_disruption'])
        print(f"  Loaded SCH gap stats from JSON: {len(sch_gap_data)} quarters")

if sch_gap_data is None or len(sch_gap_data) == 0:
    # Fallback: compute from raw SCH CSV
    SCH_PATH = None
    for candidate in [
        os.path.join(OUTPUT_DIR, '..', 'SCH_processed_data.csv'),
        r'D:\LDH_cancer\files\healthline\SCH_processed_data.csv',
        os.path.join(OUTPUT_DIR, 'SCH_processed_data.csv'),
    ]:
        if os.path.exists(candidate):
            SCH_PATH = candidate
            break

    if SCH_PATH:
        print(f"  Loading raw SCH from: {SCH_PATH}")
        sch = pd.read_csv(SCH_PATH, low_memory=False)
        date_col = None
        for c in ['入院时间', '检查时间', 'admission_time']:
            if c in sch.columns:
                date_col = c
                break
        if date_col is None:
            for c in sch.columns:
                if '时间' in c or '日期' in c:
                    date_col = c
                    break

        pid_col = None
        for c in ['病案号', 'patient_id', '住院流水号']:
            if c in sch.columns:
                pid_col = c
                break

        if date_col and pid_col:
            sch['admit_dt'] = pd.to_datetime(sch[date_col], errors='coerce')
            sch['year'] = sch['admit_dt'].dt.year
            sch['quarter'] = sch['admit_dt'].dt.quarter
            sch_sorted = sch.sort_values([pid_col, 'admit_dt'])
            sch_sorted['prev_admit'] = sch_sorted.groupby(pid_col)['admit_dt'].shift(1)
            sch_sorted['gap_days'] = (sch_sorted['admit_dt'] - sch_sorted['prev_admit']).dt.days
            sch_sorted = sch_sorted[sch_sorted['gap_days'].notna() & (sch_sorted['gap_days'] > 0)]

            gap_records = []
            for year in sch_sorted['year'].unique():
                for quarter in range(1, 5):
                    sub = sch_sorted[(sch_sorted['year'] == year) & (sch_sorted['quarter'] == quarter)]
                    if len(sub) < 10:
                        continue
                    gaps = sub['gap_days'].dropna()
                    gap_records.append({
                        'yq': f"{int(year)}Q{quarter}",
                        'mean_gap': gaps.mean(),
                        'median_gap': gaps.median(),
                        'cv_gap': gaps.std() / gaps.mean() if gaps.mean() > 0 else np.nan,
                        'n_visits': len(sub),
                    })
            sch_gap_data = pd.DataFrame(gap_records)
            print(f"  Computed SCH gap stats: {len(sch_gap_data)} quarters")

# ═══════════════════════════════════════════════════════════════
# 2. Load WHU data for respiratory proportion
# ═══════════════════════════════════════════════════════════════
print("\n[2] Loading WHU data...")
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['quarter'] = td['admit_dt'].dt.quarter
td['month'] = td['admit_dt'].dt.month

COMORBIDITY_PATTERNS = {
    'Respiratory': r'肺栓塞|肺炎|慢阻肺|COPD|哮喘|肺纤维化|呼吸衰竭',
}
diag_text = td['EMR_初步诊断'].fillna('').astype(str)
td['Respiratory'] = diag_text.str.contains(COMORBIDITY_PATTERNS['Respiratory'], na=False).astype(int)

# Quarterly respiratory proportion (2018-2020)
whu_records = []
for year in range(2018, 2021):
    for quarter in range(1, 5):
        if year == 2020 and quarter > 2:
            break
        sub = td[(td['year'] == year) & (td['quarter'] == quarter)]
        if len(sub) < 10:
            continue
        n_resp = sub['Respiratory'].sum()
        whu_records.append({
            'yq': f"{year}Q{quarter}",
            'year': year, 'quarter': quarter,
            'resp_pct': 100 * n_resp / len(sub),
            'n_total': len(sub),
            'n_resp': int(n_resp),
        })
whu_resp = pd.DataFrame(whu_records)
print(f"  WHU quarterly respiratory data: {len(whu_resp)} quarters")

# ═══════════════════════════════════════════════════════════════
# 3. Generate Figure S7 (1×3)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Generating FigureS2_pandemic_impact_extended...")

fig = plt.figure(figsize=(18, 5.5))
gs = gridspec.GridSpec(1, 3, wspace=0.30,
                       left=0.06, right=0.97, top=0.88, bottom=0.18)

# ─── Panel A: SCH mean visit gap by quarter ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

if sch_gap_data is not None and len(sch_gap_data) > 0:
    x_labels = sch_gap_data['yq'].values
    x_pos = np.arange(len(x_labels))

    bar_colors = []
    for yq in x_labels:
        if yq in ('2022Q1', '2022Q2'):
            bar_colors.append(COLORS['secondary'])
        elif yq == '2022Q4':
            bar_colors.append(COLORS['accent1'])
        else:
            bar_colors.append(COLORS['primary'])

    ax_a.bar(x_pos, sch_gap_data['mean_gap'].values, color=bar_colors,
             edgecolor='white', alpha=0.85)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(x_labels, fontsize=6, rotation=45, ha='right')
    ax_a.set_ylabel('Mean Visit Gap (days)', fontsize=9)
    ax_a.set_title('SCH Mean Visit Gap by Quarter', fontsize=10)

    # Lockdown annotation
    lockdown_idx = [i for i, yq in enumerate(x_labels) if yq in ('2022Q1', '2022Q2')]
    if lockdown_idx:
        mid = np.mean(lockdown_idx)
        ax_a.annotate('Lockdown', xy=(mid, ax_a.get_ylim()[1] * 0.92),
                      fontsize=8, ha='center', color=COLORS['secondary'],
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                ec=COLORS['secondary'], alpha=0.9))

    from matplotlib.patches import Patch
    ax_a.legend(handles=[
        Patch(facecolor=COLORS['primary'], label='Normal'),
        Patch(facecolor=COLORS['secondary'], label='Lockdown'),
        Patch(facecolor=COLORS['accent1'], label='Reopening'),
    ], fontsize=6.5, loc='upper left', framealpha=0.9)
else:
    ax_a.text(0.5, 0.5, 'SCH gap data not available', ha='center', va='center',
              transform=ax_a.transAxes, fontsize=10, color='grey')

# ─── Panel B: Quarterly gap CV ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

if sch_gap_data is not None and 'cv_gap' in sch_gap_data.columns and len(sch_gap_data) > 0:
    x_labels_b = sch_gap_data['yq'].values
    x_pos_b = np.arange(len(x_labels_b))
    cv_vals = sch_gap_data['cv_gap'].values

    ax_b.plot(x_pos_b, cv_vals, '-o', color=COLORS['secondary'], linewidth=1.8,
              markersize=5)
    ax_b.fill_between(x_pos_b, 0, cv_vals, alpha=0.12, color=COLORS['secondary'])

    # Highlight lockdown period
    for i, yq in enumerate(x_labels_b):
        if yq in ('2022Q1', '2022Q2'):
            ax_b.axvspan(i - 0.4, i + 0.4, alpha=0.15, color=COLORS['secondary'])

    # Pre-lockdown mean reference line
    pre_idx = [i for i, yq in enumerate(x_labels_b) if yq.startswith('2021')]
    if pre_idx:
        pre_mean = np.nanmean(cv_vals[pre_idx])
        ax_b.axhline(pre_mean, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax_b.text(len(x_labels_b) - 1, pre_mean + 0.02, f'2021 baseline: {pre_mean:.2f}',
                  fontsize=7, ha='right', color='grey')

    ax_b.set_xticks(x_pos_b)
    ax_b.set_xticklabels(x_labels_b, fontsize=6, rotation=45, ha='right')
    ax_b.set_ylabel('Coefficient of Variation', fontsize=9)
    ax_b.set_title('Visit Gap CV: Persistent Post-Lockdown Irregularity', fontsize=10)
else:
    ax_b.text(0.5, 0.5, 'CV data not available', ha='center', va='center',
              transform=ax_b.transAxes, fontsize=10, color='grey')

# ─── Panel C: WHU respiratory admission proportion ───
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, 'C')

if len(whu_resp) > 0:
    x_labels_c = whu_resp['yq'].values
    x_pos_c = np.arange(len(x_labels_c))

    bar_colors_c = []
    for _, row in whu_resp.iterrows():
        if row['year'] == 2020:
            bar_colors_c.append(COLORS['secondary'])
        elif row['year'] == 2019 and row['quarter'] >= 3:
            bar_colors_c.append(COLORS['accent2'])
        else:
            bar_colors_c.append(COLORS['primary'])

    ax_c.bar(x_pos_c, whu_resp['resp_pct'].values, color=bar_colors_c,
             edgecolor='white', alpha=0.85)

    # Reference line: 2018 mean
    baseline_pct = whu_resp[whu_resp['year'] == 2018]['resp_pct'].mean()
    ax_c.axhline(baseline_pct, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax_c.text(0, baseline_pct + 0.3, f'2018 mean: {baseline_pct:.1f}%',
              fontsize=7, color='grey')

    ax_c.set_xticks(x_pos_c)
    ax_c.set_xticklabels(x_labels_c, fontsize=7, rotation=45, ha='right')
    ax_c.set_ylabel('Respiratory Admission (%)', fontsize=9)
    ax_c.set_title('WHU Respiratory Proportion During Pandemic Onset', fontsize=10)

    from matplotlib.patches import Patch
    ax_c.legend(handles=[
        Patch(facecolor=COLORS['primary'], label='Baseline (2018)'),
        Patch(facecolor=COLORS['accent2'], label='Pre-signal (H2 2019)'),
        Patch(facecolor=COLORS['secondary'], label='Pandemic (2020)'),
    ], fontsize=6.5, loc='upper left', framealpha=0.9)

save_figure(fig, 'FigureS1_pandemic_impact_extended')
print("  FigureS2_pandemic_impact_extended saved.")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
