"""
Supplementary Analysis: Hospital-Internal Lead-Time Metrics
  + Sensitivity / Specificity / PPV / NPV Tables
  + FigureS9B extended threshold calibration

Generates:
  - FigureS8B_threshold_calibration (PNG/PDF/TIF)
  - TableS5_threshold_performance.csv
  - leadtime_metrics.json

Run: python gen_supp_leadtime.py
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
print("Lead-Time Metrics + Threshold Calibration")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load data & build behavioural profiles
# ═══════════════════════════════════════════════════════════════
print("[1] Loading data...")
td = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_admissions.csv'), low_memory=False)
td['admit_dt'] = pd.to_datetime(td['入院日期'], errors='coerce')
td['year'] = td['admit_dt'].dt.year
td['month'] = td['admit_dt'].dt.month
td['quarter'] = td['admit_dt'].dt.quarter
td['ym'] = td['year'] * 100 + td['month']

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
print(f"  Total admissions: {len(td):,}")

comorbidities = ['Cardiovascular', 'Hypertension', 'Diabetes',
                 'Cerebrovascular', 'Renal', 'Respiratory']

# ═══════════════════════════════════════════════════════════════
# 2. Compute monthly RDI for baseline + monitoring period
# ═══════════════════════════════════════════════════════════════
print("\n[2] Computing monthly RDI (2016-01 → 2020-06)...")

# Months of interest
monitor_start_year, monitor_start_month = 2016, 1
monitor_end_year, monitor_end_month = 2020, 6

monthly_records = []
for year in range(monitor_start_year, monitor_end_year + 1):
    for month in range(1, 13):
        if year == monitor_end_year and month > monitor_end_month:
            break
        sub_month = td[(td['year'] == year) & (td['month'] == month)]
        if len(sub_month) < 10:
            continue
        sims = {}
        for comor in comorbidities:
            csub = sub_month[sub_month[comor] == 1]
            if len(csub) >= 3:
                v = profile_to_vec(zscore_profile(csub))
                sims[comor] = cos_sim(vec_pos, v)
            else:
                sims[comor] = np.nan

        resp_sim = sims.get('Respiratory', np.nan)
        other_sims = [s for c, s in sims.items()
                      if c != 'Respiratory' and not np.isnan(s)]
        mean_other = np.mean(other_sims) if other_sims else np.nan
        rdi = resp_sim - mean_other if not np.isnan(resp_sim) and not np.isnan(mean_other) else np.nan

        # Respiratory admission rate
        resp_admissions = sub_month[sub_month['Respiratory'] == 1]
        resp_rate = len(resp_admissions) / len(sub_month) if len(sub_month) > 0 else 0

        monthly_records.append({
            'year': year, 'month': month,
            'ym': year * 100 + month,
            'resp_sim': resp_sim,
            'mean_other': mean_other,
            'rdi': rdi,
            'resp_rate': resp_rate,
            'n_admissions': len(sub_month),
            **sims,
        })

df_monthly = pd.DataFrame(monthly_records)
print(f"  Computed {len(df_monthly)} monthly records")

# ═══════════════════════════════════════════════════════════════
# 3. Baseline statistics (2017–2018)
# ═══════════════════════════════════════════════════════════════
baseline_months = df_monthly[(df_monthly['year'] >= 2017) & (df_monthly['year'] <= 2018)]
rdi_baseline_mean = baseline_months['rdi'].mean()
rdi_baseline_std = baseline_months['rdi'].std()
resp_sim_baseline = baseline_months['resp_sim'].dropna()

# Traditional: season-matched respiratory rate
resp_rate_baseline = baseline_months.groupby('month')['resp_rate'].agg(['mean', 'std']).reset_index()
resp_rate_baseline.columns = ['month', 'rate_mean', 'rate_std']

print(f"  Baseline RDI: mean={rdi_baseline_mean:.3f}, std={rdi_baseline_std:.3f}")

# ═══════════════════════════════════════════════════════════════
# 4. Ground-truth labelling
# ═══════════════════════════════════════════════════════════════
# Pandemic onset: January 2020 (first confirmed cases at WHU)
# Based on retrospective evidence: Sep 2019 onwards considered true-positive
# months for the behavioral sentinel system.
# Conservative: only Jan 2020+ as true pandemic months
# Liberal: Sep 2019+ (Apolone et al. serology)

# We test both definitions
GROUND_TRUTH_CONSERVATIVE = {(2020, m) for m in range(1, 7)}
GROUND_TRUTH_LIBERAL = {(2019, m) for m in range(9, 13)} | {(2020, m) for m in range(1, 7)}

monitoring = df_monthly[(df_monthly['year'] >= 2019) & (df_monthly['ym'] <= 202006)].copy()

# ═══════════════════════════════════════════════════════════════
# 5. Threshold sweep → Sensitivity / Specificity / PPV / NPV
# ═══════════════════════════════════════════════════════════════
print("\n[3] Threshold sweep...")

thresholds_sd = np.arange(0.5, 3.1, 0.25)
results_rows = []

for gt_name, gt_set in [('Conservative (Jan 2020+)', GROUND_TRUTH_CONSERVATIVE),
                         ('Liberal (Sep 2019+)', GROUND_TRUTH_LIBERAL)]:
    for thr_sd in thresholds_sd:
        rdi_threshold = rdi_baseline_mean + thr_sd * rdi_baseline_std
        resp_pct_threshold = np.percentile(resp_sim_baseline, 100 - (100 * 0.05 / (thr_sd / 1.5)))
        resp_pct_threshold = min(resp_pct_threshold, np.percentile(resp_sim_baseline, 97.5))

        tp = fp = tn = fn = 0
        alert_months = []
        for _, row in monitoring.iterrows():
            is_positive = (row['year'], row['month']) in gt_set
            rdi_alert = (not np.isnan(row['rdi'])) and (row['rdi'] > rdi_threshold)
            sim_alert = (not np.isnan(row['resp_sim'])) and (row['resp_sim'] > resp_pct_threshold)
            alert = rdi_alert or sim_alert

            if alert and is_positive:
                tp += 1
            elif alert and not is_positive:
                fp += 1
            elif not alert and is_positive:
                fn += 1
            else:
                tn += 1

            if alert:
                alert_months.append(f"{int(row['year'])}-{int(row['month']):02d}")

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

        # Lead time: months between first alert and Jan 2020
        first_alert_month = None
        if alert_months:
            first_dt = pd.to_datetime(alert_months[0] + '-01')
            pandemic_dt = pd.to_datetime('2020-01-01')
            lead_months = (pandemic_dt.year - first_dt.year) * 12 + (pandemic_dt.month - first_dt.month)
            first_alert_month = alert_months[0]
        else:
            lead_months = 0

        results_rows.append({
            'ground_truth': gt_name,
            'threshold_sd': thr_sd,
            'rdi_threshold': rdi_threshold,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'PPV': ppv, 'NPV': npv,
            'F1': f1,
            'n_alerts': tp + fp,
            'first_alert': first_alert_month,
            'lead_months': lead_months,
            'alert_months': '; '.join(alert_months),
        })

df_results = pd.DataFrame(results_rows)

# Print table
print(f"\n  {'Ground Truth':<28s} {'Thr(SD)':>8s} {'Sens':>6s} {'Spec':>6s} "
      f"{'PPV':>6s} {'NPV':>6s} {'F1':>6s} {'Lead':>5s} {'Alerts':>7s}")
print("  " + "-" * 90)
for _, r in df_results.iterrows():
    print(f"  {r['ground_truth']:<28s} {r['threshold_sd']:>8.2f} "
          f"{r['sensitivity']:>6.2f} {r['specificity']:>6.2f} "
          f"{r['PPV']:>6.2f} {r['NPV']:>6.2f} {r['F1']:>6.2f} "
          f"{r['lead_months']:>5.0f} {r['n_alerts']:>7d}")

# ═══════════════════════════════════════════════════════════════
# 6. Bootstrap CIs for lead-time at key thresholds
# ═══════════════════════════════════════════════════════════════
print("\n[4] Bootstrap CIs for lead-time metrics...")

N_BOOT = 2000
rng = np.random.default_rng(42)
key_thresholds = [1.0, 1.5, 2.0, 2.5]

boot_results = {}
for thr_sd in key_thresholds:
    rdi_thr = rdi_baseline_mean + thr_sd * rdi_baseline_std
    pct_thr = np.percentile(resp_sim_baseline, 97.5)

    lead_times_boot = []
    sens_boot = []
    spec_boot = []
    ppv_boot = []

    for _ in range(N_BOOT):
        # Resample baseline months to get bootstrapped threshold
        boot_idx = rng.choice(len(baseline_months), size=len(baseline_months), replace=True)
        boot_baseline = baseline_months.iloc[boot_idx]
        boot_rdi_mean = boot_baseline['rdi'].mean()
        boot_rdi_std = max(boot_baseline['rdi'].std(), 1e-6)
        boot_thr = boot_rdi_mean + thr_sd * boot_rdi_std
        boot_pct = np.percentile(boot_baseline['resp_sim'].dropna(), 97.5)

        tp = fp = tn = fn = 0
        first_alert_ym = None
        for _, row in monitoring.iterrows():
            is_pos = (row['year'], row['month']) in GROUND_TRUTH_LIBERAL
            alert = ((not np.isnan(row['rdi'])) and row['rdi'] > boot_thr) or \
                    ((not np.isnan(row['resp_sim'])) and row['resp_sim'] > boot_pct)
            if alert and is_pos:
                tp += 1
            elif alert and not is_pos:
                fp += 1
            elif not alert and is_pos:
                fn += 1
            else:
                tn += 1
            if alert and first_alert_ym is None:
                first_alert_ym = row['ym']

        if first_alert_ym is not None:
            fy = int(first_alert_ym) // 100
            fm = int(first_alert_ym) % 100
            lead = (2020 - fy) * 12 + (1 - fm)
            lead_times_boot.append(lead)
        else:
            lead_times_boot.append(0)

        sens_boot.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_boot.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv_boot.append(tp / (tp + fp) if (tp + fp) > 0 else 0)

    boot_results[thr_sd] = {
        'lead_mean': np.mean(lead_times_boot),
        'lead_ci': [float(np.percentile(lead_times_boot, 2.5)),
                     float(np.percentile(lead_times_boot, 97.5))],
        'sens_mean': np.mean(sens_boot),
        'sens_ci': [float(np.percentile(sens_boot, 2.5)),
                     float(np.percentile(sens_boot, 97.5))],
        'spec_mean': np.mean(spec_boot),
        'spec_ci': [float(np.percentile(spec_boot, 2.5)),
                     float(np.percentile(spec_boot, 97.5))],
        'ppv_mean': np.mean(ppv_boot),
        'ppv_ci': [float(np.percentile(ppv_boot, 2.5)),
                    float(np.percentile(ppv_boot, 97.5))],
    }
    ci = boot_results[thr_sd]
    print(f"  Thr={thr_sd:.1f}σ: Lead={ci['lead_mean']:.1f} mo "
          f"[{ci['lead_ci'][0]:.0f}–{ci['lead_ci'][1]:.0f}], "
          f"Sens={ci['sens_mean']:.2f} [{ci['sens_ci'][0]:.2f}–{ci['sens_ci'][1]:.2f}], "
          f"Spec={ci['spec_mean']:.2f} [{ci['spec_ci'][0]:.2f}–{ci['spec_ci'][1]:.2f}]")

# ═══════════════════════════════════════════════════════════════
# 7. Traditional surveillance comparison
# ═══════════════════════════════════════════════════════════════
print("\n[5] Traditional surveillance lead-time...")

trad_results = []
for _, row in monitoring.iterrows():
    m = int(row['month'])
    rate_row = resp_rate_baseline[resp_rate_baseline['month'] == m]
    if len(rate_row) > 0:
        thr = rate_row['rate_mean'].values[0] + 2 * rate_row['rate_std'].values[0]
        alert = row['resp_rate'] > thr
    else:
        alert = False
    trad_results.append({
        'year': row['year'], 'month': row['month'],
        'resp_rate': row['resp_rate'], 'alert': alert,
    })

df_trad = pd.DataFrame(trad_results)
trad_alerts = df_trad[df_trad['alert']]
if len(trad_alerts) > 0:
    first_trad = trad_alerts.iloc[0]
    trad_fy, trad_fm = int(first_trad['year']), int(first_trad['month'])
    trad_lead = (2020 - trad_fy) * 12 + (1 - trad_fm)
    print(f"  Traditional first alert: {trad_fy}-{trad_fm:02d} (lead = {trad_lead} months)")
else:
    trad_lead = 0
    print("  No traditional alert during monitoring period")

# ═══════════════════════════════════════════════════════════════
# 8. Generate Figure: Threshold Calibration (2×2)
# ═══════════════════════════════════════════════════════════════
print("\n[6] Generating FigureS8B_threshold_calibration...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.40,
                       left=0.08, right=0.96, top=0.93, bottom=0.08)

# ─── Panel A: Sensitivity vs Specificity across thresholds ───
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'A')

for gt_name, ls in [('Conservative (Jan 2020+)', '-'), ('Liberal (Sep 2019+)', '--')]:
    sub = df_results[df_results['ground_truth'] == gt_name]
    ax_a.plot(sub['threshold_sd'], sub['sensitivity'], ls, color=COLORS['secondary'],
              label=f'Sensitivity ({gt_name[:4]})', linewidth=1.5, marker='o', markersize=3)
    ax_a.plot(sub['threshold_sd'], sub['specificity'], ls, color=COLORS['primary'],
              label=f'Specificity ({gt_name[:4]})', linewidth=1.5, marker='s', markersize=3)

ax_a.axvline(1.5, color='grey', linestyle=':', alpha=0.5, label='Default (1.5σ)')
ax_a.set_xlabel('RDI Threshold (σ above baseline)', fontsize=9)
ax_a.set_ylabel('Rate', fontsize=9)
ax_a.set_title('Sensitivity & Specificity vs. Threshold', fontsize=11)
ax_a.legend(fontsize=6, loc='center right', framealpha=0.9)
ax_a.set_ylim(-0.05, 1.1)

# ─── Panel B: PPV and NPV ───
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'B')

sub_lib = df_results[df_results['ground_truth'] == 'Liberal (Sep 2019+)']
ax_b.plot(sub_lib['threshold_sd'], sub_lib['PPV'], '-', color=COLORS['accent1'],
          label='PPV', linewidth=1.8, marker='o', markersize=4)
ax_b.plot(sub_lib['threshold_sd'], sub_lib['NPV'], '-', color=COLORS['accent2'],
          label='NPV', linewidth=1.8, marker='s', markersize=4)
ax_b.plot(sub_lib['threshold_sd'], sub_lib['F1'], '--', color=COLORS['accent3'],
          label='F1 Score', linewidth=1.4, marker='^', markersize=3)

ax_b.axvline(1.5, color='grey', linestyle=':', alpha=0.5)
ax_b.set_xlabel('RDI Threshold (σ above baseline)', fontsize=9)
ax_b.set_ylabel('Rate', fontsize=9)
ax_b.set_title('PPV, NPV, and F1 vs. Threshold', fontsize=11)
ax_b.legend(fontsize=7, loc='center right', framealpha=0.9)
ax_b.set_ylim(-0.05, 1.1)

# ─── Panel C: Lead-time with bootstrap CIs ───
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, 'C')

thr_keys = sorted(boot_results.keys())
leads = [boot_results[k]['lead_mean'] for k in thr_keys]
lead_lo = [boot_results[k]['lead_ci'][0] for k in thr_keys]
lead_hi = [boot_results[k]['lead_ci'][1] for k in thr_keys]

ax_c.bar(thr_keys, leads, width=0.35, color=COLORS['primary'], alpha=0.8,
         edgecolor='white', label='Mean lead-time')
ax_c.errorbar(thr_keys, leads,
              yerr=[np.array(leads) - np.array(lead_lo),
                    np.array(lead_hi) - np.array(leads)],
              fmt='none', ecolor='black', capsize=4, linewidth=1.2)
ax_c.axhline(0, color='black', linewidth=0.5)
ax_c.axhline(-trad_lead, color=COLORS['secondary'], linestyle='--', linewidth=1,
             label=f'Traditional ({trad_lead} mo)')
ax_c.set_xlabel('RDI Threshold (σ)', fontsize=9)
ax_c.set_ylabel('Lead-Time (months before Jan 2020)', fontsize=9)
ax_c.set_title('Lead-Time by Threshold (Bootstrap 95% CI)', fontsize=11)
ax_c.legend(fontsize=7, framealpha=0.9)

# ─── Panel D: Number of alerts (true vs false) ───
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, 'D')

sub_lib2 = df_results[df_results['ground_truth'] == 'Liberal (Sep 2019+)'].copy()
x_pos = np.arange(len(sub_lib2))
width = 0.35
ax_d.bar(x_pos - width / 2, sub_lib2['TP'].values, width,
         label='True Positives', color=COLORS['accent1'], edgecolor='white')
ax_d.bar(x_pos + width / 2, sub_lib2['FP'].values, width,
         label='False Positives', color=COLORS['secondary'], edgecolor='white')
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels([f'{t:.1f}σ' for t in sub_lib2['threshold_sd']], fontsize=7, rotation=45)
ax_d.set_xlabel('RDI Threshold', fontsize=9)
ax_d.set_ylabel('Number of Months', fontsize=9)
ax_d.set_title('Alert Composition by Threshold', fontsize=11)
ax_d.legend(fontsize=7, framealpha=0.9)

save_figure(fig, 'FigureS8B_threshold_calibration')
print("  FigureS8B_threshold_calibration saved.")

# ═══════════════════════════════════════════════════════════════
# 9. Save results
# ═══════════════════════════════════════════════════════════════
csv_path = os.path.join(FIG_DIR, 'TableS5_threshold_performance.csv')
df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n  Saved: {csv_path}")

# JSON summary
summary = {
    'baseline_period': '2017-2018',
    'monitoring_period': '2019-01 to 2020-06',
    'rdi_baseline_mean': float(rdi_baseline_mean),
    'rdi_baseline_std': float(rdi_baseline_std),
    'traditional_first_alert': f"{trad_fy}-{trad_fm:02d}" if len(trad_alerts) > 0 else None,
    'traditional_lead_months': int(trad_lead),
    'bootstrap_n': N_BOOT,
    'bootstrap_results': {str(k): v for k, v in boot_results.items()},
}
json_path = os.path.join(FIG_DIR, 'leadtime_metrics.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
