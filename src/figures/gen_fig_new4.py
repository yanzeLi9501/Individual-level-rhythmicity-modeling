"""
Figure 4: Post-pandemic Validation (2 panels)
==============================================
Selected from gen_fig7.py (originally 4 panels).

Layout (1 × 2):
  [A: Temporal COVID wave] [B: Negative control heatmap]

Also generates:
  FigureS10_postpandemic_extended (1×2): Bootstrap CI + Cross-era concordance
"""

import sys, io, warnings, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fig_config import *

print("=" * 60)
print("Figure 4: Post-pandemic Validation (2 panels)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load results
# ═══════════════════════════════════════════════════════════════
RESULT_DIR = os.path.join(OUTPUT_DIR, 'prospective_validation')
with open(os.path.join(RESULT_DIR, 'prospective_validation_results.json'), 'r', encoding='utf-8') as f:
    results = json.load(f)

monthly = pd.DataFrame(results['monthly_timeline'])
boot_ci = results['bootstrap_ci_q4_2022']
neg_ctrl = results['negative_control_q4_2022']
concordance = results['cross_era_concordance']
wave_info = results['covid_wave_dec2022']

TOP_COMOR = ['Cardiovascular', 'Hypertension', 'Diabetes',
             'Cerebrovascular', 'Renal', 'Respiratory']

comor_colors = {
    'Respiratory': COLORS['secondary'],
    'Cardiovascular': COLORS['primary'],
    'Hypertension': COLORS['accent1'],
    'Diabetes': COLORS['accent2'],
    'Cerebrovascular': COLORS['accent3'],
    'Renal': COLORS['accent5'],
}

comor_short = {
    'Cardiovascular': 'Cardio',
    'Hypertension': 'Hypert',
    'Diabetes': 'Diabetes',
    'Cerebrovascular': 'Cerebro',
    'Renal': 'Renal',
    'Respiratory': 'Resp',
}

# ═══════════════════════════════════════════════════════════════
# FIGURE — 1 × 3 layout
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(1, 2, wspace=0.35,
                       left=0.07, right=0.97, top=0.90, bottom=0.14)

# ─── Panel A: Temporal COVID wave analysis ────────────────────
ax_a = fig.add_subplot(gs[0])
ax_a.set_box_aspect(1)
panel_label(ax_a, 'A')

# Filter to wave-relevant period: May 2022 ~ Sep 2023
wave = monthly[
    ((monthly['year'] == 2022) & (monthly['month'] >= 5)) |
    ((monthly['year'] == 2023) & (monthly['month'] <= 9))
].copy()

if len(wave) > 0:
    x_idx = np.arange(len(wave))
    x_labels = [f"{int(r['month']):02d}" if int(r['month']) != 1
                else f"{int(r['year'])}\n{int(r['month']):02d}"
                for _, r in wave.iterrows()]

    # Bar: respiratory %
    bars = ax_a.bar(x_idx, wave['resp_pct'].values, color=COLORS['secondary'],
                    alpha=0.35, width=0.75, label='Respiratory %', zorder=2)

    # Highlight Dec 2022
    dec_mask = (wave['year'] == 2022) & (wave['month'] == 12)
    dec_idx = wave.index[dec_mask]
    for di in dec_idx:
        pos = list(wave.index).index(di)
        bars[pos].set_alpha(0.65)
        bars[pos].set_edgecolor(COLORS['secondary'])
        bars[pos].set_linewidth(1.5)

    ax_a.set_ylabel('Respiratory Admission %', color=COLORS['secondary'])
    ax_a.set_ylim(0, max(wave['resp_pct'].values) * 1.2)

    # Baseline reference line
    baseline_pct = wave_info['baseline_resp_pct']
    ax_a.axhline(baseline_pct, color=COLORS['secondary'], linewidth=0.8,
                 linestyle=':', alpha=0.6, label=f'Baseline ({baseline_pct:.1f}%)')

    # Twin axis: similarity lines
    ax_a2 = ax_a.twinx()

    resp_sim = wave['resp_sim'].values.astype(float)
    nonresp_sim = wave['nonresp_sim'].values.astype(float)

    ax_a2.plot(x_idx, resp_sim, '-o', color=COLORS['secondary'],
               linewidth=2.2, markersize=5, label='Respiratory sim.', zorder=5)
    ax_a2.plot(x_idx, nonresp_sim, '-s', color=COLORS['primary'],
               linewidth=1.5, markersize=4, alpha=0.7, label='Non-respiratory sim.', zorder=4)

    ax_a2.set_ylabel('Cosine Similarity to COVID+ Profile')
    ax_a2.set_ylim(-0.5, 1.05)

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(x_labels, fontsize=7, rotation=0)
    ax_a.set_xlabel('Month (2022-2023)')

    # Annotate Dec 2022 peak
    dec_pos = None
    for i, (_, r) in enumerate(wave.iterrows()):
        if int(r['year']) == 2022 and int(r['month']) == 12:
            dec_pos = i
            break
    if dec_pos is not None:
        peak_sim = resp_sim[dec_pos]
        peak_pct = wave['resp_pct'].values[dec_pos]
        ax_a2.annotate(
            f'Dec 2022 Wave\nResp%={peak_pct:.1f}%\nSim={peak_sim:.3f}',
            xy=(dec_pos, peak_sim),
            xytext=(dec_pos + 2.5, peak_sim + 0.15),
            fontsize=7.5, fontweight='bold', color=COLORS['secondary'],
            arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=COLORS['secondary'], alpha=0.95),
            zorder=10)

    # Combined legend
    lines1, labels1 = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(lines1 + lines2, labels1 + labels2,
                fontsize=7, loc='upper left', framealpha=0.9)

ax_a.set_title('Post-lockdown COVID Wave: Respiratory Surge & Behavioral Similarity')

# ─── Panel B: Negative control heatmap ───────────────────────
ax_b = fig.add_subplot(gs[1])
ax_b.set_box_aspect(1)
panel_label(ax_b, 'B')

ref_names = ['COVID+ (2022-12)', 'Heart Disease', 'Diabetes', 'General Pop.']
ref_short = ['COVID+', 'Heart Dis.', 'Diabetes', 'General']

# Build matrix
neg_matrix = np.full((len(ref_names), len(TOP_COMOR)), np.nan)
for i, ref in enumerate(ref_names):
    if ref in neg_ctrl:
        for j, comor in enumerate(TOP_COMOR):
            val = neg_ctrl[ref].get(comor, {}).get('sim', None)
            if val is not None:
                neg_matrix[i, j] = val

im = ax_b.imshow(neg_matrix, cmap=SPRING_CMAP, aspect='auto', vmin=-0.5, vmax=1.0)

for i in range(neg_matrix.shape[0]):
    for j in range(neg_matrix.shape[1]):
        v = neg_matrix[i, j]
        if not np.isnan(v):
            tc = 'white' if v > 0.75 or v < -0.2 else COLORS['text']
            ax_b.text(j, i, f'{v:.2f}', ha='center', va='center',
                      fontsize=9, color=tc, fontweight='bold')

# Highlight expected tops with rectangles
expected_tops = {
    'COVID+ (2022-12)': 'Respiratory',
    'Heart Disease': 'Cardiovascular',
    'Diabetes': 'Diabetes',
}
for i, ref in enumerate(ref_names):
    if ref in expected_tops:
        j = TOP_COMOR.index(expected_tops[ref])
        rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                      edgecolor='#333333', linewidth=2.2, linestyle='-')
        ax_b.add_patch(rect)

ax_b.set_xticks(range(len(TOP_COMOR)))
ax_b.set_xticklabels([comor_short[c] for c in TOP_COMOR], fontsize=8)
ax_b.set_yticks(range(len(ref_names)))
ax_b.set_yticklabels(ref_short, fontsize=9)
ax_b.set_xlabel('Target Comorbidity Group (Q4 2022)')
ax_b.set_ylabel('Reference Vector')
ax_b.set_title('Negative Control: Organ-System Specificity')

cbar = plt.colorbar(im, ax=ax_b, fraction=0.025, pad=0.03, shrink=0.85)
cbar.set_label('Cosine Similarity', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════
# Save main figure
# ═══════════════════════════════════════════════════════════════
save_figure(fig, 'Figure4_postpandemic_validation')
print("  Figure 4 (post-pandemic validation, 2 panels) saved.")

# ═══════════════════════════════════════════════════════════════
# SUPPLEMENTARY: FigureS10 — Post-pandemic Extended (1×2)
# ═══════════════════════════════════════════════════════════════
print("\n  Generating FigureS10_postpandemic_extended...")

fig_s = plt.figure(figsize=(14, 6))
gs_s = gridspec.GridSpec(1, 2, wspace=0.35,
                         left=0.06, right=0.96, top=0.90, bottom=0.12)

# ─── S10-A: Bootstrap CI bars ────────────────────────────────
ax_sa = fig_s.add_subplot(gs_s[0])
panel_label(ax_sa, 'A')

y_pos = np.arange(len(TOP_COMOR))
means = []
ci_lo = []
ci_hi = []
colors_s = []
for comor in TOP_COMOR:
    ci = boot_ci.get(comor, {})
    m = ci.get('mean', None)
    lo = ci.get('ci_low', None)
    hi = ci.get('ci_high', None)
    means.append(m if m is not None else 0)
    ci_lo.append(lo if lo is not None else 0)
    ci_hi.append(hi if hi is not None else 0)
    colors_s.append(comor_colors[comor])

means = np.array(means)
ci_lo = np.array(ci_lo)
ci_hi = np.array(ci_hi)
err_lo = means - ci_lo
err_hi = ci_hi - means

ax_sa.barh(y_pos, means, xerr=[err_lo, err_hi],
           color=colors_s, alpha=0.75, height=0.65,
           capsize=4, error_kw={'linewidth': 1.3, 'capthick': 1.3},
           edgecolor='white', linewidth=0.5)

for i, (m, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
    n = boot_ci.get(TOP_COMOR[i], {}).get('n', 0)
    ax_sa.text(max(hi + 0.03, m + 0.05), i,
               f'{m:.3f}\n[{lo:.3f}, {hi:.3f}]\nn={n}',
               va='center', fontsize=7, color=COLORS['text'])

ax_sa.set_yticks(y_pos)
ax_sa.set_yticklabels(TOP_COMOR, fontsize=9)
ax_sa.set_xlabel('Cosine Similarity to COVID+ Profile (2022-12 Reference)')
ax_sa.set_title('Bootstrap 95% CIs: Q4 2022 (COVID Wave Quarter)')
ax_sa.axvline(0, color='grey', linewidth=0.8, linestyle=':', alpha=0.5)
ax_sa.set_xlim(-0.6, 1.1)
ax_sa.invert_yaxis()

# ─── S10-B: Cross-era concordance scatter ────────────────────
ax_sb = fig_s.add_subplot(gs_s[1])
panel_label(ax_sb, 'B')

original_q4_2019 = {
    'Cardiovascular': 0.34, 'Hypertension': 0.08, 'Diabetes': 0.33,
    'Cerebrovascular': 0.25, 'Renal': 0.30, 'Respiratory': 0.84
}

new_q4_2022 = {c: neg_ctrl.get('COVID+ (2022-12)', {}).get(c, {}).get('sim', None)
               for c in TOP_COMOR}

for comor in TOP_COMOR:
    x_val = original_q4_2019.get(comor, None)
    y_val = new_q4_2022.get(comor, None)
    if x_val is not None and y_val is not None:
        ax_sb.scatter(x_val, y_val, color=comor_colors[comor],
                      s=120, zorder=5, edgecolors='white', linewidths=1.2)
        offset_x, offset_y = 0.03, 0.03
        if comor == 'Hypertension':
            offset_y = -0.07
        elif comor == 'Cardiovascular':
            offset_x = -0.15
            offset_y = 0.04
        ax_sb.text(x_val + offset_x, y_val + offset_y,
                   comor_short[comor], fontsize=8, color=comor_colors[comor],
                   fontweight='bold')

ax_sb.plot([-0.5, 1.0], [-0.5, 1.0], '--', color='grey', linewidth=0.8, alpha=0.5)

rho = concordance.get('spearman_rho', None)
p_val = concordance.get('p', None)
if rho is not None:
    ax_sb.text(0.05, 0.95,
               f'Spearman $\\rho$ = {rho:.3f}\np = {p_val:.3f}\n\nBoth eras: Respiratory = TOP',
               transform=ax_sb.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round,pad=0.4', fc='white',
                         ec=COLORS['grid'], alpha=0.9))

ax_sb.set_xlabel('Original Analysis (2020 COVID+ ref, Q4 2019)')
ax_sb.set_ylabel('Prospective Validation (2022 COVID+ ref, Q4 2022)')
ax_sb.set_title('Cross-Era Concordance of Comorbidity Rankings')
ax_sb.set_xlim(-0.2, 1.0)
ax_sb.set_ylim(-0.3, 0.9)
ax_sb.axhline(0, color='grey', linewidth=0.5, linestyle=':', alpha=0.3)
ax_sb.axvline(0, color='grey', linewidth=0.5, linestyle=':', alpha=0.3)

save_figure(fig_s, 'FigureS6_postpandemic_extended')
print("  FigureS10_postpandemic_extended saved.")
print("\nFigure 4 + S10 complete!")
