"""Phase B partial — add Bonferroni-corrected columns to lgdi lag spearman csv.

Source: NC_revision/lgdi_results/lgdi_whu_influenza_lag_spearman.csv (READ)
Dest:   NC_revision/RebuildRevision/outputs/lgdi_whu_influenza_lag_spearman_bonferroni.csv (WRITE)
We do NOT overwrite the source file — derived copy in RebuildRevision per Problem 11.
"""
import csv
from pathlib import Path

SRC = Path(r"data\readmission_output\figures_v2\merge\Submit\NC_revision\lgdi_results\lgdi_whu_influenza_lag_spearman.csv")
DST = Path(r"data\readmission_output\figures_v2\merge\Submit\NC_revision\RebuildRevision\outputs\lgdi_whu_influenza_lag_spearman_bonferroni.csv")
DST.parent.mkdir(parents=True, exist_ok=True)

K = 5   # number of lag tests (lag 0..4)
ALPHA = 0.05
ALPHA_CORR = ALPHA / K   # = 0.01

rows = []
with SRC.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        p = float(r["p_value"])
        p_corr = min(1.0, p * K)
        sig = "TRUE" if p_corr < ALPHA else "FALSE"
        r["p_value_bonferroni"] = f"{p_corr:.6g}"
        r["alpha_bonferroni"] = f"{ALPHA_CORR:.4f}"
        r["significant_after_bonferroni"] = sig
        r["correction_method"] = f"Bonferroni (k={K}, family-wise α=0.05)"
        rows.append(r)

fieldnames = list(rows[0].keys())
with DST.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[WROTE] {DST}")
print(f"Rows: {len(rows)}")
for r in rows:
    print(f"  lag={r['lgdi_leads_flunet_weeks']}  rho={r['spearman_rho'][:6]}  "
          f"p={r['p_value'][:8]}  p_corr={r['p_value_bonferroni']}  sig={r['significant_after_bonferroni']}")
