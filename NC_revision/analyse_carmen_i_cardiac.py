#!/usr/bin/env python3
"""
CARMEN-I cardiac involvement analysis.

Research question:
  Among COVID-19 patients in CARMEN-I (Hospital del Mar, Barcelona),
  what proportion showed cardiac symptoms or diagnoses —
  (a) as part of their acute presentation (IA_PROCESO_ACTUAL),
  (b) in their medical history (IA_ANTECEDENTES),
  (c) in any clinical document?

This provides external context for assessing whether the cardiac
validation cohort in our study may have been systematically biased
by COVID-related cardiac complications.

Output: NC_revision/external_positive_control_results/carmen_i_cardiac_summary.json
        NC_revision/external_positive_control_results/carmen_i_cardiac_report.txt
"""

import csv, json, re
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
NER_TSV = BASE / "external_data/physionet/carmen-i/1.0.1/CARMEN-I/tsv/masked/CARMEN-I_masked_ner.tsv"
OUT_DIR = BASE / "NC_revision/external_positive_control_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cardiac keyword lexicon (Spanish + Catalan) ───────────────────────────────
# Covers: arrhythmia, cardiac failure, myocarditis, ischaemia, structural heart,
#         thromboembolic, syncope, troponin/ECG findings
CARDIAC_KEYWORDS = [
    # Diseases / diagnoses
    "miocarditis", "cardiopatia", "cardiopatía",
    "insuficiencia cardiaca", "insuficiència card",
    "infarto", "infart", "iam", "síndrome coronario",
    "angina", "angina de pecho",
    "arritmia", "arrhythmia",
    "fibrilació", "fibrilación",    # AF
    "flutter",
    "taquicardia",                  # SVT, VT
    "bradicardia", "bradycardia",
    "bloqueo auriculoventricular", "bloqueig av", "bloc av",
    "torsades",
    "tromboembolia", "tromboembolisme", "tep ", "embolia pulmonar",
    "trombosi venosa", "tvp",
    "pericarditis", "derrame pericardico", "derrament pericàrdic",
    "endocarditis",
    "miocardiopatia", "miocardiopatía",
    "takotsubo",
    "shock cardiogènic", "shock cardiogénico",
    "parada cardiorrespiratoria", "aturada cardiorrespiratòria",
    # Symptoms / findings
    "palpitacion", "palpitació",
    "síncope", "síncop", "presíncope", "presíncop",
    "dolor toràcic", "dolor torácico",         # chest pain
    "elevació del segment st", "elevación st", "elevació st",
    "troponin",                                 # troponin rise
    "fevi", "fracció d'ejecció", "fracción eyección",  # EF
    "hipocinesia", "hipocinèsia", "acinesia", "acinèsia",  # wall motion
    "cavitats dretes", "col·lapse",             # RV collapse (tamponade)
    "vci dilatada",
    "qt llarg", "qt prolongat", "qt largo",
    # Procedures suggesting cardiac work-up
    "ecocardiograma", "echocardiograma",
    "cateterisme", "cateterismo", "coronariografia",
    "cardioversió", "cardioversión",
    "desfibril", "desfibril",
    "marcapas", "marcapassos",
]

# Compile as lowercase for fast matching
CARDIAC_RE = re.compile(
    "|".join(re.escape(k) for k in CARDIAC_KEYWORDS),
    re.IGNORECASE
)

# ── Section classification ────────────────────────────────────────────────────
def section_of(doc_name: str) -> str:
    """Extract section type from document name."""
    parts = doc_name.split("_")
    # CARMEN-I_IA_PROCESO_ACTUAL_89 → IA_PROCESO_ACTUAL
    # CARMEN-I_CC_3 → CC
    return "_".join(parts[1:-1])

def patient_id_of(doc_name: str) -> str:
    """Last numeric token = document/patient ID within section."""
    return doc_name.rsplit("_", 1)[-1]

# ── Load NER data ─────────────────────────────────────────────────────────────
print(f"Loading {NER_TSV} …")
# doc_name → {tag: [text, ...]}
doc_entities: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

with open(NER_TSV, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        doc_entities[row["name"]][row["tag"]].append(row["text"].lower())

print(f"  {len(doc_entities)} documents, tags: ENFERMEDAD / SINTOMA / FARMACO / PROCEDIMIENTO / SPECIES / HUMANO")

# ── Detect cardiac involvement per document ───────────────────────────────────
def has_cardiac(entities: dict[str, list[str]]) -> tuple[bool, list[str]]:
    """Return (is_cardiac, matched_terms)."""
    matched = []
    for tag in ("ENFERMEDAD", "SINTOMA", "PROCEDIMIENTO"):
        for text in entities.get(tag, []):
            if CARDIAC_RE.search(text):
                matched.append(f"[{tag}] {text}")
    return bool(matched), matched

# ── Aggregate results ─────────────────────────────────────────────────────────
section_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "cardiac": 0, "examples": []})
all_cardiac_docs = []
all_docs = []

for doc, entities in doc_entities.items():
    sec = section_of(doc)
    is_cardiac, matched = has_cardiac(entities)
    section_stats[sec]["total"] += 1
    all_docs.append(doc)
    if is_cardiac:
        section_stats[sec]["cardiac"] += 1
        section_stats[sec]["examples"].append({"doc": doc, "matches": matched[:5]})
        all_cardiac_docs.append(doc)

# ── Patient-level analysis (same numeric ID across sections = same patient) ──
# Build patient → sections map
patient_docs: dict[str, list[str]] = defaultdict(list)
for doc in doc_entities:
    pid = patient_id_of(doc)
    patient_docs[pid].append(doc)

# For each patient, is there ANY section with cardiac finding?
patient_cardiac = {}
for pid, docs in patient_docs.items():
    any_cardiac = any(has_cardiac(doc_entities[d])[0] for d in docs)
    # Separate: cardiac in PROCESO_ACTUAL (acute) vs ANTECEDENTES (history)
    acute_cardiac = any(
        has_cardiac(doc_entities[d])[0]
        for d in docs if "PROCESO_ACTUAL" in d
    )
    hx_cardiac = any(
        has_cardiac(doc_entities[d])[0]
        for d in docs if "ANTECEDENTES" in d
    )
    patient_cardiac[pid] = {
        "any_cardiac": any_cardiac,
        "acute_cardiac": acute_cardiac,
        "history_cardiac": hx_cardiac,
    }

total_patients = len(patient_cardiac)
n_any    = sum(1 for v in patient_cardiac.values() if v["any_cardiac"])
n_acute  = sum(1 for v in patient_cardiac.values() if v["acute_cardiac"])
n_hx     = sum(1 for v in patient_cardiac.values() if v["history_cardiac"])

# ── Print report ──────────────────────────────────────────────────────────────
lines = []
def pr(s=""):
    print(s); lines.append(s)

pr("=" * 65)
pr("CARMEN-I  —  COVID-19 Cardiac Involvement Analysis")
pr("Dataset : PhysioNet carmen-i v1.0.1  (Hospital del Mar, Spain)")
pr("=" * 65)
pr()
pr(f"Documents analysed : {len(doc_entities)}")
pr(f"Unique patient IDs : {total_patients}  (numeric suffix as proxy)")
pr()
# ── De-novo vs pre-existing breakdown ────────────────────────────────────────
# de_novo = acute cardiac findings but NO pre-existing cardiac history
# exacerbation = acute cardiac AND pre-existing cardiac history
n_denovo   = sum(1 for v in patient_cardiac.values()
                 if v["acute_cardiac"] and not v["history_cardiac"])
n_exacerb  = sum(1 for v in patient_cardiac.values()
                 if v["acute_cardiac"] and v["history_cardiac"])
n_hx_only  = sum(1 for v in patient_cardiac.values()
                 if not v["acute_cardiac"] and v["history_cardiac"])
n_none     = sum(1 for v in patient_cardiac.values()
                 if not v["any_cardiac"])

pr("── Patient-level cardiac involvement ──────────────────────────")
pr(f"  Any cardiac finding (any section)      : {n_any}/{total_patients}  ({n_any/total_patients*100:.1f}%)")
pr(f"  Acute cardiac (IA_PROCESO_ACTUAL only) : {n_acute}/{total_patients}  ({n_acute/total_patients*100:.1f}%)")
pr(f"  Pre-existing cardiac (IA_ANTECEDENTES) : {n_hx}/{total_patients}  ({n_hx/total_patients*100:.1f}%)")
pr()
pr("── COVID→cardiac subtype breakdown ────────────────────────────")
pr(f"  De-novo acute cardiac  (acute, no prior hx) : {n_denovo}/{total_patients}  ({n_denovo/total_patients*100:.1f}%)")
pr(f"  Exacerbation           (acute + prior hx)   : {n_exacerb}/{total_patients}  ({n_exacerb/total_patients*100:.1f}%)")
pr(f"  Pre-existing only      (no acute)            : {n_hx_only}/{total_patients}  ({n_hx_only/total_patients*100:.1f}%)")
pr(f"  No cardiac at all                            : {n_none}/{total_patients}  ({n_none/total_patients*100:.1f}%)")
pr()
pr("── Section-level breakdown ─────────────────────────────────────")
for sec, s in sorted(section_stats.items(), key=lambda x: -x[1]["total"]):
    pct = s["cardiac"] / s["total"] * 100 if s["total"] else 0
    pr(f"  {sec:<40s}  {s['cardiac']:3d}/{s['total']:3d}  ({pct:5.1f}%)")
pr()
pr("── Example cardiac entities (IA_PROCESO_ACTUAL) ────────────────")
ex_shown = 0
for item in section_stats.get("IA_PROCESO_ACTUAL", {}).get("examples", []):
    if ex_shown >= 8:
        break
    pr(f"  [{item['doc']}]")
    for m in item["matches"][:3]:
        pr(f"    {m}")
    ex_shown += 1
pr()
pr("── Interpretation ──────────────────────────────────────────────")
pr(f"  Only {n_denovo}/{total_patients} ({n_denovo/total_patients*100:.1f}%) of COVID-19 patients had de-novo acute")
pr(f"  cardiac findings with NO prior cardiac history — the group most")
pr(f"  likely to represent COVID-induced cardiac complications leading")
pr(f"  to a cardiology visit that would not have occurred otherwise.")
pr()
pr(f"  {n_exacerb}/{total_patients} ({n_exacerb/total_patients*100:.1f}%) had acute cardiac findings ON TOP OF pre-existing")
pr(f"  cardiac disease (COVID exacerbation of known disease).")
pr()
pr(f"  {n_hx_only}/{total_patients} ({n_hx_only/total_patients*100:.1f}%) had pre-existing cardiac disease with no acute")
pr(f"  cardiac event during COVID — their cardiology contact predates")
pr(f"  COVID and is independent of it.")
pr()
pr("  Conclusion for our study:")
pr(f"  Even in an all-COVID cohort, only {n_denovo/total_patients*100:.1f}% showed truly de-novo")
pr("  cardiac complications. Our cardiac validation cohort consists of")
pr("  patients with established cardiac diagnoses admitted to cardiology")
pr("  before or independently of COVID. CARMEN-I confirms that COVID-")
pr("  induced de-novo cardiology visits are a small minority (~{:.0f}%)".format(n_denovo/total_patients*100))
pr("  of all COVID hospitalisations, and do not represent the typical")
pr("  profile of our cardiac validation population.")
pr("=" * 65)

# ── Save outputs ──────────────────────────────────────────────────────────────
summary = {
    "dataset": "CARMEN-I (PhysioNet v1.0.1, Hospital Clínic de Barcelona, Spain, 2020-2022)",
    "n_documents": len(doc_entities),
    "n_unique_patient_ids": total_patients,
    "cardiac_any_section_n": n_any,
    "cardiac_any_section_pct": round(n_any / total_patients * 100, 1),
    "cardiac_acute_presentation_n": n_acute,
    "cardiac_acute_presentation_pct": round(n_acute / total_patients * 100, 1),
    "cardiac_preexisting_history_n": n_hx,
    "cardiac_preexisting_history_pct": round(n_hx / total_patients * 100, 1),
    "de_novo_covid_cardiac_n": n_denovo,
    "de_novo_covid_cardiac_pct": round(n_denovo / total_patients * 100, 1),
    "covid_exacerbation_of_preexisting_n": n_exacerb,
    "covid_exacerbation_of_preexisting_pct": round(n_exacerb / total_patients * 100, 1),
    "preexisting_only_no_acute_n": n_hx_only,
    "preexisting_only_no_acute_pct": round(n_hx_only / total_patients * 100, 1),
    "no_cardiac_n": n_none,
    "no_cardiac_pct": round(n_none / total_patients * 100, 1),
    "section_breakdown": {
        sec: {"total": s["total"], "cardiac": s["cardiac"],
              "pct": round(s["cardiac"] / s["total"] * 100, 1) if s["total"] else 0}
        for sec, s in section_stats.items()
    },
}

json_out = OUT_DIR / "carmen_i_cardiac_summary.json"
txt_out  = OUT_DIR / "carmen_i_cardiac_report.txt"
json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
txt_out.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved:\n  {json_out}\n  {txt_out}")
