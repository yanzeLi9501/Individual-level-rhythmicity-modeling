# Individual-level Rhythmicity Modeling

This repository is a scripts-only snapshot for the NC revision analyses. Raw
patient records, public-database extracts, intermediate CSV/JSON tables, figures,
documents, and binary outputs are intentionally excluded.

## Contents

```text
NC_revision/
  *.py
    Main NC revision analysis, validation, audit, and figure-generation scripts.

  RebuildGoal/
    Additional reviewer-response sensitivity analyses and policy-volume checks.

  RebuildRevision/scripts/
    Rebuild pipeline scripts, shared helpers, tables, main/supplementary figures,
    LGDI influenza-reference analyses, multiplicity correction, and expanded
    XGBoost validation experiments.
```

## Data Policy

No raw data are included in this repository. In particular, this snapshot does
not contain:

- private hospital EHR records or exported database folders
- public database extracts from PhysioNet or other providers
- derived CSV/JSON/TSV tables
- figures, DOCX files, PDFs, archives, caches, or model artifacts

Scripts that originally used local absolute paths have been changed to use the
repository-relative layout or placeholder private-data locations such as:

```text
data/private/readmission_output/
data/private/healthline/
external_data/physionet/
```

To reproduce an analysis, provide the required source data locally at the
expected placeholder paths, or update the path constants in the relevant script
for your own environment. Do not commit those data files.

## Notes

The original relative script layout is preserved under `NC_revision/` so that
path logic based on `Path(__file__)` remains close to the revision workspace.
Only executable Python scripts and this README are tracked.
