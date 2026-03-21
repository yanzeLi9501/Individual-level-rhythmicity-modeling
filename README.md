# Individual-level Rhythmicity Modeling

**Predicting Readmission Patterns in Cancer Patients Using Historical Electronic Health Records**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository implements a machine-learning pipeline for **individual-level rhythmicity modeling** of hospital readmission patterns in cancer patients. By leveraging longitudinal electronic health records (EHR) spanning 2012–2020, we construct patient-specific sequential features and train gradient-boosted models to predict:

1. **Readmission interval (Gap Days)** — when a patient is likely to return  
2. **Length of Stay (LOS)** — how long their next hospitalization will last

The approach captures individual temporal "rhythms" in healthcare utilization through exponential moving averages, trend indicators, acceleration features, and rolling statistics over each patient's full admission history.

## Key Results

| Prediction Task | R² | MAE (days) | RMSE (days) | Sample Size |
|---|---|---|---|---|
| Readmission Interval | 0.925 ± 0.028 | 0.53 ± 0.10 | 1.26 | 299 encounters |
| Length of Stay | 0.556 ± 0.024 | 0.94 ± 0.04 | 1.41 | 3,368 encounters |

## Dataset

- **Source**: 37,384 de-identified inpatient JSON records from a tertiary hospital EHR system
- **Patients**: 32,057 unique patients; 18,956 follow-up eligible encounters across 9,482 patients
- **Features**: 150 predictive features across 6 categories (demographics, lab values, utilization, sequential history, derived interactions, socio-economic context)
- **External Validation**: MIMIC-IV and SCH cohorts

> **Note**: Raw patient data is not included due to privacy regulations. The pipeline code and figure generation scripts are fully reproducible given appropriately formatted input data.

## Repository Structure

```
Individual-level-rhythmicity-modeling/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── fig_config.py              # Shared configuration (paths, palettes, save functions)
├── src/
│   ├── pipeline/                   # Data processing & model training
│   │   ├── data_processing.py              # JSON → structured CSV extraction
│   │   ├── readmission_step1_extract.py    # Step 1: EHR data extraction
│   │   ├── readmission_step2_features.py   # Step 2: Feature engineering + city merge
│   │   ├── readmission_step3a_xgb_feature_select.py  # Step 3a: XGBoost feature selection
│   │   ├── readmission_step3b_multi_model.py          # Step 3b: Multi-model comparison
│   │   ├── readmission_history_train.py    # Sequential history feature construction
│   │   └── readmission_full_train.py       # Final model training
│   ├── figures/                    # Main manuscript figure generation
│   │   ├── gen_fig_new1.py         # Figure 1: Epidemiology & behavioral patterns
│   │   ├── gen_fig_new2.py         # Figure 2: Sentinel discovery analysis
│   │   ├── gen_fig_new3.py         # Figure 3: Early warning system
│   │   ├── gen_fig_new4.py         # Figure 4: Post-pandemic validation
│   │   ├── gen_fig_new5.py         # Figure 5: Model generalizability
│   │   ├── gen_fig1.py             # Figure 6: Model performance (scatter, CV, residuals)
│   │   ├── gen_fig2.py             # Figure 7: Disease-stratified analysis
│   │   ├── gen_paper_v2.py         # Manuscript assembly (Word/DOCX)
│   │   └── run_all_figures.bat     # Batch runner for all figures
│   └── supplementary/             # Supplementary figure generation
│       ├── gen_supp_figs.py        # Figures S1–S6: Data overview, model comparison
│       ├── gen_supp_ablation.py    # Figure S12: Feature dimension ablation
│       ├── gen_supp_augmented_ref.py  # Figure S13: Augmented sentinel analysis
│       ├── gen_supp_extended.py    # Figure S15: Extended supplementary analysis
│       ├── gen_supp_leadtime.py    # Figure S9B: Lead-time threshold calibration
│       ├── gen_supp_model_extended.py  # Figure S11: Extended model performance (8 panels)
│       └── gen_supp_sch_rdi.py     # Figure S14: SCH RDI workflow validation
└── docs/
    ├── METHODS_DOCUMENTATION.md    # Full methods & reproducibility guide
    └── paper_framework_npjDM.md    # Paper framework for npj Digital Medicine
```

## Pipeline

### Step 1: Data Extraction
```bash
python src/pipeline/readmission_step1_extract.py
```
Extracts structured data from JSON EHR files → `all_admissions.csv`

### Step 2: Feature Engineering
```bash
python src/pipeline/readmission_step2_features.py
```
Constructs clinical features, merges city-level socio-economic data → `train_data.csv`, `structured_features.csv`

### Step 3: History Feature Construction
```bash
python src/pipeline/readmission_history_train.py
```
Builds sequential features (lags, rolling stats, EMA, trends) from patient admission histories → `history_features.csv`

### Step 4: Model Training
```bash
python src/pipeline/readmission_full_train.py
```
Trains XGBoost models with GPU acceleration using optimized hyperparameters from 3-stage grid search.

### Step 5: Figure Generation
```bash
cd src/figures && run_all_figures.bat
```
Generates all manuscript and supplementary figures in PNG, PDF, and TIF formats.

## Feature Categories

| Category | Count | Description |
|---|---|---|
| Demographics | 8 | Age, sex, marital status, insurance, admission pathway |
| Laboratory | 32 | 16 lab panels × current + delta values |
| Healthcare Utilization | 22 | 11 cost categories, procedure/medication counts |
| Sequential History | 48 | Lag, rolling stats, EMA, trends over full admission history |
| Derived Interactions | 20 | Log transforms, ratios, cross-term features |
| Socio-economic Context | 30 | City-level GDP, healthcare density, insurance coverage |

## Model Configuration

| Parameter | Gap Days Model | LOS Model |
|---|---|---|
| Minimum visit_order | ≥ 20 | ≥ 5 |
| Target cap | 10 days | 7 days |
| n_estimators | 500 | 800 |
| max_depth | 4 | 5 |
| learning_rate | 0.03 | 0.005 |
| subsample | 1.0 | 0.8 |
| colsample_bytree | 0.9 | 0.4 |
| min_child_weight | 10 | 30 |
| Evaluation | 5-fold CV | 5-fold CV |

## Requirements

```
python>=3.10
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
python-docx>=0.8
openpyxl>=3.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## External Validation

The model was validated on two external cohorts:
- **MIMIC-IV**: publicly available ICU database (PhysioNet)
- **SCH**: independent hospital cohort with RDI (Relative Dose Intensity) workflow analysis


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
