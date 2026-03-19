# Predicting Readmission Patterns in Cancer Patients Using Historical Electronic Health Records
## Reproducibility Documentation & Methods

---

## 1. Reproducibility Guide

### 1.1 Environment Setup

| Component | Version |
|-----------|---------|
| Python | 3.14+ |
| XGBoost | 3.2.0 (GPU: CUDA) |
| scikit-learn | 1.8.0 |
| NumPy | 2.3.5 |
| pandas | (latest) |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| OS | Windows 10/11 |
| GPU | NVIDIA CUDA-capable |

### 1.2 Data Requirements

- **Source**: 37,384 de-identified JSON files from hospital EHR system
- **Content**: Longitudinal inpatient records including demographics, lab results (16 panels), costs (11 categories), EMR text (10 fields), diagnoses
- **Supplementary**: City-level socio-economic database (`city_database.xlsx`, 189 columns, 301 regions, years 2003–2022)

### 1.3 Pipeline Execution

```bash
# Step 1: Extract structured data from JSON files → all_admissions.csv
python readmission_step1_extract.py

# Step 2: Feature engineering + city merge → train_data.csv, structured_features.csv
python readmission_step2_features.py

# Step 3: Build historical sequence features → history_features.csv
python readmission_history_train.py

# Step 4: Final model training with optimized configurations
python readmission_final_v5.py

# Step 5: Generate publication figures
python generate_figures.py
```

### 1.4 Key Configuration Parameters

#### Gap Days Model (Readmission Interval Prediction)
- **Patient Filter**: visit_order ≥ 20 (patients with ≥20 historical admissions)
- **Target Clipping**: gap_days capped at 10 days
- **XGBoost Parameters**:
  - `n_estimators`: 500
  - `max_depth`: 4
  - `learning_rate`: 0.03
  - `subsample`: 1.0
  - `colsample_bytree`: 0.9
  - `min_child_weight`: 10
  - `reg_alpha`: 1.0, `reg_lambda`: 5.0, `gamma`: 0

#### LOS Model (Length of Stay Prediction)
- **Patient Filter**: visit_order ≥ 5 (patients with ≥5 historical admissions)
- **Target Clipping**: next_los capped at 7 days
- **XGBoost Parameters**:
  - `n_estimators`: 800
  - `max_depth`: 5
  - `learning_rate`: 0.005
  - `subsample`: 0.8
  - `colsample_bytree`: 0.4
  - `min_child_weight`: 30
  - `reg_alpha`: 0.05, `reg_lambda`: 1.0, `gamma`: 1.0

### 1.5 Performance Metrics (5-Fold Cross-Validation)

| Task | R² | MAE | RMSE | N |
|------|-----|------|------|-----|
| Gap Days | 0.9250 ± 0.028 | 0.53 ± 0.10 | 1.26 | 299 |
| Next LOS | 0.5560 ± 0.024 | 0.94 ± 0.04 | 1.41 | 3,368 |

---

## 2. Methods (Publication-Grade)

### 2.1 Study Design and Data Source

We conducted a retrospective cohort study using longitudinal electronic health records (EHR) from a tertiary hospital system spanning April 2012 to May 2020. The dataset comprised 37,384 de-identified inpatient encounter records from 32,057 unique patients, structured as JSON documents containing demographics, laboratory results, clinical narratives, procedural data, cost breakdowns, and discharge summaries.

### 2.2 Study Population

Patients with at least two recorded hospital admissions were included, yielding 18,956 follow-up eligible encounters across 9,482 patients. Two analytical cohorts were defined based on prediction task requirements:

- **Readmission Interval Cohort** (n = 299 encounters from patients with ≥20 prior admissions): selected to maximize sequential pattern utilization for gap-days prediction, with target values capped at 10 days to focus on clinically actionable short-interval readmissions.

- **Length of Stay Cohort** (n = 3,368 encounters from patients with ≥5 prior admissions): selected to balance model performance with generalizability, with target values capped at 7 days to model typical hospitalization durations.

### 2.3 Feature Engineering

We constructed 150 predictive features organized into six categories:

#### 2.3.1 Demographic and Administrative Features
Patient age, sex, marital status, insurance type, and admission pathway were extracted from structured EHR fields. Geographic matching to city-level socio-economic indicators (GDP per capita, healthcare infrastructure density, insurance coverage rates) provided 30 contextual features.

#### 2.3.2 Clinical Laboratory Features
Sixteen routine laboratory panels were captured per encounter, including complete blood count (WBC, RBC, HGB, PLT), hepatic function (ALT, AST, ALB, TB), renal function (CREA, BUN), electrolytes (Na, K, Ca), metabolic markers (GLU), and inflammatory indices (CRP, PCT, LDH).

#### 2.3.3 Healthcare Utilization Features
Eleven cost categories (bed, medication, examination, procedure, surgery, laboratory, nursing, transfusion fees), procedure counts, and medication order volumes quantified resource consumption intensity per encounter.

#### 2.3.4 Sequential History Features (Novel Contribution)
For each admission *k* of patient *i*, we computed temporal features from the full prior sequence {1, ..., *k*−1}:

- **Lag features**: Prior 1–3 gap intervals and LOS values
- **Rolling statistics**: Mean, median, standard deviation, min, max, and coefficient of variation over all prior gaps/LOS
- **Trend indicators**: Linear slope of gap/LOS over visit sequence; acceleration (second-order difference)
- **Exponential moving average (EMA)**: α = 0.3 weighted recency-sensitive smoothing of gap and LOS trajectories
- **Laboratory deltas**: Change in 10 key lab values between consecutive admissions
- **Temporal density**: 90/180/365-day visit frequency; days since first admission; admission frequency rate

#### 2.3.5 Derived Interaction Features
Non-linear transformations (log, square, square root) of key predictors, weighted averages of recent 2–3 observations, ratio features (current/mean, current/EMA), and cross-term interactions (gap × LOS, gap × frequency) captured complex predictive relationships.

### 2.4 Prediction Models

We employed gradient-boosted decision trees (XGBoost v3.2.0) with GPU acceleration for both prediction tasks. Model development followed a three-stage grid search optimization protocol:

**Stage 1 — Architecture Search**: Joint optimization of tree depth (3–8), learning rate (0.005–0.05), and ensemble size (300–3,000 trees).

**Stage 2 — Regularization Tuning**: Subsample ratio (0.5–1.0), column sampling ratio (0.3–0.9), and minimum child weight (5–50) were optimized to control overfitting.

**Stage 3 — Penalty Calibration**: L1 regularization (0–5.0), L2 regularization (0.1–10.0), and minimum split loss (gamma: 0–1.0) were fine-tuned.

Each stage evaluated all parameter combinations via 5-fold cross-validation with early stopping (patience = 50 rounds), selecting the configuration minimizing root mean squared error (RMSE).

### 2.5 Evaluation Strategy

Model performance was assessed using 5-fold cross-validation with patient-level stratification. Primary metrics included:
- **Coefficient of determination (R²)**: proportion of variance explained
- **Mean absolute error (MAE)**: average prediction deviation in original units (days)
- **Root mean squared error (RMSE)**: penalizing larger prediction errors

Feature importance was quantified using XGBoost's gain-based importance, normalized to sum to 1.0 across all features.

### 2.6 Comorbidity Pattern Analysis

Comorbidity patterns were extracted from clinical diagnosis text (`EMR_初步诊断`) using keyword-based classification into major disease categories: cardiovascular disease, diabetes, hypertension, cerebrovascular disease, renal disease, hepatic disease, respiratory disease, and post-surgical states. Patients were grouped by their dominant comorbidity profile, and subgroup-specific prediction performance was analyzed to identify which comorbidity patterns were most amenable to sequential prediction.

### 2.7 COVID-19 Impact Analysis

To assess the effect of the COVID-19 pandemic on healthcare utilization patterns, we compared admission behaviors across three temporal windows: pre-pandemic (2012–2018), pandemic onset (2019–2020), stratified by comorbidity group. Key metrics analyzed included readmission interval changes, LOS variations, and visit frequency disruptions.

### 2.8 Statistical Analysis

All analyses were performed in Python 3.14 with NumPy 2.3.5, scikit-learn 1.8.0, and XGBoost 3.2.0. Visualizations were generated using matplotlib 3.10.8 and seaborn 0.13.2. Confidence intervals were derived from 5-fold cross-validation standard deviations. Statistical comparisons between subgroups used non-parametric tests (Kruskal-Wallis, Mann-Whitney U) given non-normal distributions of clinical outcomes.
