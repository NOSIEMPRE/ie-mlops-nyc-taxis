# 修订对照表 (Change Log)

> 以下为逐点修改说明，随后附完整修订稿。


| #   | 位置         | 原文                                                                                                                          | 问题                                                                            | 修改后                                                                                                                                                                                                                                                                                                                     |
| --- | ---------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 1.2        | "at the moment a diabetic patient is about to be discharged, gives the care team a risk score…"                             | 与 4.1 批量预测矛盾（用户已自行修改）                                                         | 采用用户新版本：预计算 + 出院时已就绪                                                                                                                                                                                                                                                                                                    |
| 2   | 1.3 第2条    | "Make sure the cost of interventions is less (attention time, follow-up calls…)."                                           | "less" than what? 比较对象缺失                                                      | "Ensure the cost of targeted interventions (e.g., follow-up calls, care-coordinator time) remains below the financial savings from avoided readmissions and HRRP penalties."                                                                                                                                            |
| 3   | 2.2 开头     | "Given **close than** 11% class imbalance"                                                                                  | 笔误                                                                            | "Given **approximately** 11% class imbalance"                                                                                                                                                                                                                                                                           |
| 4   | 2.2 第3条指标  | "**Recall at fixed precision**: … care coordinators can only handle a certain number of flags per day (e.g., top 20%)."     | 指标名称与描述矛盾：固定的是容量（top K%），不是 precision                                         | 改为 "**Recall at K** (operational metric): of all patients who are truly high-risk, how many does the model catch when we can only flag the top K % of discharges per day (e.g., K = 20), reflecting realistic care-coordinator capacity."                                                                               |
| 5   | 2.3 可扩展性   | "being able to grow regardless of traffic volume, model complexity, or number of models"                                    | 本项目是单一批量模型，讨论"流量/模型数量"扩展不太贴切                                                  | "Scalability: batch pipeline should handle growing patient volumes (e.g., onboarding additional hospitals) without manual re-engineering."                                                                                                                                                                              |
| 6   | 3.1 数据时效   | 数据 1999–2008，未提及局限性                                                                                                         | 数据距今近 20 年，读者会质疑泛化能力                                                          | 末尾新增一句："A known limitation is that the data spans 1999–2008; clinical protocols and coding practices have since evolved. We treat this dataset as a proof-of-concept and discuss generalisability in Section 4.3."                                                                                                      |
| 7   | 4.1        | "every night you run the model on that day's discharges and deliver a risk list to care coordinators the next morning"      | (a) 与新 1.2 矛盾：应对在院患者预计算而非已出院患者；(b) 使用第二人称 "you"，不够正式                          | 见修订稿 4.1                                                                                                                                                                                                                                                                                                                |
| 8   | 4.2        | "orchestrated and controlled in GitHub"                                                                                     | GitHub 是代码托管 + CI/CD，不是流水线编排器                                                 | "Pipeline code and configuration are version-controlled in GitHub. A CI/CD workflow (GitHub Actions) runs unit and integration tests on every pull request. For pipeline orchestration, we plan to use a lightweight scheduler (e.g., GitHub Actions scheduled workflows or Prefect) to coordinate nightly batch runs." |
| 9   | 4.4        | "recall and False Positive Rate … separated by race and payer type. A demographic gap exceeding 5% triggers a model review" | 4.4 说"triggers a model review"（仅审查）；2.3 说"block the deployment"（阻止部署），严重程度不一致 | 统一为 4.4："A demographic gap exceeding 5 pp in recall or FPR triggers an automatic deployment block; the model must pass a fairness review before promotion."                                                                                                                                                             |
| 10  | References | Lundberg & Lee (2017) 在正文中从未被引用                                                                                             | 悬空引用                                                                          | 在 2.3 Interpretability 中增加引用 "(Lundberg & Lee, 2017)"                                                                                                                                                                                                                                                                   |


---

# Hospital Readmission Risk Prediction: An End-to-End ML System for Proactive Patient Care

**Submission Date:** 8 March 2026  
**Dataset:** Diabetes 130-US Hospitals — UCI ML Repository (id=296) — Strack et al., 2014  
**Document:** Group Project Checkpoint Proposal  
**Team Members:** Marian, Marco, Yaxin, Lorenz, Jorge, and Omar

---

## Table of Contents

1. [Business Problem](#1-business-problem)
  - 1.1 Context
  - 1.2 Proposed Solution
  - 1.3 Business Objectives
2. [Clear Objectives](#2-clear-objectives)
  - 2.1 ML Task
  - 2.2 Evaluation Metrics
  - 2.3 System Requirements
3. [Data and Modeling Strategy](#3-data-and-modeling-strategy)
  - 3.1 Dataset
  - 3.2 Key Features
  - 3.3 Data Challenges and Mitigations
  - 3.4 Modeling Approach
4. [System-Level Considerations](#4-system-level-considerations)
  - 4.1 Serving Architecture
  - 4.2 Pipeline and Versioning
  - 4.3 Monitoring and Drift Detection
  - 4.4 Fairness Monitoring
  - 4.5 Privacy and Compliance
5. [References](#references)

---

## 1. Business Problem

### 1.1 Context

Hospitals operating under value-based care models face financial penalties if too many of their patients are readmitted within 30 days of discharge. In the US, the Hospital Readmissions Reduction Program (HRRP) penalizes hospitals up to 3% of all Medicare reimbursements (CMS, 2023). Diabetic patients have high readmission rates because of complex comorbidities and challenging medication management, making this a costly yet addressable problem for hospitals. Beyond finances, readmissions are a sign of failures in care continuity that directly harm patients.

### 1.2 Proposed Solution

We want to build a Machine Learning system that gives the care team a risk score at the point of discharge, indicating whether a patient is likely to be readmitted within 30 days. Scores are pre-computed nightly on all active diabetic inpatients, ensuring they are ready and available when the discharge decision is made, enabling doctors to intervene with targeted follow-up before the patient leaves the hospital. The system will integrate into the discharge workflow, providing a risk score and its key clinical drivers for every diabetic discharge event.

### 1.3 Business Objectives

- Reduce the 30-day readmission rate by at least 10% among flagged high-risk patients who receive care coordinator intervention.
- Ensure the cost of targeted interventions (e.g., follow-up calls, care-coordinator time) remains below the financial savings from avoided readmissions and HRRP penalties.
- Improve care teams' efficiency by prioritizing the patients most likely to benefit from follow-up, given constrained clinical capacity.

Primary stakeholders: hospital finance teams (penalty avoidance), care coordinators (daily users of the risk score), and attending physicians (discharge decision support).

---

## 2. Clear Objectives

### 2.1 ML Task

Binary classification: given a patient's clinical record at discharge, predict 30-day readmission (yes/no). The original dataset has three possible outcomes: ≤30 days, >30 days, and no readmission. We will collapse that into just two: readmitted within 30 days or not readmitted. We are doing this because 30 days is the window that matters for the HRRP penalty, and it is what the care coordinators/teams can make decisions with.

### 2.2 Evaluation Metrics

Given approximately 11% class imbalance, if the model simply predicted "nobody gets readmitted", it would be 89% accurate—which is useless. We will use the following metrics:

- **AUROC**: model selection metric; measures how well the model ranks high-risk patients above low-risk ones across all possible thresholds (comparing models against each other).
- **Precision-Recall AUC**: similar to AUROC but focuses specifically on the minority class (readmitted within 30 days). It is more informative for imbalanced data.
- **Recall at K** (operational metric): of all patients who are truly high-risk, how many does the model catch when we can only flag the top K% of discharges per day (e.g., K = 20), reflecting realistic care-coordinator capacity.

(Huyen, 2022)

### 2.3 System Requirements

- **Reliability:** the system should continue to perform the correct function at the desired level of performance regardless of adversity.
- **Maintainability:** versioned models and pipeline code; reproducible runs via MLflow experiment tracking.
- **Scalability:** the batch pipeline should handle growing patient volumes (e.g., onboarding additional hospitals) without manual re-engineering.
- **Interpretability:** every risk score comes with an explanation (using SHAP values) so doctors know why a patient was flagged (Lundberg & Lee, 2017).
- **Fairness:** the model's performance is checked across demographic and socioeconomic groups; if there is a gap bigger than 5 percentage points, the system blocks deployment until a fairness review is passed (we want to avoid bias).

(Huyen, 2022; IE University, 2026)

---

## 3. Data and Modeling Strategy

### 3.1 Dataset

The dataset used in this project is the Diabetes 130-US Hospitals dataset (UCI ML Repository, id=296) (UCI ML Repository, 2014), introduced by Strack et al. (2014). It was extracted from the Health Facts database (Cerner Corporation), a national clinical data warehouse that covers 74 million unique encounters across 17 million unique patients. The final dataset has 101,766 inpatient diabetic encounters from 130 US hospitals and integrated delivery networks from 1999 to 2008, with 50 features per encounter. Only inpatients with diabetic admissions of 1–14 days duration with laboratory tests performed and medications administered are included. Encounters resulting in patient death or transfer were excluded to ensure readmission was a possible outcome. A known limitation is that the data spans 1999–2008; clinical protocols and coding practices have since evolved. We treat this dataset as a proof-of-concept and discuss generalisability in Section 4.3.

An initial exploratory data analysis reveals several properties of the dataset that directly shape our preprocessing, splitting, and modeling decisions:

**Target variable.** The original `readmitted` column has three classes: NO (54,864; 53.9%), >30 days (35,545; 34.9%), and <30 days (11,357; 11.2%). For our binary task, the positive class (<30 days) represents 11.2% of encounters, confirming substantial class imbalance.

**Patient overlap.** The 101,766 encounters correspond to only 71,518 unique patients; 16,773 patients appear in multiple encounters (up to 40 per patient). This means that a naive random split would leak patient-level information between training and test sets. We therefore adopt a patient-level split strategy (Section 3.4).

**Demographics.** The cohort skews older (64.7% of encounters are aged 50–80) and is predominantly Caucasian (74.8%), with African American patients comprising 18.9%. Gender is roughly balanced (53.8% female, 46.2% male). The 30-day readmission rate is fairly uniform across race (10–11%) and gender (~11%), which provides a reasonable baseline for fairness evaluation, though small sample sizes for Asian (641) and Hispanic (2,037) subgroups may produce noisy fairness estimates.

### 3.2 Key Features

- **Demographics:** age (10-year bins), gender, race
- **Admission context:** admission type, discharge disposition, admission source (mapped via IDS codes)
- **Clinical utilization:** number of lab procedures (mean 43.1), number of medications (mean 16.0), number of procedures (mean 1.3), number of diagnoses (mean 7.4), prior inpatient/outpatient/emergency visits in the preceding year (heavily right-skewed; most patients have 0)
- **Diagnoses:** primary/secondary/tertiary ICD-9 codes (717–790 unique codes each) → mapped to CCS groupings (~9 clinical categories)
- **Lab results:** HbA1c result (4 values: None/Norm/>7/>8), max glucose serum (4 values: None/Norm/>200/>300)
- **Medications:** 23 binary/categorical medication change flags. Insulin dominates (53.4% of encounters), followed by metformin (19.6%), glipizide (12.5%), and glyburide (10.5%). Five medications have near-zero variance (examide, citoglipton, troglitazone, metformin-pioglitazone, glimepiride-pioglitazone) and will be dropped.
- **Medication change indicators:** `change` (whether any medication was changed: 46.2% yes) and `diabetesMed` (whether any diabetes medication was prescribed: 77.0% yes)
- **Engineered (planned):** comorbidity index (Charlson approximation), medication burden score, care intensity composite

### 3.3 Data Challenges and Mitigations


| Challenge                                                                           | Mitigation                                                                                                |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Class imbalance (11.2% positive)                                                    | Stratified splits; SMOTE on training only; PR-AUC as primary metric; threshold tuned to clinical capacity |
| Patient-level data leakage (16,773 patients with multiple encounters)               | Split by `patient_nbr`, not by encounter, to ensure no patient appears in both train and test sets        |
| `weight` nearly entirely missing (96.9%)                                            | Drop the column entirely                                                                                  |
| `medical_specialty` missing (49.1%)                                                 | Impute as "Unknown"; group rare specialties into an "Other" category                                      |
| `payer_code` missing (39.6%)                                                        | Impute as "Unknown"; evaluate predictive value before inclusion                                           |
| HbA1c missingness (83.3%, higher than the ~70% initially estimated)                 | Treat "None" as an informative "not tested" category rather than imputing a clinical value                |
| ICD-9 high cardinality (700+ unique codes per diagnosis slot) (Strack et al., 2014) | Map to CCS groupings (~9 categories) instead of one-hot-encoding; target-encode top predictive groups     |
| Near-zero-variance medication columns (5 columns with ≤3 non-"No" records)          | Drop these columns to reduce noise                                                                        |
| 30-day label delay                                                                  | Actual outcomes arrive with a 30-day delay; monitor data and prediction drift in the meantime             |


### 3.4 Modeling Approach

- **Baseline — Logistic Regression:** calibrated, interpretable; sets the minimum performance bar.
- **Primary — XGBoost / LightGBM:** the standard for structured tabular clinical data; handles class imbalance natively (Huyen, 2022).

**Data splitting.** We will split by `patient_nbr` into train (70%), validation (15%), and test (15%) sets, ensuring no patient overlap across splits. Stratification on the binary target preserves the 11.2% positive rate in each subset.

**Hyperparameter tuning.** We plan to use Optuna for Bayesian hyperparameter search over key parameters (learning rate, max depth, number of estimators, sub-sample ratio, class weight / scale_pos_weight). Each trial will be evaluated on the validation set using PR-AUC as the primary objective.

All experiments will be tracked in MLflow (hyperparameters, AUROC, PR-AUC, F1 at multiple thresholds). A model is promoted to the registry only if it exceeds the current champion on the held-out validation set.

---

## 4. System-Level Considerations

### 4.1 Serving Architecture

Batch prediction is the appropriate serving pattern. Every night, the system scores all active diabetic inpatients (and those flagged as likely to be discharged soon). Risk scores are stored and made available for lookup, so they are ready when the discharge decision is made the following day. Care coordinators also receive a prioritized list of high-risk patients for post-discharge follow-up planning. This approach is simpler, cheaper, and more reliable than building a real-time inference API (Huyen, 2022).

### 4.2 Pipeline and Versioning

Pipeline code and configuration are version-controlled in GitHub. A CI/CD workflow (GitHub Actions) runs unit and integration tests on every pull request. For pipeline orchestration, we plan to use a lightweight scheduler (e.g., GitHub Actions scheduled workflows or Prefect) to coordinate nightly batch runs. Feature definitions are shared between training and inference pipelines to prevent training-serving skew, a common production failure mode (Huyen, 2022).

### 4.3 Monitoring and Drift Detection

Patient populations are not stationary: COVID-19, seasonal illness cycles, and clinical protocol changes can all shift the feature distribution and the label relationship. This is especially relevant given that our training data (1999–2008) predates many modern clinical practices; monitoring is therefore critical once the system is deployed on contemporary data. Three monitoring layers:

- **Data drift:** PSI (Population Stability Index) on key input features — are the input features changing? (Population Stability Index, 2025).
- **Prediction drift:** monitor the distribution of output risk scores for unexpected shifts.
- **Outcome drift:** track actual 30-day readmission rate with a 30-day lag as actual-outcome feedback — is the actual readmission rate changing? (Huyen, 2022).

Drift alerts trigger automated retraining on a rolling 12-month window, with a validation gate before promotion. Target: MLOps Maturity Level 1 (automated CT + CD, passive monitoring) as baseline; Level 2 (active auto-trigger) as stretch goal.

### 4.4 Fairness Monitoring

To avoid potential bias, recall and False Positive Rate are continuously monitored, separated by race and payer type. Our EDA shows that baseline 30-day readmission rates are relatively consistent across racial groups (10–11%), but sample sizes for Asian (641) and Hispanic (2,037) subgroups are substantially smaller than for Caucasian (76,099) and African American (19,210) patients. This means fairness metrics for smaller groups will have wider confidence intervals, which we will account for by using bootstrapped significance tests rather than raw point estimates. A demographic gap exceeding 5 percentage points in recall or FPR triggers an automatic deployment block; the model must pass a fairness review before promotion to production.

### 4.5 Privacy and Compliance

Healthcare data is subject to strict regulatory requirements. Although the dataset used in this project is a publicly available, de-identified research dataset, the production system we design must be HIPAA-compliant. Key considerations include: (1) all patient identifiers (`encounter_id`, `patient_nbr`) are used solely for deduplication and patient-level splitting during development, and will not be included as model features; (2) model inputs and outputs will be logged for audit purposes with access restricted to authorised personnel; (3) the serving layer will enforce role-based access controls so that only care coordinators and attending physicians can view risk scores for their assigned patients.

---

## References

- CMS (2023). Hospital Readmissions Reduction Program (HRRP). CMS.gov
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media
- IE University (2026). Machine Learning Operations (Course Sessions 1–5)
- Lundberg, S. M. & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Population Stability Index. (August 13, 2025). GeeksForGeeks. [https://www.geeksforgeeks.org/data-science/population-stability-index-psi/](https://www.geeksforgeeks.org/data-science/population-stability-index-psi/)
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, vol. 2014, Article ID 781670, 11 pages. [https://doi.org/10.1155/2014/781670](https://doi.org/10.1155/2014/781670)
- UCI Machine Learning Repository (2014). Diabetes 130-US Hospitals. [https://doi.org/10.24432/C5230J](https://doi.org/10.24432/C5230J)

