# EDA Summary

## 1. Dataset Overview

- **101,766 rows** (encounters), **50 columns**
- **71,518 unique patients** — 16,773 patients have multiple encounters (max: 40 encounters/patient)

## 2. Target Variable (`readmitted`)

| Category | Count | % |
|----------|------:|----:|
| NO (not readmitted) | 54,864 | 53.9% |
| >30 days | 35,545 | 34.9% |
| <30 days | 11,357 | 11.2% |

Binary target (readmitted within 30 days): **11.2%** — confirms the ~11% class imbalance mentioned in the proposal.

## 3. Missing Values

| Feature | Missing | % |
|---------|--------:|----:|
| weight | 98,569 | 96.9% |
| medical_specialty | 49,949 | 49.1% |
| payer_code | 40,256 | 39.6% |
| race | 2,273 | 2.2% |
| diag_3 | 1,423 | 1.4% |
| diag_2 | 358 | 0.4% |
| diag_1 | 21 | 0.0% |

**Key finding:** `weight` is nearly entirely missing (96.9%) — essentially unusable. `medical_specialty` and `payer_code` are ~40–50% missing.

## 4. Numeric Features

| Feature | Mean | Median | Min | Max |
|---------|-----:|-------:|----:|----:|
| time_in_hospital | 4.4 | 4 | 1 | 14 |
| num_lab_procedures | 43.1 | 44 | 1 | 132 |
| num_procedures | 1.3 | 1 | 0 | 6 |
| num_medications | 16.0 | 15 | 1 | 81 |
| number_outpatient | 0.37 | 0 | 0 | 42 |
| number_emergency | 0.20 | 0 | 0 | 76 |
| number_inpatient | 0.64 | 0 | 0 | 21 |
| number_diagnoses | 7.4 | 8 | 1 | 16 |

Prior visit counts (outpatient/emergency/inpatient) are heavily right-skewed — most patients have 0.

## 5. Demographics

- **Race:** Caucasian 74.8%, African American 18.9%, Hispanic 2.0%, Asian 0.6%
- **Gender:** Female 53.8%, Male 46.2%
- **Age:** predominantly 50–80 years (65,807 encounters, 64.7%)

## 6. HbA1c (`A1Cresult`)

The EDA reveals `A1Cresult` has 4 values: `None`, `>8`, `Norm`, `>7`. The **84,748 records** with value `None` (**83.3%**) represent "not tested" — even higher than the proposal's stated ~70%.

## 7. Medication Flags

- **insulin:** 53.4% of encounters involve insulin (dominant medication)
- **metformin:** 19.6%
- 5 medications have **zero or near-zero** usage: `examide`, `citoglipton`, `troglitazone`, `metformin-pioglitazone`, `glimepiride-pioglitazone` — candidates for removal.

## 8. ICD-9 Diagnosis Cardinality

- `diag_1`: **717** unique codes
- `diag_2`: **749** unique codes
- `diag_3`: **790** unique codes

Confirms the proposal's statement of 700+ high-cardinality codes.

## 9. 30-Day Readmission Rate by Group

**By Race:**

| Race | Rate | n |
|------|-----:|----:|
| Caucasian | 11.3% | 76,099 |
| African American | 11.2% | 19,210 |
| Hispanic | 10.4% | 2,037 |
| Asian | 10.1% | 641 |

Rates are fairly uniform across race (~10–11%), which is worth noting for the fairness discussion.

**By Age:**

| Age | Rate |
|-----|-----:|
| [20-30) | 14.2% (highest) |
| [80-90) | 12.1% |
| [50-60) | 9.7% (lowest among adults) |

## 10. Key Findings for the Proposal

1. **`weight` should be dropped** (96.9% missing) — not mentioned in current proposal
2. **HbA1c missingness is 83.3%**, not ~70% as stated — need to correct
3. **`medical_specialty` (49.1% missing) and `payer_code` (39.6% missing)** need mitigation strategies — not discussed in proposal
4. **Duplicate patients** (16,773 patients with multiple encounters) — risk of data leakage if the same patient appears in both train and test sets. Must split by `patient_nbr`, not by encounter. This is not mentioned in the proposal.
5. **5 near-zero-variance medication columns** can be dropped
6. **Readmission rates are fairly consistent across race/gender** — good baseline for fairness but small sample sizes for Asian/Hispanic groups may cause noisy fairness metrics
