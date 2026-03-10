# NYC Taxi Duration Prediction — 02 Line-by-Line Code Explanation

> Notebook: `nyc_taxi_duration_prediction_data_fe.ipynb`
>
> This notebook extends Notebook 01 with two additions:
> 1. **Deterministic sampling (10%)** — shrink the dataset to speed up iteration during development.
> 2. **One-Hot Encoding (OHE) deep dive** — inspect how `DictVectorizer` works under the hood.

---

## Cell 2 — Imports

```python
import warnings
import time

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
```

| Line | Explanation |
|---|---|
| `warnings` | Python standard library for controlling warning display |
| `time` | Python standard library for timing; used later to measure training/prediction speed |
| `numpy` | Numerical computing library providing efficient array operations |
| `pandas` | Data analysis library; core data structure is the DataFrame |
| `seaborn` / `matplotlib` | Visualization libraries; seaborn is a high-level wrapper around matplotlib |
| `DictVectorizer` | Converts a list of dictionaries into a sparse matrix — the key tool for One-Hot Encoding here |
| `LinearRegression` | Ordinary Least Squares linear regression model |
| `root_mean_squared_error` | Computes RMSE (Root Mean Squared Error); available in newer scikit-learn versions |
| `warnings.filterwarnings('ignore')` | Suppress all warnings for cleaner output |
| `sns.set_style('whitegrid')` | Set seaborn chart style to white background with grid lines |

---

## Cell 4 — Define Data URLs

```python
TRAIN_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
VAL_URL   = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet'
```

- **Training set**: NYC yellow taxi trip records from **January 2024**
- **Validation set**: **February 2024**
- **Format**: Parquet — a columnar binary format with built-in compression, much faster to read than CSV for large datasets
- Data sourced from the NYC TLC (Taxi & Limousine Commission) open dataset

---

## Cell 6 — Data Loading & Preprocessing Function

```python
CATEGORICAL = ['PULocationID', 'DOLocationID']

def read_data(url: str) -> pd.DataFrame:
    df = pd.read_parquet(url)
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    return df
```

| Line | Explanation |
|---|---|
| `CATEGORICAL = ['PULocationID', 'DOLocationID']` | Define the two categorical features for OHE: **pickup location ID** and **dropoff location ID** |
| `def read_data(url: str) -> pd.DataFrame:` | Function signature — takes a URL string, returns a DataFrame |
| `df = pd.read_parquet(url)` | Read a Parquet file directly from a remote URL into memory (pandas handles the HTTP download + Parquet parsing automatically) |
| `df['duration'] = ...` | Compute trip duration in minutes: dropoff time − pickup time → convert to seconds (`.dt.total_seconds()`) → divide by 60 |
| `df = df[(df.duration >= 1) & (df.duration <= 60)].copy()` | **Remove outliers**: keep only trips between 1 and 60 minutes. Trips shorter than 1 min are likely cancellations; longer than 60 min are likely data errors. `.copy()` prevents the pandas `SettingWithCopyWarning` |
| `df[CATEGORICAL] = df[CATEGORICAL].astype(str)` | Cast location IDs from integers to strings. **Why?** `DictVectorizer` treats string values as categorical (→ OHE) but treats numeric values as continuous features (passes them through as-is) |

---

## Cell 7 — Load Data & Quick Look

```python
df_train = read_data(TRAIN_URL)
df_val   = read_data(VAL_URL)

print(f"Train rows: {len(df_train):,};  Val rows: {len(df_val):,}")
df_train.head(3)
```

- Calls the function above to load training and validation sets
- Prints row counts (`:,` adds thousand-separators, e.g. `2,898,906`)
- `head(3)` shows the first 3 rows for a quick structural check
- **Output**: ~2.9M train rows, ~2.9M validation rows

---

## Cell 9 — Deterministic Sampling (10%)

```python
SAMPLE_FRAC = 0.10
SEED = 42

len_train_before, len_val_before = len(df_train), len(df_val)

if SAMPLE_FRAC < 1.0:
    df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=SEED)
    df_val   = df_val.sample(frac=SAMPLE_FRAC,   random_state=SEED)

print(f"Train size: {len_train_before:,} -> {len(df_train):,}")
print(f"Val   size: {len_val_before:,} -> {len(df_val):,}")
```

| Line | Explanation |
|---|---|
| `SAMPLE_FRAC = 0.10` | Sample 10% of the data. Set to `1.0` to disable sampling and use the full dataset |
| `SEED = 42` | Random seed for **reproducibility** — ensures the exact same subset is selected every run |
| `len_train_before, len_val_before` | Store row counts before sampling so we can print a before/after comparison |
| `df.sample(frac=..., random_state=...)` | pandas random sampling. `frac=0.10` takes 10%; `random_state=42` locks the random number generator |
| **Output** | Train: 2,898,906 → 289,891; Val: 2,938,060 → 293,806 |

**Why sample?** During development and debugging, running on the full dataset is slow. Start with 10% to validate logic and results quickly, then switch to 100% for the final run.

---

## Cell 10 — Quick Sanity Check

```python
df_train.duration.mean()
```

- Computes the mean trip duration of the sampled training set
- **Output**: ~14.43 minutes
- This is a **sanity check**: if the mean were far outside a reasonable range (e.g. 0.5 or 500), it would indicate a preprocessing bug

---

## Cell 12 — Feature Engineering: One-Hot Encoding

```python
train_dicts = df_train[CATEGORICAL].to_dict(orient='records')

dv = DictVectorizer()

X_train = dv.fit_transform(train_dicts)

y_train = df_train['duration'].values

print('Feature matrix (train):', X_train.shape)
```

| Line | Explanation |
|---|---|
| `df_train[CATEGORICAL].to_dict(orient='records')` | Convert the two DataFrame columns into a list of dictionaries, one per row. Example: `[{'PULocationID': '186', 'DOLocationID': '79'}, ...]` |
| `dv = DictVectorizer()` | Initialize a DictVectorizer. It automatically **one-hot encodes string values** and **passes numeric values through unchanged** |
| `X_train = dv.fit_transform(train_dicts)` | **fit**: scan all dictionaries to build a "category → column index" mapping. **transform**: convert each dictionary into a sparse vector. Both steps in one call |
| `y_train = df_train['duration'].values` | Extract the target variable (trip duration) as a numpy array |
| **Output** | Feature matrix shape `(289891, 491)`: 289,891 samples × 491 OHE features (i.e. 491 distinct PU/DO location values appeared in the training set) |

**Key concept — why a sparse matrix?** Out of 491 columns, each row has only **2 non-zero values** (one PU, one DO); the rest are all zeros. A sparse matrix stores only the non-zero elements, saving a huge amount of memory.

---

## Cell 14 — Inspecting OHE Results

```python
feature_names = dv.get_feature_names_out()
print('Num OHE features:', len(feature_names))

row = 0
nz_cols = X_train[row].nonzero()[1]
decoded = [(feature_names[i], X_train[row, i]) for i in nz_cols]
pd.DataFrame(decoded, columns=['feature', 'value'])
```

| Line | Explanation |
|---|---|
| `dv.get_feature_names_out()` | Retrieve all feature names, e.g. `['DOLocationID=1', 'DOLocationID=10', ..., 'PULocationID=263', ...]` |
| `X_train[row].nonzero()[1]` | Find the **column indices of non-zero elements** in row 0. `nonzero()` returns `(row_indices, col_indices)`; `[1]` gets the column indices |
| `decoded = [...]` | Translate column indices into (feature_name, value) tuples |
| **Output** | Row 0 has two non-zero features: `DOLocationID=261` (value 1.0) and `PULocationID=163` (value 1.0). This is exactly how one-hot encoding works: the matching category is 1, everything else is 0 |

---

## Cell 15 — Top-10 Pickup Locations Visualization

```python
top_pu = (df_train['PULocationID']
          .value_counts()
          .head(10)
          .sort_values(ascending=True))

plt.figure(figsize=(7,4))
top_pu.plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('PULocationID')
plt.title('Top-10 Pickup Locations (train sample)')
plt.tight_layout()
```

| Line | Explanation |
|---|---|
| `.value_counts()` | Count occurrences of each pickup location, sorted descending by default |
| `.head(10)` | Keep the 10 most frequent locations |
| `.sort_values(ascending=True)` | Reverse to ascending order so the most common location appears at the top of the horizontal bar chart (matplotlib barh draws bottom-to-top) |
| `kind='barh'` | Horizontal bar chart |
| `plt.tight_layout()` | Automatically adjust subplot layout to prevent labels from being clipped |

**Purpose**: Understand the data distribution — which locations are the busiest. In practice, models tend to perform better on high-frequency categories and may need special handling for rare ones.

---

## Cell 17 — Train the Linear Regression Model

```python
lr = LinearRegression()

start_train = time.time()
lr.fit(X_train, y_train)
end_train = time.time()
train_duration = end_train - start_train

start_pred = time.time()
y_pred_train = lr.predict(X_train)
end_pred = time.time()
pred_duration = end_pred - start_pred

rmse_train = root_mean_squared_error(y_train, y_pred_train)

print(f"Train RMSE: {rmse_train:.3f}")
print(f"Training time: {train_duration:.2f} seconds")
print(f"Prediction time: {pred_duration:.2f} seconds")
```

| Line | Explanation |
|---|---|
| `LinearRegression()` | Initialize a linear regression model (Ordinary Least Squares, no regularization) |
| `time.time()` wrapping `lr.fit(...)` | Time the training process. Internally, `fit` solves the normal equation X^T X β = X^T y |
| `lr.predict(X_train)` | Generate predictions on the training set using the fitted model |
| `root_mean_squared_error(y_train, y_pred_train)` | Compute RMSE = √(Σ(y_actual − y_pred)² / n). Measures the average magnitude of prediction error, in the same units as y (minutes) |
| **Output** | Train RMSE ≈ 7.924 min, training time 0.76s, prediction nearly instantaneous |

**Note**: This is only the **training** RMSE — it does not reflect generalization. We need the validation RMSE to assess overfitting.

---

## Cell 19 — Training Set Distribution Visualization

```python
bins = np.arange(0, 61, 1)
plt.figure(figsize=(7,5))
sns.histplot(y_train, bins=bins, alpha=0.5, label='actual')
sns.histplot(y_pred_train, bins=bins, alpha=0.5, label='prediction')
plt.xlim(0, 60)
plt.xlabel('Duration (min)')
plt.ylabel('Count')
plt.title(f'Train: RMSE = {rmse_train:.2f}')
plt.legend();
```

| Line | Explanation |
|---|---|
| `np.arange(0, 61, 1)` | Generate bin edges 0, 1, 2, ..., 60 (one bin per minute) |
| `alpha=0.5` | Set transparency to 50% so both histograms are visible when overlaid |
| `sns.histplot(y_train, ...)` | Plot the distribution of **actual** trip durations |
| `sns.histplot(y_pred_train, ...)` | Overlay the distribution of **predicted** durations |
| `plt.legend();` | The trailing semicolon suppresses matplotlib's text return-value from being printed |

**What to look for**: High overlap between the two distributions means the model captures the overall trend well. Areas where they diverge indicate where the model under/overpredicts.

---

## Cell 21 — Validation Set Evaluation

```python
val_dicts = df_val[CATEGORICAL].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = df_val['duration'].values

y_pred_val = lr.predict(X_val)
rmse_val = root_mean_squared_error(y_val, y_pred_val)
print(f"Validation RMSE: {rmse_val:.3f}")
```

| Line | Explanation |
|---|---|
| `df_val[CATEGORICAL].to_dict(orient='records')` | Same as training: convert the validation set's categorical columns into a list of dicts |
| `dv.transform(val_dicts)` | **Use `transform` only — NOT `fit_transform`!** Apply the mapping table built on the training set to the validation set. Any category in the validation set that was never seen during training is silently ignored |
| `X_val = ...` | Validation feature matrix |
| `y_val = ...` | Validation ground-truth trip durations |
| `lr.predict(X_val)` | Predict on the validation set using the model trained on the training set |
| **Output** | Validation RMSE ≈ 8.113 minutes |

**Key takeaways**:
- Train RMSE (7.924) vs Val RMSE (8.113): the gap is small (~0.19), indicating **no significant overfitting**
- However, RMSE ≈ 8 minutes on trips that average ~14 minutes means there is substantial room for improvement (e.g. adding more features, trying non-linear models)

---

## End-to-End Pipeline Summary

```
Remote Parquet Files
       │
       ▼
  read_data()
  ├─ Read parquet
  ├─ Compute duration (minutes)
  ├─ Filter outliers (1–60 min)
  └─ Cast categorical columns to string
       │
       ▼
  Deterministic 10% Sampling (seed=42)
       │
       ▼
  DictVectorizer
  ├─ fit_transform (train) → sparse matrix X_train
  └─ transform (val)       → sparse matrix X_val
       │
       ▼
  LinearRegression
  ├─ fit(X_train, y_train)
  ├─ predict(X_train) → Train RMSE ≈ 7.92
  └─ predict(X_val)   → Val RMSE   ≈ 8.11
       │
       ▼
  Visualization: actual vs predicted histogram
```

### Quick Reference — Key Concepts

| Concept | How It Appears in This Notebook |
|---|---|
| **Parquet format** | Columnar storage with compression; much faster than CSV for large-scale analytics |
| **One-Hot Encoding** | DictVectorizer converts PU/DOLocationID into a 491-dimensional sparse vector |
| **Sparse matrix** | Only 2 non-zero values per row; saves massive memory |
| **Deterministic sampling** | `random_state=42` guarantees reproducibility |
| **Train/Val split** | Temporal split (Jan vs Feb); validation set uses `transform` only — never `fit` |
| **RMSE** | Measures prediction error in the same units as the target (minutes) |
| **Overfitting detection** | Compare Train RMSE vs Validation RMSE — a small gap means no significant overfitting |
