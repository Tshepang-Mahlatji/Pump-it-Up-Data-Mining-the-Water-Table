# 💧 Data Mining the Water Table — Pump Status Prediction

Predicting the operational status of Tanzanian water pumps — **functional**, **functional needs repair**, or **non functional** — from 40 descriptive features. Based on the [DrivenData "Pump it Up"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition. A **3-class classification** task scored on **accuracy**.

## 📊 Results

| Metric | Value |
|---|---|
| Best model | Tuned Random Forest |
| Hold-out accuracy | **0.815** |
| Hold-out macro-F1 | **0.690** |
| Training rows | 59,400 |
| Test rows | 14,850 |
| Engineered features | 26 |

Per-class recall (hold-out): functional **0.91**, non-functional **0.78**, needs-repair **0.30** — the rare middle class (7% of data) is the model's main weakness.

## 📁 Contents

```
water_table_analysis/
├── Water_Table_Analysis.ipynb   # Full runnable notebook: EDA → FE → FS → modeling → tuning
├── Water_Table_Report.pdf       # Polished 11-page report with figures + interpretation
├── Water_Table_Dashboard.html   # Interactive dashboard (open in a browser)
├── submission.csv               # Test-set predictions (competition format)
├── results.json                 # All computed metrics and statistics
├── figures/                     # 14 analysis figures (F01–F14)
└── README.md
```

Input data (in the parent folder): `Training set values.csv`, `Training set labels.csv`, `Test set values.csv`, `SubmissionFormat.csv`.

## 🔬 Method

1. **EDA** — target imbalance (54% functional / 38% non-functional / 7% needs-repair), data-quality audit (placeholder zeros in `amount_tsh`, `construction_year`, `gps_height`, `population`), geospatial clustering, and status rates by category.
2. **Statistical analysis** — χ² tests of independence and bias-corrected **Cramér's V** for categorical association; Pearson correlation for numerics; **mutual information** for feature selection. Strongest drivers: `quantity`, `waterpoint_type`, `extraction_type_class`, `region`, and pump `age`.
3. **Feature engineering** — placeholder zeros → `NaN` with missingness flags; `age` from construction year; log transforms; top-K grouping of high-cardinality columns (`funder`, `installer`, `ward`, `lga`); dropping constant/redundant columns. 40 → 26 features.
4. **Model selection** — five estimators compared under 3-fold cross-validation in leakage-safe scikit-learn Pipelines: stratified baseline, Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting.
5. **Tuning** — randomized search over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` for the winning Random Forest.

### Cross-validated model comparison

| Model | CV accuracy | CV macro-F1 |
|---|---|---|
| Baseline (stratified) | 0.449 | 0.335 |
| Logistic Regression | 0.659 | 0.445 |
| Decision Tree | 0.764 | 0.624 |
| Hist Gradient Boosting | 0.798 | 0.670 |
| **Random Forest** (selected) | **0.803** | **0.687** |

## 🚀 How to run

Requirements: Python 3.9+, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy jupyter
```

Open the notebook and run **Kernel → Restart & Run All** (all NaN handling lives inside the Pipelines, so cells must run in order). To view the report or dashboard, open the `.pdf` or `.html` file directly.

## 📝 Notes & next steps

- Encoding and imputation are fit **inside** each CV fold to avoid leakage.
- The needs-repair class is the main limitation; improvements to try: class weighting / SMOTE, target (or CatBoost) encoding of high-cardinality geography, and stacking Random Forest with gradient boosting.

## 🛠️ Tech stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `SciPy` · `matplotlib` · `seaborn` · `Jupyter`
