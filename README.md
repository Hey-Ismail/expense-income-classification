# Expense vs Income Classification — Final Term Project

## Overview
Predicts whether a personal-finance transaction is an **Income** or an **Expense** from its
note/description, amount, payment mode, and date — motivated by automating the manual
Income/Expense tagging step in an expense-tracker app (this team's mid-term project).

## Project Type
Classification (binary): `Income` (1) vs `Expense` (0).

## Data
- **Source:** [Daily Transactions Dataset](https://www.kaggle.com/datasets/prasad22/daily-transactions-dataset)
  by Prasad Patil, Kaggle (v4).
- **File used:** `Daily Household Transactions.csv` (included in this submission).
- **Size:** 2,461 rows, 8 columns (before cleaning).
- **Note on provenance:** the dataset description states these are *dummy transactions made by
  an individual* — i.e. synthetic-style data, not raw anonymized bank records. This is an
  acknowledged limitation (see notebook Section 10).

## Repository Structure
```
.
├── Daily Household Transactions.csv   # raw dataset
├── expense_income_classification.ipynb # full pipeline (run top-to-bottom)
├── requirements.txt
└── README.md
```

## How to Run
1. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter and run `expense_income_classification.ipynb` top-to-bottom:
   ```bash
   jupyter notebook expense_income_classification.ipynb
   ```
   All cells execute in order with no manual steps in between; the CSV file must be in the same
   directory as the notebook.

## Pipeline Summary
1. **Data audit** — missing values, duplicates, class balance check.
2. **Scope decision** — `Transfer-Out` records dropped (conceptually neither income nor expense).
3. **Leakage check** — `Category`/`Subcategory` dropped because they were assigned using the same
   information as the target; `Currency` dropped (constant column).
4. **Stratified train/test split (80/20)** — done before any preprocessing.
5. **EDA** — on training data only.
6. **Preprocessing** — missing-note imputation, log-transform on `Amount`, one-hot encoding of
   `Mode`, TF-IDF on `Note` — all fit on training data only, then applied to the test set.
7. **Baseline** — majority-class `DummyClassifier`.
8. **Models compared** — Logistic Regression vs Random Forest (both `class_weight='balanced'`),
   5-fold stratified cross-validation, then `GridSearchCV` tuning of Random Forest on CV only.
9. **Final evaluation** — tuned Random Forest evaluated once on the untouched test set.
10. **Error analysis** — false positive/negative inspection, feature importances, discussion of
    limitations.

## Key Results (Test Set)
| Model | Accuracy | F1 (Income class) |
|---|---|---|
| Baseline (majority class) | 94.6% | 0.00 |
| Logistic Regression | 74.8% | 0.25 |
| **Random Forest (tuned, final)** | **95.4%** | **0.59** |

See the notebook for the full confusion matrix, classification report, and error analysis.

## Team
_(add team member names and IDs here)_
