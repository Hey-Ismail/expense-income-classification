# Expense vs Income Classification

**Course:** PROGRAMMING IN PYTHON - ML WORKFLOW

**Student/group members:**  
- 23-50009-1@student.aiub.edu - ISMAIL HOSSAIN FAHIM
- 23-50224-1@student.aiub.edu - MD. MOUDUD AHMED ALVE
- 23-50573-1@student.aiub.edu - REZVINE ENJOY NAKIB

## 1. Project Title and Group Information

**Project title:** Expense vs Income Classification

**Date:** 11 September 2026

This project builds a machine learning classifier that predicts whether a household transaction is an **Expense** or an **Income**. The idea came from a normal expense-tracker problem: users often type a note and amount quickly, but they may forget to choose the correct transaction type.

---

## 2. Problem Statement, Stakeholders, Objective, and Success Criteria

### Problem statement

In a personal finance application, every transaction needs to be labelled as either income or expense. Manually doing this is inconvenient, especially when a user is entering many transactions. This project investigates whether the transaction note, amount, payment mode, and related information can be used to predict the correct label automatically.

### Stakeholders

The main stakeholders are:

- People using personal expense-tracking applications.
- Developers who want to add automatic transaction categorisation.
- Students and researchers studying practical classification problems.
- Financial app designers who want to reduce manual data entry.

### Objective

The objective is to train and compare classification models while avoiding data leakage. The model should identify Income transactions reasonably well, because the dataset contains many more Expense records and accuracy by itself could be misleading.

### Success criteria

The project considers the following points important:

- The model should perform better than a majority-class baseline on the Income class.
- The Income-class F1 score should be used alongside accuracy, precision, and recall.
- The test set should remain untouched until final evaluation.
- The result should be reproducible using the same random seed and project files.

---

## 3. Dataset Source, License, Data Dictionary, Privacy, and Ethics

### Dataset source

The dataset is the **Daily Transactions Dataset** from Kaggle, published by Prasad Patil:

<https://www.kaggle.com/datasets/prasad22/daily-transactions-dataset>

The original file contains 2,461 household transactions recorded in India between 2015 and 2018. The dataset description says that the transactions are dummy or synthetic-style entries made by one individual. It should not be treated as a representative sample of all households.

The dataset license and permission conditions are those shown on the Kaggle dataset page. The source should be cited if the dataset is reused or shared.

### Data dictionary

| Column | Meaning | Use in this project |
|---|---|---|
| `Date` | Date and sometimes time of the transaction | Kept in the cleaned data, but not directly used as a model feature |
| `Mode` | Payment or account mode, such as Cash or Saving Bank account 1 | Used as a categorical feature |
| `Category` | Original transaction category | Removed because it can reveal the target label |
| `Subcategory` | More detailed original category | Removed for the same leakage reason |
| `Note` | User-entered description of the transaction | Converted into TF-IDF text features |
| `Amount` | Transaction amount in INR | Log-transformed and used as a numeric feature |
| `Income/Expense` | Original transaction label | Target variable |
| `Currency` | Currency of the transaction | Removed because it is constant (`INR`) |

### Privacy and ethical considerations

The data does not appear to contain names, bank account numbers, passwords, or other direct identifying information. However, transaction notes can still contain personal context. For that reason, transaction data should not be shared carelessly or used to make important financial decisions without user review.

There is also a fairness and generalisation issue. The records came from one person's household transactions, and the Income class is very small. A model trained on this data may work differently for another person's writing style, income sources, spending habits, or banking modes. The model should therefore be used as an assistant, not as an unquestioned financial decision-maker.

---

## 4. Data Audit, Exploratory Analysis, and Visualisation Findings

### Data audit

The first audit found:

- Original size: **2,461 rows and 8 columns**.
- Missing values in `Subcategory`: **635**.
- Missing values in `Note`: **521**.
- Duplicate rows: **9**.
- Target values: `Expense`, `Transfer-Out`, and `Income`.

There were 2,176 Expense records, 160 Transfer-Out records, and 125 Income records before cleaning.

### Cleaning decisions

`Transfer-Out` was removed because moving money between a person's own accounts is not naturally an income or an expense. Including it in either class would have made the target definition confusing. Duplicate rows were also removed. After these steps, the project had **2,298 rows**, including 2,173 Expense records and 125 Income records.

The original `Category` and `Subcategory` columns were not used. These columns were assigned using information closely related to the transaction label. For example, categories such as salary or investment can make an Income label obvious. Keeping them would make the score look better without proving that the model can generalise to new transactions. `Currency` was removed because every row used INR.

### Exploratory findings

The training data showed a large difference in typical transaction amounts:

- Median Expense amount: **INR 85**.
- Median Income amount: **INR 2,250**.

The median Income amount was therefore about 26 times higher than the median Expense amount, although there were exceptions in both classes. The `Mode` column also appeared useful. Cash transactions were mostly expenses, while Saving Bank account 1 contained a much larger share of the Income transactions.

The project created a bar chart called `eda_amount_by_class.png` showing the median amount for each class. The chart supports the conclusion that amount is likely to be one of the strongest predictors.

---

## 5. Split Strategy, Preprocessing Decisions, and Leakage Controls

The cleaned data was split into training and test sets using an 80/20 stratified split:

- Training set: **1,838 rows**.
- Test set: **460 rows**.
- Income proportion in training set: about **5.44%**.
- Income proportion in test set: about **5.43%**.

Stratification was important because the target was strongly imbalanced. Without it, the test set could have contained an unusually small or large number of Income transactions.

Preprocessing was done after the split and fitted only on the training data:

1. Missing notes were replaced with `unknown`.
2. Amount was transformed using `log1p` to reduce the effect of very large values.
3. `Mode` was one-hot encoded.
4. `Note` was converted into up to 50 TF-IDF features, with English stop words removed.
5. Income was encoded as 1 and Expense as 0.

The final feature matrices contained **58 features**. The test data was only transformed using encoders fitted on the training data. This is important because fitting the vocabulary or categories on the whole dataset would allow information from the test set to enter the training process.

---

## 6. Baseline Definition and Result

The baseline was a `DummyClassifier` using the `most_frequent` strategy. It always predicted the majority class, which was Expense.

The five-fold cross-validation results were:

| Metric | Baseline result |
|---|---:|
| Accuracy | 0.9456 |
| Income F1 score | 0.0000 |

The accuracy looks high, but the F1 score is zero because the baseline never predicts Income. This is a useful warning for this project: a model can have high accuracy while failing to identify the minority class at all.

---

## 7. Candidate Models, Model Choice, Hyperparameters, and Validation

Two candidate models were compared:

- **Logistic Regression:** chosen as a relatively simple and interpretable linear model.
- **Random Forest:** chosen because it can model non-linear relationships between amount, payment mode, and text features.

Both models used `class_weight='balanced'` so that the minority Income class received more attention during training. The models were evaluated using five-fold stratified cross-validation on the training set only.

The Random Forest was selected for tuning because it had the best balance of Income F1, precision, and recall in the first comparison.

The grid search tested:

- `n_estimators`: 100, 200, and 300.
- `max_depth`: None, 10, and 20.
- `min_samples_leaf`: 1, 2, and 4.

This produced 27 parameter combinations, evaluated using five-fold cross-validation and Income-class F1 as the scoring metric. The best parameters were:

```text
n_estimators = 300
max_depth = 20
min_samples_leaf = 1
```

The best cross-validation F1 score was **0.6806**.

---

## 8. Final Metrics, Diagnostic Results, Model Comparison, and Computational Cost

### Cross-validation comparison

| Model | Accuracy | Income F1 | Income precision | Income recall |
|---|---:|---:|---:|---:|
| Majority baseline | 0.9456 | 0.0000 | Not useful | 0.0000 |
| Logistic Regression | 0.8134 | 0.3260 | 0.2031 | 0.8300 |
| Random Forest | 0.9451 | 0.5878 | 0.5007 | 0.7200 |

The Logistic Regression model found more Income transactions, which is shown by its higher recall, but it also produced many false positives. The Random Forest gave a better overall balance and was selected as the final model.

### Final test-set results

The final model was evaluated once on the untouched test set:

| Model | Accuracy | Income F1 |
|---|---:|---:|
| Baseline | 0.9457 | 0.0000 |
| Logistic Regression | 0.7478 | 0.2468 |
| Tuned Random Forest | **0.9435** | **0.5357** |

For the tuned Random Forest, the class-specific results were:

| Class | Precision | Recall | F1 score | Support |
|---|---:|---:|---:|---:|
| Expense | 0.98 | 0.96 | 0.97 | 435 |
| Income | 0.48 | 0.60 | 0.54 | 25 |

The confusion matrix was:

```text
                 Predicted Expense   Predicted Income
Actual Expense          419                 16
Actual Income            10                 15
```

The model correctly found 15 of the 25 Income transactions. It missed 10 Income transactions and incorrectly labelled 16 Expense transactions as Income.

### Computational cost

The dataset is small, so the individual model fits were not expensive. The grid search was the most costly part because it evaluated 27 parameter combinations across five folds, for 135 training fits. The final Random Forest used 300 trees. This is reasonable for a small offline project, but the search could be reduced if the model had to be retrained frequently on a phone or web server.

---

## 9. Error Analysis, Imbalance, Assumptions, Bias, Uncertainty, and Limitations

### Error analysis

The model's most important feature was `log_amount`, with an importance of about **0.418**. This agrees with the EDA result that Income transactions generally had much larger amounts. Other important features included Cash mode, Saving Bank account 1, and several TF-IDF words.

Some false negatives were small Income transactions such as:

- `HDFC Stocks` - INR 88.
- `Pidilite Stocks` - INR 22.
- `interest paid` - INR 28.
- `Tata Steel Stocks` - INR 180.

These examples show a weakness of relying heavily on amount. Some genuine Income transactions are small, so they look like normal expenses to the model.

False positives included large expenses such as tours, travel, a television, and family-related expenses. The word `Family` seems difficult because phrases such as `From Family` may indicate Income, while `to Family` can describe an Expense. The simple TF-IDF representation does not fully understand this direction.

### Imbalance

After cleaning, only 125 of 2,298 records were Income, which is about 5.4%. This explains why accuracy is not enough. Class weights, stratified splitting, and Income F1 were used to reduce the effect of the imbalance.

### Assumptions and uncertainty

The project assumes that the historical label is correct and that the transaction note and amount are available when a prediction is made. It also assumes that future users will write notes that are similar enough to the training data. That assumption is uncertain because the dataset came from one person and many notes are very specific.

The test set contains only 25 Income records, so a few changed predictions could noticeably change the final Income recall or F1 score. The reported results should therefore be treated as an estimate, not a guarantee of future performance.

### Main limitations

- The data is small and highly imbalanced.
- It comes from one individual and may not represent other users.
- The data is described as dummy or synthetic-style household data.
- Some notes are missing and are replaced with `unknown`.
- The model does not fully understand phrases such as `from Family` versus `to Family`.
- The original date column was not developed into useful calendar features.
- A prediction should still be reviewed by the user before it changes a financial record.

---

## 10. Reproducibility Information

### Files

The project uses:

- `expense_income_classification.ipynb` - main notebook.
- `Daily Household Transactions.csv` - input dataset.
- `requirements.txt` - Python package requirements.
- `eda_amount_by_class.png` - generated EDA figure.

### Software and packages

The notebook uses Python with the following main libraries:

- pandas
- NumPy
- Matplotlib
- scikit-learn

The requirements file specifies compatible minimum package versions. The notebook also uses a fixed random seed of **42** for NumPy, train/test splitting, cross-validation, and model initialisation where applicable.

### Run instructions

1. Put the CSV file in the same folder as the notebook.
2. Install the packages listed in `requirements.txt`.
3. Open `expense_income_classification.ipynb` in Jupyter or VS Code.
4. Select a Python kernel with the required packages installed.
5. Run the cells from top to bottom.

The notebook includes a file check that gives a clear error if `Daily Household Transactions.csv` cannot be found. Running the cells in order is important because later cells use variables created earlier.

---

## 11. Conclusion, Practical Interpretation, Responsible Use, and Future Improvement

This project shows that transaction type can be predicted better than the majority baseline when the evaluation focuses on the minority Income class. The Random Forest was the strongest model tested. Its final test accuracy was 0.9435 and its Income F1 score was 0.5357. It correctly detected 15 of 25 Income transactions, but it also made 16 false Income predictions.

The result is useful as an automatic suggestion in an expense-tracking app. It is not accurate enough to silently change financial records without confirmation. A sensible application would show the predicted label and allow the user to correct it.

Future work could include:

- Collecting transactions from more people and more types of accounts.
- Adding date features such as month, weekday, and recurring-payment indicators.
- Using word and character n-grams to handle names and spelling variations better.
- Creating explicit text features for direction words such as `from`, `to`, `received`, and `paid`.
- Testing probability thresholds so the app can choose when to ask the user for confirmation.
- Evaluating the model on a newer dataset collected separately from the training data.
- Comparing the model with a calibrated gradient boosting model or a stronger text model.

The main lesson is that a high accuracy score alone does not mean the classifier is useful. The model must be judged by how well it identifies the less common but important Income class.

---

## 12. References and Contribution Statement

### References

1. Prasad Patil, **Daily Transactions Dataset**, Kaggle. Available at: <https://www.kaggle.com/datasets/prasad22/daily-transactions-dataset>
2. Pedregosa et al., **Scikit-learn: Machine Learning in Python**, Journal of Machine Learning Research, 2011.
3. McKinney, **Python for Data Analysis**, O'Reilly Media.

### Contribution statement

- **MD. MOUDUD AHMED ALVE (23-50224-1)** 
- **REZVINE ENJOY NAKIB (23-50573-1)** 
- **ISMAIL HOSSAIN FAHIM (23-50009-1)**