# Expense Income Classification

## Overview

This project uses machine learning to classify household transactions as either **Income** or
**Expense**. The model uses transaction notes, amounts, and payment modes to make predictions.

## File Structure

```text
.
├── expense_income_classification.ipynb
├── Daily Household Transactions.csv
├── requirements.txt
├── report.md
└── README.md
```

- `expense_income_classification.ipynb` - analysis and machine learning notebook.
- `Daily Household Transactions.csv` - dataset used by the notebook.
- `requirements.txt` - required Python packages.
- `expense_income_classification_report.md` - project report.

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or VS Code with the Jupyter extension
- pip

## Environment Setup

Open a terminal in the project folder and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, install the packages using:

```powershell
python -m pip install -r requirements.txt
```

## Data Placement

Keep `Daily Household Transactions.csv` in the project root, in the same folder as
`expense_income_classification.ipynb`. The notebook loads the dataset from this location.

The dataset is the [Daily Transactions Dataset](https://www.kaggle.com/datasets/prasad22/daily-transactions-dataset)
from Kaggle. If the CSV is missing or stored in another folder, the notebook cannot load the data.

## Run the Project

With the virtual environment activated, start Jupyter Notebook:

```bash
jupyter notebook
```

Then open `expense_income_classification.ipynb` and run the cells from top to bottom.

You can also open the project folder in VS Code, select the `.venv` Python interpreter, open the
notebook, and run all cells. Make sure the notebook kernel uses the environment where the
requirements were installed.

To stop the virtual environment after finishing, run:

```bash
deactivate
```

## Expected Output

After all cells run successfully, the notebook performs data cleaning, exploratory analysis,
model training, and evaluation. It should produce charts, classification metrics, a confusion
matrix, and model comparison results.

The expected final test-set result is approximately:

| Model | Accuracy | Income F1 score |
|---|---:|---:|
| Majority baseline | 94.6% | 0.00 |
| Logistic Regression | 74.8% | 0.25 |
| Tuned Random Forest | 95.4% | 0.59 |

Small differences may occur because of package versions or changes to the dataset. The tuned
Random Forest is the final selected model.

## Team Members

- MD. MOUDUD AHMED ALVE - 23-50224-1@student.aiub.edu
- REZVINE ENJOY NAKIB - 23-50573-1@student.aiub.edu
- ISMAIL HOSSAIN FAHIM - 23-50009-1@student.aiub.edu
 