# Loan Approval Prediction — Data Analysis & Machine Learning

An end-to-end data analysis project exploring what drives loan approval
decisions, using a real-world-style dataset of 4,269 loan applicants.
The project covers data cleaning, exploratory data analysis (EDA), feature
engineering, and baseline machine learning classification.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

Financial institutions need to understand which applicant factors most
reliably predict loan approval, both to build automated scoring systems
and to ensure the process is fair and explainable. This project analyzes
a loan approval dataset to answer:

- What is the overall approval rate, and how does it vary across features?
- Which single factor most strongly predicts approval?
- Do education level, employment type, or number of dependents matter?
- Can a simple machine learning model reliably predict the outcome?

## Key Findings

| Finding | Detail |
|---|---|
| **CIBIL (credit) score is the dominant driver** | Correlation with approval ≈ **0.77** |
| **Score ≥ 650 → ~99% approval rate** | Acts almost like a hard cutoff |
| **Score < 650 → ~35% approval rate** | Sharp drop below the threshold |
| **Education / self-employment: no effect** | ~62% approval either way — no real signal |
| **Best model: Random Forest** | **97.9% accuracy, 0.983 F1-score** on held-out test data |

## Repository Structure

```
loan-approval-analysis/
├── loan_approval_analysis.ipynb   # Full analysis notebook (runs end-to-end)
├── generate_analysis.py            # Standalone EDA + chart generation script
├── model.py                        # Standalone modeling script
├── README.md
├── data/
│   ├── loan_approval_dataset_raw.csv
│   └── loan_approval_cleaned.csv   # Cleaned + feature-engineered dataset
└── images/                         # All exported chart PNGs (14 charts)
```

## Dataset

- **Rows:** 4,269 loan applicants
- **Raw features:** dependents, education, employment type, annual income,
  loan amount, loan term, CIBIL score, and four asset value categories
  (residential, commercial, luxury, bank)
- **Target:** `loan_status` (Approved / Rejected)
- No missing values or duplicate records.

## Methodology

1. **Data Cleaning** — stripped whitespace from headers/values, dropped
   the non-predictive `loan_id` column, verified no nulls or duplicates.
2. **Feature Engineering** — added `total_assets_value`,
   `loan_to_income_ratio`, `asset_to_loan_ratio`, and a categorical
   `cibil_band` (Poor / Fair / Good / Excellent).
3. **Exploratory Data Analysis** — 10 visualizations covering approval
   distribution, CIBIL score effects, education/employment, income vs.
   loan amount, assets, loan term, dependents, and a full correlation
   heatmap.
4. **Modeling** — trained and compared **Logistic Regression** (with
   standardized features) and **Random Forest**, evaluated with accuracy,
   precision, recall, F1-score, a confusion matrix, and ROC/AUC curves.
5. **Feature Importance** — extracted from the Random Forest to confirm
   which variables the model actually relies on.

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 91.3% | 92.1% | 94.2% | 93.1% |
| **Random Forest** | **97.9%** | **97.9%** | **98.7%** | **98.3%** |

## How to Run

```bash
git clone <this-repo-url>
cd loan-approval-analysis
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Option 1: open the full notebook
jupyter notebook loan_approval_analysis.ipynb

# Option 2: run the standalone scripts
python generate_analysis.py   # produces cleaned data + EDA charts
python model.py                # trains models + produces evaluation charts
```

## Tools Used

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter

## Business Recommendations

- Use CIBIL score as the primary underwriting gate — it is by far the
  most reliable single predictor in this data.
- Use income, assets, and loan-to-income ratio as secondary refinements
  for borderline applicants (Fair/Good CIBIL bands), not as standalone
  decision criteria.
- Avoid over-weighting education or self-employment in a scoring policy;
  they showed no measurable predictive value here and would only add
  potential bias.

## Author

Data Analyst — available for freelance data analysis, EDA, dashboards,
and machine learning projects.

---
*This project was built for portfolio and freelance-platform purposes,
demonstrating a full analysis workflow from raw data to a validated
predictive model.*
