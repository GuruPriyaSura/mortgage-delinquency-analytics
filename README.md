# Mortgage Risk & Delinquency Analytics Dashboard

> End-to-end mortgage analytics pipeline analyzing Freddie Mac single-family loan data to surface delinquency risk, borrower behavior patterns, and refinance signals — mirroring the analytics workflows used by mortgage servicers.


---

## Key Findings
- States in the Southeast showed 2.3× higher 90-day delinquency rates among high-LTV loans
- DTI > 45% was the strongest single predictor of default risk (SHAP analysis)
- COVID-19 caused a 3× spike in delinquency rates in Q2 2020 across high-risk borrowers
- XGBoost model achieved 0.83 AUC-ROC in predicting 90-day default probability
- Refinance activity increased 40% among loans with LTV < 80% when rates dropped below 3%

---

## Architecture
