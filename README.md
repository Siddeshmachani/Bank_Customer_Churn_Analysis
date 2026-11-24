# Bank Customer Churn Analysis and Prediction

An educational machine-learning project that explores bank customer churn, compares classification models, explains model behaviour, and demonstrates predictions through a Streamlit application.

**Student:** Machani Siddesh  
**Institution:** JSS Academy of Technical Education (JSSATE)  
**Programme:** B.E. – Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
**Internship provider:** Unified Mentor

## Objectives
- Inspect and prepare the supplied customer dataset.
- Explore churn patterns across customer attributes.
- Engineer additional model features.
- Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Evaluate accuracy, precision, recall, F1-score, and ROC-AUC.
- Explore model behaviour using feature importance, SHAP, and partial dependence.
- Build a Streamlit churn-risk calculator and what-if simulator.

## Dataset
The supplied dataset contains **10,000 records and 14 original columns**. The target is `Exited` (`1` = churned, `0` = retained).

| Class | Customers | Share |
|---|---:|---:|
| Retained (`Exited = 0`) | 7,963 | 79.63% |
| Churned (`Exited = 1`) | 2,037 | 20.37% |

The inspection output reported no missing values and no duplicate rows. Preprocessing removes `Year`, `CustomerId`, and `Surname`, then one-hot encodes `Geography` and `Gender`.

**Dataset notice:** Confirm the dataset licence and the provider's permission before redistributing the CSV publicly. Its real-world provenance has not been independently verified.

## Model results
A stratified 80/20 split was used (8,000 training records and 2,000 test records). The test set contains 1,593 retained and 407 churned customers.

| Model | Accuracy | Churn precision | Churn recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8065 | 0.5735 | 0.1916 | 0.2873 | 0.7734 |
| Decision Tree | 0.8600 | 0.7822 | 0.4324 | 0.5570 | 0.8392 |
| Random Forest | 0.8715 | 0.8319 | 0.4619 | 0.5940 | 0.8647 |
| Gradient Boosting | 0.8685 | 0.7857 | 0.4865 | 0.6009 | 0.8687 |

Gradient Boosting had the highest churn recall, churn F1-score, and ROC-AUC in this experiment. Random Forest had the highest churn precision and accuracy. Gradient Boosting is used as the provisional dashboard model. This choice should be revisited after cross-validation, threshold analysis, and calibration checks.

## Repository structure
```text
Bank_Customer_Churn_Analysis/
├── analysis/
│   ├── 01_data_inspection.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda_analysis.py
│   ├── 04_feature_engineering.py
│   ├── 05_model_training.py
│   ├── 06_model_comparison.py
│   └── 07_model_explainability.py
├── dashboard/
│   └── app.py
├── dataset/
├── models/
├── visualizations/
├── requirements.txt
├── .gitignore
└── README.md
```

## Tools
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SHAP, joblib, and Streamlit.

## Run locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Run analysis scripts from the repository root in order:
```bash
python analysis/01_data_inspection.py
python analysis/02_data_cleaning.py
python analysis/03_eda_analysis.py
python analysis/04_feature_engineering.py
python analysis/05_model_training.py
python analysis/06_model_comparison.py
python analysis/07_model_explainability.py
```

## Interpretation and limitations
- Churn is the minority class (20.37%); accuracy alone is insufficient.
- Gradient Boosting churn recall is 0.4865, so it identifies fewer than half of actual churners at the default threshold.
- Probability calibration has not been formally assessed.
- Feature importance and SHAP explain model behaviour; they do not establish causality.
- The what-if simulator is illustrative, not a guarantee of real customer behaviour.
- Evaluation uses one stratified hold-out split. Repeated cross-validation, threshold tuning, calibration, fairness checks, and external validation are recommended before operational use.
- This is an educational project, not a production banking decision system.

## References
1. Tékouabou, S. C. K., Gherghina, Ș. C., Toulni, H., Mata, P. N., & Martins, J. M. (2022). *Towards Explainable Machine Learning for Bank Churn Prediction Using Data Balancing and Ensemble-Based Methods*. Mathematics, 10(14), 2379. https://doi.org/10.3390/math10142379
2. Lemos, R. A. de L., Silva, T. C., & Tabak, B. M. (2022). *Propension to customer churn in a financial institution: a machine learning approach*. Neural Computing and Applications, 34, 11751–11768. https://doi.org/10.1007/s00521-022-07067-x
3. Brito, J. B. G., et al. (2024). *A framework to improve churn prediction performance in retail banking*. Financial Innovation, 10, 17. https://doi.org/10.1186/s40854-023-00558-3
4. Gkonis, V., & Tsakalos, I. (2024). *Deep Dive Into Churn Prediction in the Banking Sector: The Challenge of Hyperparameter Selection and Imbalanced Learning*. Journal of Forecasting. https://doi.org/10.1002/for.3194
5. Bhuria, R., et al. (2025). *Ensemble-based customer churn prediction in banking: a voting classifier approach for improved client retention using demographic and behavioral data*. Discover Sustainability, 6, 28. https://doi.org/10.1007/s43621-025-00807-8

## Author
**Machani Siddesh** — JSS Academy of Technical Education (JSSATE), B.E. CSE (AI & ML). Internship project with Unified Mentor.
