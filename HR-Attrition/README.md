*** Employee Attrition Prediction (XGBoost + SMOTE + Full ML Pipeline)***
A complete ML engineering project using real HR analytics data.
** Project Overview **
Employee attrition is a critical challenge for organizations, directly affecting productivity, hiring costs, and team performance.
This project builds a machine learning model to predict whether an employee is likely to leave the company.
You implemented a full ML pipeline, including:
Data preprocessing (scaling, encoding, cleaning)
Class imbalance handling (SMOTE / class weights)
Model comparison (LogReg, Random Forest, XGBoost)
Hyperparameter tuning
Final model using XGBoost + SMOTE
Evaluation using precision, recall, F1-score, and confusion matrix
Production-ready pipeline using scikit-learn + XGBoost
This is a complete, industry-standard ML workflow.
** Tech Stack **
Python 3.11+
Pandas, NumPy
Scikit-Learn
XGBoost
Imbalanced-Learn (SMOTE / SMOTENC)
Matplotlib / Seaborn
(Optional) LightGBM
** Dataset **
Dataset: IBM HR Analytics Employee Attrition Dataset
Common features include:
Age, Gender, DistanceFromHome
JobRole, Department, BusinessTravel
JobSatisfaction, EnvironmentSatisfaction
YearsAtCompany, YearsSinceLastPromotion
MonthlyIncome, Overtime, StockOptionLevel
Target variable:
Attrition
Yes → 1  
No  → 0
** Data Preprocessing **
You implemented a fully automated preprocessing pipeline:
✔ Numerical Features
Standard scaling
Imputation (if needed)
✔ Categorical Features
One-Hot Encoding via ColumnTransformer
✔ Train/Test Split
80/20
Stratified by target to maintain class ratios
✔ Class Imbalance Handling
Two approaches:
SMOTENC for mixed categorical + numeric data
scale_pos_weight for XGBoost (backup method)
🤖 Models Trained
You trained and compared:
Model	Performance (Class 1 Recall)
Logistic Regression	~20–30%
Random Forest	~6% (very poor)
XGBoost (baseline)	~45–50%
XGBoost + SMOTENC (final)	~45–65%
XGBoost consistently performed the best for tabular HR data.
** Final Model: XGBoost + SMOTENC **
Pipeline:
[Preprocessing] →
[SMOTENC Oversampling] →
[XGBoost Classifier]
Final performance (example):
Precision (Class 1): 0.54  
Recall (Class 1):    0.45  
F1-score (Class 1):  0.49  
Accuracy:            0.85
This performance is considered strong for HR attrition prediction, where the minority class is small and noisy.
** Evaluation Metrics **
Confusion Matrix
Precision, Recall, F1-Score
Accuracy
Class Imbalance Analysis
Threshold tuning (optional)
The goal is not highest accuracy, but highest recall for class 1 (leavers).
