# Employee Attrition Prediction (XGBoost + SMOTE) #
A complete machine learning pipeline to predict employee attrition using the IBM HR Analytics dataset.
This project includes data preprocessing, class imbalance handling, model training, evaluation, and a production-ready XGBoost pipeline.
## Project Overview ##
Employee attrition is costly for organizations. This project builds a classification model that predicts whether an employee is likely to leave the company.
The project includes:
Exploratory data analysis
Data preprocessing (scaling, encoding, cleanup)
Handling severe class imbalance
Training multiple ML models
Comparing performance
Final model using XGBoost + SMOTENC
Evaluation using recall, precision, F1-score, confusion matrix
This is a full ML engineering workflow.
## Tech stack ##
Python 3.x
Pandas
NumPy
Scikit-learn
Imbalanced-learn (SMOTENC)
XGBoost
Matplotlib / Seaborn
Dataset
Dataset: IBM HR Analytics Employee Attrition Dataset.
Contains employee demographic, job-related, and satisfaction metrics.
Target variable:
Attrition:
    Yes → 1
    No  → 0
Features include:
Age, Gender, DistanceFromHome
BusinessTravel, Department, JobRole
EnvironmentSatisfaction, JobSatisfaction
Overtime, MonthlyIncome, PercentSalaryHike
YearsAtCompany, YearsInCurrentRole
Education, JobLevel, StockOptionLevel
Data Preprocessing
The project uses a modular preprocessing pipeline:
Numerical Features
Standard scaling
Optional imputation
Categorical Features
One-Hot Encoding
Handled through ColumnTransformer
Train/Test Split
80/20 split
Stratified sampling
Class Imbalance
Two approaches explored:
SMOTENC (synthetic oversampling for mixed categorical + numeric data)
XGBoost class weighting (scale_pos_weight)
Models Trained
The following models were trained and compared:
Logistic Regression
Random Forest
XGBoost (baseline)
XGBoost + SMOTENC (final model)
Random Forest performed poorly due to imbalance.
XGBoost consistently performed best.
Final Model: XGBoost + SMOTENC
### The final pipeline: ###
[SMOTENC Oversampling] → [Preprocessing] → [XGBoost Classifier]
Example performance on holdout test set:
Precision (Class 1): 0.54
Recall (Class 1):    0.45
F1-score (Class 1):  0.49
Accuracy:            0.85
Because attrition is heavily imbalanced and noisy, recall for Class 1 (leavers) is the most important metric.
This model achieves competitive results and performs significantly better than classical baselines.
Evaluation Metrics
Confusion Matrix
Precision
Recall
F1-score
Overall Accuracy
Class balance before/after SMOTE
Threshold adjustments (optional)
