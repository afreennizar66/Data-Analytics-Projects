# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Reading the data

df = pd.read_csv('C://Users//Afreen N//Desktop//ML project//Diabetes.csv')

# EDA

print(df.head(), "\n")
print(df.describe(), "\n")
print(df.info(), "\n")
print(df.isnull().sum(), "\n")

# Histogram for feature distribution

columns = ['HighBP', 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
fig, axes = plt.subplots(4,2, figsize=(10, 10))
axes = axes.flatten()
for i, col in enumerate(columns):
  sns.histplot(data=df, x=col, ax=axes[i], hue='Diabetes_binary', palette='dark')
  axes[i].set_title(f"Distribution of {col}", fontweight='bold')
  axes[i].set_xlabel(col)
  axes[i].set_ylabel("Frequency")
plt.tight_layout()
plt.show()
print("\n")

# Plotting boxplot of features for outlier detection

df.boxplot(figsize=(20,8))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Data preparation

x = df.drop(columns=['Diabetes_binary'])
y = df['Diabetes_binary']
print(x.head(), "\n")
print(y.head(), "\n")
print(x.isnull().sum(), "\n")

# Correlation matrix grouped by target

print("Correlation matrix: \n")
data_corr = df.corr()
target_corr = data_corr['Diabetes_binary'].sort_values(ascending=False)
target_corr = target_corr.to_frame()
sns.heatmap(target_corr, annot=True)
plt.show()
print("\n")

# Correlation matrix of features

plt.figure(figsize=(15,8))
sns.heatmap(data_corr, annot=True)
plt.show()

# Feature engineering using RandomForestClassifier

print("Importance of features: \n")
features_rf = RandomForestClassifier()
features_rf.fit(x, y)
importance = (features_rf.feature_importances_)
plt.bar(x=x.columns, height=importance)
plt.xticks(rotation=90)
plt.show()

# Dropping less significant columns

cols_drop = ['Stroke', 'HeartDiseaseorAttack', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost']
x = x.drop(columns=cols_drop)
print(x.head(), "\n")

# Splitting data into train and test

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)
print(x_train.isnull().sum(), y_train.isnull().sum())

# Feature scaling

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Modelling

# Logistic Regression

p_lr = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}  # Parameters
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross validation
gs_lr = GridSearchCV(estimator=LogisticRegression(), param_grid=p_lr, cv=cv_lr)    # Hyperparameter tuning
gs_lr.fit(x_train, y_train)   # Model fitting
m_lr = gs_lr.best_estimator_   # Finding best estimator with best parameters
y_pred_lr = m_lr.predict(x_test)  # Prediction

# Logistic Regression Evaluation

print("Best parameters: ", gs_lr.best_params_)
print("Best score: ", gs_lr.best_score_)
print("Accuracy: ", accuracy_score(y_test, y_pred_lr))
print("Classification Report: \n", classification_report(y_test, y_pred_lr))
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion matrix: \n", cm_lr, "\n")
for i in range(len(cm_lr)):     # Calculating specificity and sensitivity
  tp = cm_lr[i][i]
  fp = np.sum(cm_lr[i, :]) - tp
  fn = np.sum(cm_lr[:, i]) - tp
  tn = np.sum(cm_lr) - tp - fp - fn
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Class: ", i, "\n")
  print("Specificity: ", specificity, "\n")
kapp_lr = cohen_kappa_score(y_test, y_pred_lr)
print("Kappa score: ", kapp_lr, "\n")

# Random Forest Classifier

p_rfc = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [2, 4]}  # Parameters
cv_rfc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross validation
gs_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=p_rfc, cv=cv_rfc)    # Hyperparameter tuning
gs_rfc.fit(x_train, y_train)   # Model fitting
m_rfc = gs_rfc.best_estimator_   # Finding best estimator with best parameters
y_pred_rfc = m_rfc.predict(x_test)  # Prediction

# Random Forest Classifier Evaluation

print("Best parameters: ", gs_rfc.best_params_)
print("Best score: ", gs_rfc.best_score_)
print("Accuracy: ", accuracy_score(y_test, y_pred_rfc))
print("Classification Report: \n", classification_report(y_test, y_pred_rfc))
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print("Confusion matrix: \n", cm_rfc, "\n")
for i in range(len(cm_rfc)):     # Calculating specificity and sensitivity
  tp = cm_rfc[i][i]
  fp = np.sum(cm_rfc[i, :]) - tp
  fn = np.sum(cm_rfc[:, i]) - tp
  tn = np.sum(cm_rfc) - tp - fp - fn
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Class: ", i, "\n")
  print("Specificity: ", specificity, "\n")
kapp_rfc = cohen_kappa_score(y_test, y_pred_rfc)
print("Kappa score: ", kapp_rfc, "\n")

# Decision Tree Classifier

p_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [2, 4]}  # Parameters
cv_dt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross validation
gs_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=p_dt, cv=cv_dt)    # Hyperparameter tuning
gs_dt.fit(x_train, y_train)   # Model fitting
m_dt = gs_dt.best_estimator_   # Finding best estimator with best parameters
y_pred_dt = m_dt.predict(x_test)

# Decision Tree Classifier Evaluation

print("Best parameters: ", gs_dt.best_params_)
print("Best score: ", gs_dt.best_score_)
print("Accuracy: ", accuracy_score(y_test, y_pred_dt))
print("Classification Report: \n", classification_report(y_test, y_pred_dt))
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion matrix: \n", cm_dt, "\n")
for i in range(len(cm_dt)):  # Calculating specificity and sensitivity
  tp = cm_dt[i][i]
  fp = np.sum(cm_dt[i, :]) - tp
  fn = np.sum(cm_dt[:, i]) - tp
  tn = np.sum(cm_dt) - tp - fp - fn
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Class: ", i, "\n")
  print("Specificity: ", specificity, "\n")
kapp_dt = cohen_kappa_score(y_test, y_pred_dt)
print("Kappa score: ", kapp_dt, "\n")

# ROC curve

fpr_model1, tpr_model1, _ = roc_curve(y_test, y_pred_lr)
fpr_model2, tpr_model2, _ = roc_curve(y_test, y_pred_rfc)
fpr_model3, tpr_model3, _ = roc_curve(y_test, y_pred_dt)
auc_model1 = auc(fpr_model1, tpr_model1)   # Calculating auc score
auc_model2 = auc(fpr_model2, tpr_model2)
auc_model3 = auc(fpr_model3, tpr_model3)
plt.plot(fpr_model1, tpr_model1, label='Logistic Regression (AUC = %0.2f)' % auc_model1)
plt.plot(fpr_model2, tpr_model2, label='Random Forest Classifier (AUC = %0.2f)' % auc_model2)
plt.plot(fpr_model3, tpr_model3, label='Decision Tree Classifier (AUC = %0.2f)' % auc_model3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
