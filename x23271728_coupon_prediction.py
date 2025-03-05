# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Reading the data

df = pd.read_csv('C://Users//Afreen N//Desktop//ML project//in-vehicle-coupon-recommendation.csv')

# EDA

print(df.head(), "\n")
print(df.describe(), "\n")
print(df.info(), "\n")
print(df.isnull().sum(), "\n")
null_values = df.isnull().sum()

# Plotting distribution of missing values

sns.barplot(x=null_values.index, y=null_values.values, color='#90EE90')
plt.title('Distribution of missing values')
plt.xlabel('Missing values')
plt.ylabel('Count of missing values')
plt.xticks(rotation=90)
plt.show()
print("\n")

# Plotting histogram of features against target

flg, ax = plt.subplots(2,5, figsize=(28,20))
sns.histplot(x = df['destination'], hue=df['Y'], ax=ax[0,0])
ax[0,0].set_title('Destination by coupon usage')
ax[0,0].tick_params(axis='x', rotation=45)
sns.histplot(x = df['passanger'], hue=df['Y'], ax=ax[0,1])
ax[0,1].set_title('Passenger type by coupon usage')
ax[0,1].tick_params(axis='x', rotation=45)
sns.histplot(x = df['weather'], hue=df['Y'], ax=ax[0,2])
ax[0,2].set_title('Weather by coupon usage')
ax[0,2].tick_params(axis='x', rotation=45)
sns.histplot(x = df['income'], hue=df['Y'], ax=ax[0,3])
ax[0,3].set_title('Income by coupon usage')
ax[0,3].tick_params(axis='x', rotation=45)
sns.histplot(x = df['time'], hue=df['Y'], ax=ax[0,4])
ax[0,4].set_title('Time by coupon usage')
ax[0,4].tick_params(axis='x', rotation=45)
sns.histplot(x = df['coupon'], hue=df['Y'], ax=ax[1,0])
ax[1,0].set_title('Coupon type by coupon usage')
ax[1,0].tick_params(axis='x', rotation=45)
sns.histplot(x = df['gender'], hue=df['Y'], ax=ax[1,1])
ax[1,1].set_title('Gender by coupon usage')
ax[1,1].tick_params(axis='x', rotation=45)
sns.histplot(x = df['age'], hue=df['Y'], ax=ax[1,2])
ax[1,2].set_title('Age by coupon usage')
ax[1,2].tick_params(axis='x', rotation=45)
sns.histplot(x = df['maritalStatus'], hue=df['Y'], ax=ax[1,3])
ax[1,3].set_title('Martial Status by coupon usage')
ax[1,3].tick_params(axis='x', rotation=45)
sns.histplot(x = df['car'], hue=df['Y'], ax=ax[1,4])
ax[1,4].set_title('Car by fraudulent')
ax[1,4].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
print("\n")

# Data preparation

df = df.drop(columns=['car', 'toCoupon_GEQ5min'])   # Dropping insignificant columns
print(df.columns, "\n")

# Replacing missing values with mode imputation

missing_cols = ['Bar','CoffeeHouse','CarryAway','RestaurantLessThan20', 'Restaurant20To50']
for i in missing_cols:
  df[i] = df[i].fillna(df[i].mode()[0])
print(df.isnull().sum(), "\n")

# Assigning independent and dependent variables

x = df.drop(columns=['Y'])
y = df['Y']
print(x.shape, "\n")
print(y.shape, "\n")

# Splitting data into train and test

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print(x_train.isnull().sum())

# Categorical features encoding

one_hot_cols = ['destination', 'passanger', 'weather', 'time', 'coupon', 'gender', 'maritalStatus', 'occupation', 'age']
x_train = pd.get_dummies(x_train, columns=one_hot_cols, drop_first=True)
x_test = pd.get_dummies(x_test, columns=one_hot_cols, drop_first=True)

expiration_map = {'1d': 1, '2h': 2}
x_train['expiration'] = x_train['expiration'].map(expiration_map)
x_test['expiration'] = x_test['expiration'].map(expiration_map)

education_map = {'Some High School': 0, 'High School Graduate': 1, 'Bachelors degree': 2, 'Some college - no degree': 3, 'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)':5, 'Associates degree':6}
x_train['education'] = x_train['education'].map(education_map)
x_test['education'] = x_test['education'].map(education_map)

income_map = {'$100000 or More': 0, '$12500 - $24999': 1, '$25000 - $37499': 2, '$37500 - $49999': 3, '$50000 - $62499': 4, '$62500 - $74999': 5, '$75000 - $87499': 6, '$87500 - $99999': 7, 'Less than $12500': 8}
x_train['income'] = x_train['income'].map(income_map)
x_test['income'] = x_test['income'].map(income_map)

time_frame_map = {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4}
x_train['Bar'] = x_train['Bar'].map(time_frame_map)
x_train['CoffeeHouse'] = x_train['CoffeeHouse'].map(time_frame_map)
x_train['CarryAway'] = x_train['CarryAway'].map(time_frame_map)
x_train['RestaurantLessThan20'] = x_train['RestaurantLessThan20'].map(time_frame_map)
x_train['Restaurant20To50'] = x_train['Restaurant20To50'].map(time_frame_map)
x_test['Bar'] = x_test['Bar'].map(time_frame_map)
x_test['CoffeeHouse'] = x_test['CoffeeHouse'].map(time_frame_map)
x_test['CarryAway'] = x_test['CarryAway'].map(time_frame_map)
x_test['RestaurantLessThan20'] = x_test['RestaurantLessThan20'].map(time_frame_map)
x_test['Restaurant20To50'] = x_test['Restaurant20To50'].map(time_frame_map)

print(x_train.head(), "\n")
print(x_test.head())

print(x_train.isnull().sum())
print("\n")
x_test.isnull().sum()

# Finding significant features using RandomForestClassifier()

print("Importance of features: \n")
features_rf = RandomForestClassifier()
features_rf.fit(x_train, y_train)
importance = (features_rf.feature_importances_)
plt.figure(figsize=(15,8))
plt.bar(x=x_train.columns, height=importance)
plt.xticks(rotation=90)
plt.show()

# Dropping insignificant columns

cols_drop = ['time_7AM','coupon_Restaurant(20-50)','time_6PM','time_2PM','maritalStatus_Unmarried partner','weather_Sunny', 'occupation_Arts Design Entertainment Sports & Media',
 'time_10PM','passanger_Partner','destination_Work','passanger_Kid(s)','weather_Snowy','passanger_Friend(s)','destination_No Urgent Place','maritalStatus_Widowed','gender_Male',
  'maritalStatus_Single','maritalStatus_Married partner',
  'direction_opp','direction_same','toCoupon_GEQ25min','toCoupon_GEQ15min','has_children','occupation_Building & Grounds Cleaning & Maintenance',
 'occupation_Business & Financial', 'occupation_Community & Social Services', 'occupation_Computer & Mathematical', 'occupation_Construction & Extraction',
 'occupation_Education&Training&Library','occupation_Farming Fishing & Forestry', 'occupation_Food Preparation & Serving Related', 'occupation_Healthcare Support', 'occupation_Healthcare Practitioners & Technical',
 'occupation_Installation Maintenance & Repair', 'occupation_Legal', 'occupation_Life Physical Social Science', 'occupation_Management', 'occupation_Office & Administrative Support',
 'occupation_Personal Care & Service', 'occupation_Production Occupations', 'occupation_Protective Service', 'occupation_Retired','occupation_Sales & Related', 'occupation_Student','occupation_Transportation & Material Moving',
 'occupation_Unemployed', 'age_26', 'age_31', 'age_36', 'age_41', 'age_46', 'age_50plus', 'age_below21','temperature']

x_train = x_train.drop(columns=cols_drop)
x_test = x_test.drop(columns=cols_drop)
print(x_train.shape, "\n")
print(x_test.shape)

# Feature Scaling

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Modelling

# kNN

p_kn = {'n_neighbors': [3,5,7,9]}  # Parameters
cv_kn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross validation
gs_kn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=p_kn, cv=cv_kn)    # Hyperparameter tuning
gs_kn.fit(x_train_scaled, y_train)   # Model fitting
m_kn = gs_kn.best_estimator_   # Finding best estimator with best parameters
y_pred_kn = m_kn.predict(x_test_scaled)

# kNN Evaluation

print("Best parameters: ", gs_kn.best_params_)
print("Best score: ", gs_kn.best_score_)
print("Accuracy: ", accuracy_score(y_test, y_pred_kn))
print("Classification Report: \n", classification_report(y_test, y_pred_kn))
cm_kn = confusion_matrix(y_test, y_pred_kn)
print("Confusion matrix: \n", cm_kn, "\n")
for i in range(len(cm_kn)):
  tp = cm_kn[i][i]
  fp = np.sum(cm_kn[i, :]) - tp
  fn = np.sum(cm_kn[:, i]) - tp
  tn = np.sum(cm_kn) - tp - fp - fn
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Class: ", i, "\n")
  print("Specificity: ", specificity, "\n")
kapp_kn = cohen_kappa_score(y_test, y_pred_kn)
print("Kappa score: ", kapp_kn)

# Decision Tree Classifier

p_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [2, 4]}  # Parameters
cv_dt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross validation
gs_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=p_dt, cv=cv_dt)    # Hyperparameter tuning
gs_dt.fit(x_train, y_train)   # Model fitting
m_dt = gs_dt.best_estimator_   # Finding best estimator with best parameters
y_pred_dt = m_dt.predict(x_test)  # Prediction

# Decision Tree Classifier Evaluation

print("Best parameters: ", gs_dt.best_params_)
print("Best score: ", gs_dt.best_score_)
print("Accuracy: ", accuracy_score(y_test, y_pred_dt))
print("Classification Report: \n", classification_report(y_test, y_pred_dt))
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion matrix: \n", cm_dt, "\n")
for i in range(len(cm_dt)):
  tp = cm_dt[i][i]
  fp = np.sum(cm_dt[i, :]) - tp
  fn = np.sum(cm_dt[:, i]) - tp
  tn = np.sum(cm_dt) - tp - fp - fn
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Class: ", i, "\n")
  print("Specificity: ", specificity, "\n")
kapp_dt = cohen_kappa_score(y_test, y_pred_dt)
print("Kappa score: ", kapp_dt)

# ROC curve and AUC score

fpr_model1, tpr_model1, _ = roc_curve(y_test, y_pred_kn)
fpr_model2, tpr_model2, _ = roc_curve(y_test, y_pred_dt)
auc_model1 = auc(fpr_model1, tpr_model1)   # Calculating auc score
auc_model2 = auc(fpr_model2, tpr_model2)
plt.plot(fpr_model1, tpr_model1, label='kNN Classifier (AUC = %0.2f)' % auc_model1)
plt.plot(fpr_model2, tpr_model2, label='Decision Tree Classifier (AUC = %0.2f)' % auc_model2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
