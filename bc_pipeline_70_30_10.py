#Importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA

bc_data = pd.read_csv("bc_data.csv")
Y = bc_data["target"]
print(Y.shape,"\n")
X = bc_data.drop("target", axis=1)
print(X.shape,"\n")

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    stratify=Y,
                                                    random_state=42)
print(X_train.shape,"\n")
print(X_test.shape,"\n")
print(y_train.shape,"\n")
print(y_test.shape,"\n")

pipeline1 = imbpipeline(steps = [['ROS', RandomOverSampler(random_state=42)],
                                ['scaler', StandardScaler()],
                                ['model_1', LogisticRegression(random_state=42, max_iter=10000)]])
pipeline2 = imbpipeline(steps = [['ROS', RandomOverSampler(random_state=42)],
                                ['scaler', StandardScaler()],
                                ['model_2', RandomForestClassifier(random_state=42)]])
pipeline3 = imbpipeline(steps = [['ROS', RandomOverSampler(random_state=42)],
                                ['scaler', StandardScaler()],
                                ['model_3', KNeighborsClassifier(p = 1)]])
pipeline4 = imbpipeline(steps = [['ROS', RandomOverSampler(random_state=42)],
                                ['scaler', StandardScaler()],
                                ['model_4', SVC(gamma='auto',random_state=42)]])
pipeline5 = imbpipeline(steps = [['ROS', RandomOverSampler(random_state=42)],
                                ['scaler', StandardScaler()],
                                ['model_5',DecisionTreeClassifier(random_state=42)]])

p1 = {"model_1__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],"model_1__penalty": ['l2']}
p2 = [{"model_2__n_estimators": [500,1000]}]
p3 = [{"model_3__n_neighbors": list(range(1, 10))}]
p4 = [{'model_4__kernel': ['rbf'],
                'model_4__C': [0.01, 0.1, 1, 10, 100],
                'model_4__gamma': [0.001, 0.01, 0.1, 1, 10]}]
p5 = {'model_5__criterion': ['gini', 'entropy'], "model_5__max_depth" : [2,4,6,8,10,12]}

#Logistic Regression
print("Logistic Regression","\n")
loop_runs = 10
test_accuracy_lr = []
test_roc_auc_lr = []
for i in range(loop_runs):
  inner_cv_lr = KFold(n_splits=10, shuffle=True, random_state=i)
  outer_cv_lr = KFold(n_splits=10, shuffle=True, random_state=i)
  gs_lr = GridSearchCV(estimator=pipeline1, param_grid=p1, cv=inner_cv_lr)
  nested_score_lr = cross_val_score(gs_lr, X=X_train, y=y_train, cv=outer_cv_lr)
  gs_lr.fit(X_train,y_train)
  m_lr = gs_lr.best_estimator_
  y_pred_lr = m_lr.predict(X_test)
  test_accuracy_lr.append(accuracy_score(y_test, y_pred_lr))
  test_roc_auc_lr.append(roc_auc_score(y_test, gs_lr.predict_proba(X_test)[:, 1]))
  nested_scores_lr =  nested_score_lr.mean()
mean_cm = confusion_matrix(y_test, y_pred_lr)
mean_cr = classification_report(y_test, y_pred_lr)
print(nested_score_lr,"\n","\n",nested_scores_lr,"\n","\n",test_accuracy_lr)
for i in range(loop_runs):
    print("Run ", i+1, " test accuracy: ", test_accuracy_lr[i], " test ROC AUC: ", test_roc_auc_lr[i], "\n")
print("Mean test accuracy: ", np.mean(test_accuracy_lr))
print("Mean test ROC AUC: ", np.mean(test_roc_auc_lr))
print("Mean confusion matrix:\n", mean_cm)
print("Mean classification report:\n", mean_cr)

#Random Forest
print("\nRandom Forest\n")
loop_runs = 10
test_accuracy_rf = []
test_roc_auc_rf = []
for i in range(loop_runs):
  inner_cv_rf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  outer_cv_rf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  gs_rf = GridSearchCV(estimator=pipeline2, param_grid=p2, cv=inner_cv_rf)
  nested_score_rf = cross_val_score(gs_rf, X=X_train, y=y_train, cv=outer_cv_rf)
  gs_rf.fit(X_train,y_train)
  m_rf = gs_lr.best_estimator_
  y_pred_rf = m_rf.predict(X_test)
  test_accuracy_rf.append(accuracy_score(y_test, y_pred_rf))
  test_roc_auc_rf.append(roc_auc_score(y_test, y_pred_rf))
  nested_scores_rf =  nested_score_rf.mean()
mean_cm_rf = confusion_matrix(y_test, y_pred_rf)
mean_cr_rf = classification_report(y_test, y_pred_rf)
print(nested_score_rf,"\n","\n",nested_scores_rf,"\n")
for i in range(loop_runs):
    print("Run ", i+1, " test accuracy: ", test_accuracy_rf[i], " test ROC AUC: ", test_roc_auc_rf[i], "\n")
print("Mean test accuracy: ", np.mean(test_accuracy_rf))
print("Mean test ROC AUC: ", np.mean(test_roc_auc_rf))
print("Mean confusion matrix:\n", mean_cm_rf)
print("Mean classification report:\n", mean_cr_rf)

#KNN
print("\nKNN\n")
loop_runs = 10
test_accuracy_kn = []
test_roc_auc_kn = []
for i in range(loop_runs):
  inner_cv_kn = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  outer_cv_kn = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  gs_kn = GridSearchCV(estimator=pipeline3, param_grid=p3, cv=inner_cv_kn)
  nested_score_kn = cross_val_score(gs_kn, X=X_train, y=y_train, cv=outer_cv_kn)
  gs_kn.fit(X_train,y_train)
  m_kn = gs_kn.best_estimator_
  y_pred_kn = m_kn.predict(X_test)
  test_accuracy_kn.append(accuracy_score(y_test, y_pred_kn))
  test_roc_auc_kn.append(roc_auc_score(y_test, y_pred_kn))
  nested_scores_kn =  nested_score_kn.mean()
mean_cm_kn = confusion_matrix(y_test, y_pred_kn)
mean_cr_kn = classification_report(y_test, y_pred_kn)
print(nested_score_kn,"\n","\n", nested_scores_kn,"\n", "\n",test_accuracy_kn)
for i in range(loop_runs):
    print("Run ", i+1, " test accuracy: ", test_accuracy_kn[i], " test ROC AUC: ", test_roc_auc_kn[i], "\n")
print("Mean test accuracy: ", np.mean(test_accuracy_kn))
print("Mean test ROC AUC: ", np.mean(test_roc_auc_kn))
print("Mean confusion matrix:\n", mean_cm_kn)
print("Mean classification report:\n", mean_cr_kn)

#SVC
print("\nSVC\n")
loop_runs = 10
test_accuracy_svc = []
test_roc_auc_svc = []
for i in range(loop_runs):
  inner_cv_svc = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  outer_cv_svc = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  gs_svc = GridSearchCV(estimator=pipeline4, param_grid=p4, cv=inner_cv_svc)
  nested_score_svc = cross_val_score(gs_svc, X=X_train, y=y_train, cv=outer_cv_svc)
  gs_svc.fit(X_train,y_train)
  m_svc = gs_svc.best_estimator_
  y_pred_svc = m_svc.predict(X_test)
  test_accuracy_svc.append(accuracy_score(y_test, y_pred_svc))
  test_roc_auc_svc.append(roc_auc_score(y_test, y_pred_svc))
  nested_scores_svc =  nested_score_svc.mean()
mean_cm_svc = confusion_matrix(y_test, y_pred_svc)
mean_cr_svc = classification_report(y_test, y_pred_svc)
print(nested_score_svc,"\n")
print(nested_scores_svc,"\n")
print(test_accuracy_svc)
for i in range(loop_runs):
    print("Run ", i+1, " test accuracy: ", test_accuracy_svc[i], " test ROC AUC: ", test_roc_auc_svc[i], "\n")
print("Mean test accuracy: ", np.mean(test_accuracy_svc))
print("Mean test ROC AUC: ", np.mean(test_roc_auc_svc))
print("Mean confusion matrix:\n", mean_cm_svc)
print("Mean classification report:\n", mean_cr_svc)

#Decision Tree
print("\nDecision Tree\n")
loop_runs = 10
test_accuracy_dt = []
test_roc_auc_dt = []
for i in range(loop_runs):
  inner_cv_dt = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  outer_cv_dt = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
  gs_dt = GridSearchCV(estimator=pipeline5, param_grid=p5, cv=inner_cv_dt)
  nested_score_dt = cross_val_score(gs_dt, X=X_train, y=y_train, cv=outer_cv_dt)
  gs_dt.fit(X_train,y_train)
  m_dt = gs_dt.best_estimator_
  y_pred_dt = m_dt.predict(X_test)
  test_accuracy_dt.append(accuracy_score(y_test, y_pred_dt))
  test_roc_auc_dt.append(roc_auc_score(y_test, gs_dt.predict_proba(X_test)[:, 1]))
  nested_scores_dt = nested_score_dt.mean()
mean_cm_dt = confusion_matrix(y_test, y_pred_dt)
mean_cr_dt = classification_report(y_test, y_pred_dt)
print(nested_score_dt,"\n", nested_scores_dt,"\n")
for i in range(loop_runs):
    print("Run ", i+1, " test accuracy: ", test_accuracy_dt[i], " test ROC AUC: ", test_roc_auc_dt[i], "\n")
print("Mean test accuracy: ", np.mean(test_accuracy_dt))
print("Mean test ROC AUC: ", np.mean(test_roc_auc_dt))
print("Mean confusion matrix:\n", mean_cm_dt)
print("Mean classification report:\n", mean_cr_dt)

fpr_model1, tpr_model1, _ = roc_curve(y_test, y_pred_lr)
fpr_model2, tpr_model2, _ = roc_curve(y_test, y_pred_rf)
fpr_model3, tpr_model3, _ = roc_curve(y_test, y_pred_kn)
fpr_model4, tpr_model4, _ = roc_curve(y_test, y_pred_svc)
fpr_model5, tpr_model5, _ = roc_curve(y_test, y_pred_dt)
auc_model1 = auc(fpr_model1, tpr_model1)
auc_model2 = auc(fpr_model2, tpr_model2)
auc_model3 = auc(fpr_model3, tpr_model3)
auc_model4 = auc(fpr_model4, tpr_model4)
auc_model5 = auc(fpr_model5, tpr_model5)
plt.plot(fpr_model1, tpr_model1, label='Logistic Regression (AUC = %0.2f)' % auc_model1)
plt.plot(fpr_model2, tpr_model2, label='Random Forest (AUC = %0.2f)' % auc_model2)
plt.plot(fpr_model3, tpr_model3, label='KNN (AUC = %0.2f)' % auc_model3)
plt.plot(fpr_model4, tpr_model4, label='SVC (AUC = %0.2f)' % auc_model4)
plt.plot(fpr_model5, tpr_model5, label='Decision Tree (AUC = %0.2f)' % auc_model5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

