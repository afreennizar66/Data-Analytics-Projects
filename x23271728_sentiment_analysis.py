# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# Reading the data

pd_df = pd.read_csv('C://Users//Afreen N//Desktop//ML project//Uber_text_analytics.csv')

# EDA

print(pd_df.head(), "\n")
print(pd_df.columns, "\n")
print(pd_df.info(), "\n")
print(pd_df.describe(), "\n")

df = pd_df[['review_description', 'rating']]
print(df.isna().sum(), "\n")

# Sentiment creation

df['sentiment'] = df['rating'].apply(lambda x:
                                     'negative' if x<=2 else
                                     'neutral' if x==3 else
                                     'positive' if x>=4 else
                                     None)

sns.set_style("whitegrid")
plt.figure(figsize=(8,6))

# Histogram for rating distribution

sns.histplot(df['rating'], kde=True, bins=20, color='cyan')
plt.title("Rating Distribution", fontweight='bold')
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print("\n")

# Pie chart for sentiment distribution

plt.pie(df['sentiment'].value_counts(), labels=df['sentiment'].value_counts().index, autopct='%1.1f%%')
plt.title("Sentiment Distribution", fontweight='bold')
plt.xlabel("Sentiment")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# Data preparation

x = df['review_description']
y = df['sentiment']

# removing punctuations

x = x.str.replace('[^\w\s]', '', regex=True)
print(x.head(), "\n", y.head(), "\n")

# Splitting data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Combining all words
pop_words = ' '.join(x_train)

# Word Cloud
wordcloud = WordCloud(width=700, height=300, colormap='viridis').generate(pop_words)

# Displaying word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Most used words')
plt.show()

vectorizer = TfidfVectorizer(stop_words='english',      # Performing vectorization
                            max_features=5000,
                            strip_accents='unicode',
                            ngram_range=(1, 2),
                            use_idf=True,
                            smooth_idf=True,
                            sublinear_tf=True,
                            min_df=5,
                            max_df=0.85,
                            norm='l2',
                            lowercase=True)

x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)
print(x_train_vector, "\n", x_test_vector,"\n")

# Feature scaling

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_vector, y_train)
print(x_train_resampled, "\n", y_train_resampled, "\n")

# Modelling

# Naive Bayes

p_nb = {'alpha': [0.01, 0.1, 1, 10]}     # Parameters
cv_nb = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)     # Cross validation
gs_nb = GridSearchCV(estimator=MultinomialNB(), param_grid=p_nb, cv=cv_nb)      # Hyperparameter tuning
gs_nb.fit(x_train_resampled, y_train_resampled)      # Model fitting
m_nb = gs_nb.best_estimator_     # Finding best estimator with best parameters
y_pred_nb = m_nb.predict(x_test_vector)     # Prediction

# Naive Bayes Evaluation

print("Results of Naive Bayes model\n")
best_params_nb = gs_nb.best_params_
best_score_nb = gs_nb.best_score_
print("Best parameters for the model: ", best_params_nb, "\n")
print("Best cross validation score: ", best_score_nb, "\n")
print("Accuracy: ", accuracy_score(y_test, y_pred_nb), "\n")
print("Classification report: \n", classification_report(y_test, y_pred_nb), "\n")
cm_nb = confusion_matrix(y_test, y_pred_nb)
print("Confusion matrix: \n", cm_nb, "\n")
for i in range(len(cm_nb)):
  tp = cm_nb[i][i]
  fp = np.sum(cm_nb[i, :]) - tp
  fn = np.sum(cm_nb[:, i]) - tp
  tn = np.sum(cm_nb) - tp - fp - fn
  specificity = tn / (tn + fp)
  print("Class: ", i, "\n")
  print("Specificity: ", specificity, "\n")
kapp_nb = cohen_kappa_score(y_test, y_pred_nb)
print("Kappa score: ", kapp_nb, "\n")

# Random Forest Classifier

p_rfc = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [2, 4]}  # Parameters
cv_rfc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross validation
gs_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=p_rfc, cv=cv_rfc)    # Hyperparameter tuning
gs_rfc.fit(x_train_resampled, y_train_resampled)   # Model fitting
m_rfc = gs_rfc.best_estimator_   # Finding best estimator with best parameters
y_pred_rfc = m_rfc.predict(x_test_vector)  # Prediction

# Random Forest Classifier Evaluation

print("Best parameters: ", gs_rfc.best_params_)
print("Best score: ", gs_rfc.best_score_)
print("Accuracy: ", accuracy_score(y_test, y_pred_rfc))
print("Classification report: \n", classification_report(y_test, y_pred_rfc))
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print("Confusion matrix: \n", cm_rfc, "\n")
for i in range(len(cm_rfc)):
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

# AUC score

y_test_binary = label_binarize(y_test, classes=['negative', 'neutral', 'positive'])
y_pred_nb_binary = label_binarize(y_pred_nb, classes=['negative', 'neutral', 'positive'])
y_pred_rfc_binary = label_binarize(y_pred_rfc, classes=['negative', 'neutral', 'positive'])
auc_model1 = roc_auc_score(y_test_binary, y_pred_nb_binary, average='weighted', multi_class='ovr')   # Calculating auc score
auc_model2 = roc_auc_score(y_test_binary, y_pred_rfc_binary, average='weighted', multi_class='ovr')
print("AUC score for Naive Bayes: ", auc_model1)
print("AUC score for Random Forest Classifier: ", auc_model2)
