# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:37:07 2022

@author: Abhishek
"""

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

def write_to_file(test_data, predictions, name):
    test_data["predictions"] = predictions
    sub = test_data[["Citation Text", "predictions"]]
    sub.to_csv(f"predictions_{name}.csv", index=False)

# We must use multiclass classification algorithms as we have more than two classifications
# Positive, Negative and Neutral classes
# Use KNN, Naive Bayes, Decision Trees, Support Vector Machines, Random Forest, Gradient Boosting

dataset = pd.read_csv('citation_sentiment_corpus.csv',
                      usecols=['Sentiment', 'Citation'])
citations = pd.read_csv('citations-data.csv', usecols=['Citation Text'])

vectorizer = TfidfVectorizer(max_features=2048)
vectorizer2 = TfidfVectorizer(max_features=2048)

vectorizer.fit(dataset.iloc[:, -1].values)

vectorizer2.fit(citations.iloc[:, -1].values)
testing = vectorizer2.transform(citations.iloc[:, -1].values)

features = vectorizer.transform(dataset.iloc[:, -1].values)  # Citation text
labels = dataset.iloc[:, -2].values  # Positive/Negative/Neutral

'''
textclassifier =Pipeline([
  ('vect', CountVectorizer()),
   ('tfidf', TfidfTransformer()),
   ('smote', SMOTE(random_state=12)),
   ('mnb', MultinomialNB(alpha =0.1))
])
'''
oversample = SMOTE()

# ct = OneHotEncoder(categories='auto')
# t_labels = np.array(ct.fit_transform(labels.reshape(-1,1)))
'''
0 - Negative
1 - Neutral
2 - Positive
'''
le = LabelEncoder()
t_labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features, t_labels, test_size=0.25, random_state=0)
X_train, y_train = oversample.fit_resample(X_train, y_train.ravel())

# print(X_train)
# Logistic Regression
logisticRegr = LogisticRegression(
    multi_class='multinomial', solver='newton-cg')
logisticRegr.fit(X_train, y_train)
logisticPredictions = logisticRegr.predict(X_test)
print(
    f"Logistic Regression: {accuracy_score(y_test, logisticPredictions)*100:.3f}%")
logisticPredictions2 = logisticRegr.predict(testing)
print(
    f"Logistic Regression Testing: {len(logisticPredictions2)} :: {(logisticPredictions2 == 1).sum()}\n")
write_to_file(citations, logisticPredictions2, 'LOG')

# K-Nearest Neighbours
kClassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
kClassifier.fit(X_train, y_train)
knnPredictions = kClassifier.predict(X_test)
print(f"KNN: {accuracy_score(y_test, knnPredictions)*100:.3f}%")
knnPredictions2 = kClassifier.predict(testing)
print(
    f"KNN Testing: {len(knnPredictions2)} :: {(knnPredictions2 == 1).sum()}\n")
write_to_file(citations, knnPredictions2, 'KNN')

NBclassifier = GaussianNB()
NBclassifier.fit(X_train.todense(), y_train)
NBpredictions = NBclassifier.predict(X_test.todense())
print(f"Naive Bayes: {accuracy_score(y_test, NBpredictions)*100:.3f}%")
NBpredictions2 = NBclassifier.predict(testing.todense())
print(
    f"Naive Bayes Testing: {len(NBpredictions2)} :: {(NBpredictions2 == 1).sum()}\n")
write_to_file(citations, NBpredictions2, 'NB')

DTclassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTclassifier.fit(X_train, y_train)
DTpredictions = DTclassifier.predict(X_test)
print(f"Decision Tree: {accuracy_score(y_test, DTpredictions)*100:.3f}%")
DTpredictions2 = DTclassifier.predict(testing.todense())
print(
    f"Decision Tree Testing: {len(DTpredictions2)} :: {(DTpredictions2 == 1).sum()}\n")
write_to_file(citations, DTpredictions2, 'DT')

ETCclassifier = ExtraTreesClassifier(
    n_estimators=5, criterion='entropy', max_features=2)
ETCclassifier.fit(X_train, y_train)
ETCpredictions = ETCclassifier.predict(X_test)
print(
    f"Extra Tree Classifier: {accuracy_score(y_test, ETCpredictions)*100:.3f}%")
ETCpredictions2 = ETCclassifier.predict(testing)
print(
    f"ETC Classifier Testing: {len(ETCpredictions2)} :: {(ETCpredictions2 == 1).sum()}\n")
write_to_file(citations, ETCpredictions2, 'ETC')

'''
# SVM is very slow compared to the rest
SVMClassifier = SVC(gamma='auto')
SVMClassifier.fit(X_train, y_train)
SVMpredictions = SVMClassifier.predict(X_test)
print(
    f"SVM: {accuracy_score(y_test, SVMpredictions)*100:.3f}%")
SVMpredictions2 = SVMClassifier.predict(testing)
print(
    f"SVM Testing: {len(SVMpredictions2)} :: {(SVMpredictions2 == 1).sum()}\n")
write_to_file(citations, SVMpredictions2, 'SVM')
'''

'''
for X,Y in zip(logisticPredictions,y_test):
    print("Model Score:", X, "actual score:", Y)
'''

'''
Output:

    Logistic Regression: 83.608%
    Logistic Regression Testing: 605 :: 597

    KNN: 26.419%
    KNN Testing: 605 :: 4

    Naive Bayes: 69.231%
    Naive Bayes Testing: 605 :: 603

    Decision Tree: 80.907%
    Decision Tree Testing: 605 :: 498

    Extra Tree Classifier: 82.005%
    ETC Classifier Testing: 605 :: 513

    SVM: 80.861%
    SVM Testing: 605 :: 604
'''
