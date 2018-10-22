#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 04:05:51 2018

@author: amogh
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:, -1:].values

# Train and test data
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.25,
                                                     random_state = 0)


# Scaling the features
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test) 

# Classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

# Predict the values
y_pred = knn.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

