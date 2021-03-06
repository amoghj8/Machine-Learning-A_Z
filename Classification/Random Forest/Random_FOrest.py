#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 04:05:51 2018

@author: amogh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

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
classifier = RandomForestClassifier(criterion='entropy', random_state=0,
                                    n_estimators=10)
classifier.fit(X_train,y_train)

# Predict the values
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
