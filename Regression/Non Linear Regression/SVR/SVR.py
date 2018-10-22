#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:32:46 2018

@author: amogh
"""
#Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#Load the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1:].values

#Feature Scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_svr = sc_X.fit_transform(X)
y_svr = sc_Y.fit_transform(y)
svr = SVR(kernel = 'rbf')
regressor = svr.fit(X_svr, y_svr)

#Predict any value
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array(6.5))))

#Plot results
plt.scatter(X_svr , y_svr, color = 'red')
plt.plot(X_svr, regressor.predict(X_svr), color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Position vs Salary')
plt.show()