#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:16:39 2018

@author: amogh
"""
#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

#Load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1:].values

#Regressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#Predict y 
y_pred = regressor.predict(6.5)

#PLot results
X_grid = np.arange(min(X), max(X), .01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red' )
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Truth or Bluff  (Decision Tree)')
plt.show()