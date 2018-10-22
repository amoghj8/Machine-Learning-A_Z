#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:44:34 2018

@author: amogh
"""
#Import libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[: , -1:].values

#Regressor
regressor = RandomForestRegressor(random_state=0, n_estimators=10)
regressor.fit(X, y)

#Predict
y_pred = regressor.predict(6.5)

#Plot
X_grid = np.arange(min(X), max(X), .01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y , color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()