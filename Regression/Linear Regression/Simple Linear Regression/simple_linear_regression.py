#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 20:05:32 2018

@author: amogh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

#Extract the dependent and independent variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=1/3,
                                                    random_state=0
                                                    )

#Linear Regression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict test results
y_pred = regressor.predict(X_test)

#Plot train results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Train)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.figure()
plt.show()

#Plot test results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
