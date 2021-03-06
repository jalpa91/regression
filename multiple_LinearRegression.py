# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:56:57 2017

@author: Jalpa S.
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using backward Elimination
X = np.append(arr = np.ones((50,1)), values = X, axis = 1)
x_opt = X[:, [0,1,2,3,4]]
regressor_ols = sm.OLS(endog = y , exog = x_opt).fit()
regressor_ols.summary()
x_opt = X[:, [0,1,2,3]]
regressor_ols = sm.OLS(endog = y , exog = x_opt).fit()
regressor_ols.summary()
x_opt = X[:, [0,1,2]]
regressor_ols = sm.OLS(endog = y , exog = x_opt).fit()
regressor_ols.summary()





