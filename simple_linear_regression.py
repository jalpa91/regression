# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:34:26 2017

@author: Jalpa S.
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression 

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#splitting data into training data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3 ,random_state =0)

# foting simple linear regresssion to the training set
regressor = LinearRegression ()
regressor.fit(x_train, y_train)

# prdecting the test data
y_pred = regressor.predict(x_test)

# visulizing the tyraing set result
plt.scatter(x_train,y_train, color='red') 
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, regressor.predict(x_train) , color ='blue')
plt.title('Salary vs Experience(test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
