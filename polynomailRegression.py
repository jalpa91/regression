# Polynomial Regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#fiting Linear regression to the data set
Lin_reg = LinearRegression()
Lin_reg.fit(x,y)

# Fiting Ploynomial Regression to the data set
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
Lin_reg2 = LinearRegression()
Lin_reg2.fit(x_poly,y)

#visulising the linear regression results
plt.scatter(x,y,color='red')
plt.plot(x, Lin_reg.predict(x),color='blue')
plt.title('Linear Regression ')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visulising the polynomial Regression  results
plt.scatter(x,y,color='red')
plt.plot(x, Lin_reg2.predict(poly_reg.fit_transform(x)), color ='blue')
plt.title('Polynomial Regression ')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# prediting a new result with Linear regression
# we need predicting result for level 6.5
Lin_reg.predict(6.5)

# Predicting a new result with Polynomial regression 
Lin_reg2.predict(poly_reg.fit_transform(6.5))
