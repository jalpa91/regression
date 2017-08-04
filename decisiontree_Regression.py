# Decisition Tree Regression for one dimension 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Fitting the decision tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predict the decision tree regression to the dataset
y_pred = regressor.predict(6.5)

# Visulising the decision tree regression
x_grid = np.arange(min(X) , max(X), 0.01 )
x_grid = x_grid.reshape((len(x_grid)),1)
plt.scatter(X , y , color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color ='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Decision Tree Regression')
plt.show()
