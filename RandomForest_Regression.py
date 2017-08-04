# Random Forest Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting the Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 600 , random_state = 0)
regressor.fit(x,y)

# Predict a new result
y_pred = regressor.predict(6.5)

# Visulising the Random forest Regressor 
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid)),1)
plt.scatter(x , y , color = 'red')
plt.plot(x_grid , regressor.predict(x_grid) ,color = 'blue' )
plt.title ('Random forest Regressoin')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()