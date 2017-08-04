# Support Vector Regression (SVR)

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Imporrting the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# fitting SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

# Predict a new result 
y_pred = sc_y.inverse_transform( regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Visualising the SVR result
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color= 'blue') 
plt.title('SVR Model')
plt.xlabel('Position Label')
plt.ylabel('Salaries')
plt.show()