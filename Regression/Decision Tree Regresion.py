"""
CART: Classification and Regression Tree

#Random Forest Regression
Scatter plot would be split into segments(leaves),
stops when no info can be added from split or when less pts are in split

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#DT doesn't req feat scaling. Splitting is not done as dataset is small

#DT Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#Predict
y_pred = regressor.predict([[6.5]])

#Viz (Use higher res, this model is stepwise)
X_grid = np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()