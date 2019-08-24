"""

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #vec to matrix (1:2)
y = dataset.iloc[:, 2].values   #pos is encoded into level

# No splitting, because set is small & acc for future data imp
# feature scaling done by lib

#Fitting Linear Regression to data
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting polynomial regression to data
from sklearn.preprocessing import PolynomialFeatures #will add additional poly terms
poly_reg = PolynomialFeatures(degree = 4) #Play around with degree
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#Viz Linear Reg
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Viz Poly Reg
X_grid = np.arange(min(X),max(X),0.1) #inc res. ret evenely spaced arr withing given range (vector)
X_grid = X_grid.reshape((len(X_grid),1)) #reshape. Used to conv to matrix
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predict linear reg
lin_reg.predict([[6.5]])

#Predict poly reg
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))