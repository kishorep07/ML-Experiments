"""
y = b0 + b1*x1 (y=> dep, x=>indep, b1=> slope)
min(Sum(obs - actual)^2) -> Ordinary least squares
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0) #20train and 10test

#Library will take care of feature scaling

#Fitting Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict
y_pred = regressor.predict(X_test)

#Visualizing
plt.scatter(X_train, y_train, color='red') #train pts
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #prediction line
plt.title('Salary Vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red') #test pts
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #prediction line
plt.title('Salary Vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

