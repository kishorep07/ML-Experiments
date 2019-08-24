"""
#Logistic Regression (Used to predict probablity)
Action is discrete (Yes or No)
Instead of predicting wether Y or No, we will predict the probablity
Below and above Y&N are very likely to say No&Yes respectively

Substitute y=mx+b into the sigmoid function and solve to attain a smooth curve
Probablity(p_hat) Y Axis and Values X Axis
Prediction: <50% No, >50% Yes (linear always straight line)

"""
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values #Age and Salary
y = dataset.iloc[:, 4].values #Purchase?

#Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature Scaling, no need for y beacuse classification 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predict
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix #its a function, not class
cm = confusion_matrix(y_test, y_pred)



#Viz Train Set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#Color all pts in the graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
#Predict all pts
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap=ListedColormap(('red', 'green')))
#set limit of axis
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plot all train pts
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set== j, 0], X_set[y_set == j, 1], c=ListedColormap(('black','white'))(i), label=j)
#details
plt.title('Logistic Regression(Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimate Salary')
plt.legend()
plt.show()


#Viz Test Set
X_set, y_set = X_test, y_test
#Color all pts in the graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
#Predict all pts
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap=ListedColormap(('red', 'green')))
#set limit of axis
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plot all test pts
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set== j, 0], X_set[y_set == j, 1], c=ListedColormap(('black','white'))(i), label=j)
#details
plt.title('Logistic Regression(Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimate Salary')
plt.legend()
plt.show()