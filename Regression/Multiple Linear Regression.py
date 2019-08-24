"""
Here, y=b(const) + b1x1 + b2x2 + .... + bnxn
There are certain assumptions of linear regression, which should be verified
Dummy var: dont include all vars, no bias beacuse 0 part will be compensated in const b
            the model will not perf ideally. *Always emit one dummy var per set*
Building Models: We dont't use all column to predict result (garbage in garbage out)
                Easy to understand smaller variables
1)All in: use all vars. Case: Prior knowledge, have to, prep for backward elimination
2)Backward Elimination: Select significance level (SL), if P>SL for a var fit model w/o var (high P more error)
3)Forward Selection: Select SL, fit all simple reg models and select one with lowest P. s3:Fit all possbile models with one extra var. Choose pred with smallest val, if p<sl goto s3 else keep prev model
4)BiDir Elimination: Select SLEnter SLStay. Perf next step of FS(SLEnter), Perf all steps of BE(SLStay). Repeat until no new var can enter/exit 
5)All Possbile: Very thorough. Build all possbile reg models (2^n-1), Select one with best criterion (ex: Akaike Criterion)

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy var trap
X=X[:,1:] #remove the first col, actually not needed, lib will tc

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling will be taken care by lib

#Fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict
y_pred = regressor.predict(X_test)

#Optimize using Backward elimination
import statsmodels.formula.api as sm #this model takes x0=1 for b0 const (called as intercept)
X = np.append(arr=np.ones(shape=(50, 1), dtype=int), values=X, axis = 1) #add col of 1s in the beg of X

X_opt = X[:, [0,1,2,3,4,5]] #rn contains all indp vars
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]] #2nd has highest p value > sl(0.05), so removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]] #1st has highest p value > sl(0.05), so removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Ordinary least squares
regressor_OLS.summary()


X_opt = X[:, [0,3,5]] #2nd has highest p value > sl(0.05), so removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,3]] #2nd has highest p value > sl(0.05), so removed
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Ordinary least squares
regressor_OLS.summary() #R&D Spent most significant