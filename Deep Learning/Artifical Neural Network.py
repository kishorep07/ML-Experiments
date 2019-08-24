"""
Neuron: Takens n input (independant vars with diff wts) and provides and o/p

Activation function: Thershold, Sigmoid, rectifier(popular), hyperbolic (similar to sigmoid)
usuall hidden layer rectifier and o/p signmoid

Working: Multiple nodes in hidden 
Learning: Compare predicted and actual to tweak weights
           EPOC: Once go trhough data

(Batch) Gradient Descent: Adjust wts after go trhough all rows, very time consuming
Stochastic Graident Dedscent: Adjust wts after each row

Tensorflow and  Theano: Computation libs, uses GPU for speed
Keras: Wrap of both, lesser lines of code
"""

# Data Preprocessing

#Import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

#Encoding categorical vars
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #country

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #gender

onehotencoder = OneHotEncoder(categorical_features = [1]) #dummy vars
X = onehotencoder.fit_transform(X).toarray()

#split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feat scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Make ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

#init ANN
classifier = Sequential() #Sequence of layers
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) #hidden rectifier, output sigmoid



 