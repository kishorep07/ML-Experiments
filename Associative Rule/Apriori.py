"""
Apriori : People who bought also bought
Ex: Movie Recommendations
Netflix, Amazon etc uses more complex version of this

Support(I) = #Transactions containing I / #Total Transactions
Confidence(I1->I2) = #Transcations containing I1 and I2 / #Trans containing I1
Lift (I1->I2) = Confidence(I1->I2) / Support(I2)

Set min support and confidence, take subsets of transactions having higher support
Take all rules of these subsets having higher confidence than min confidence
Sort rules by dec lift

apyori expects input as a list of list
Support: 3 times a day = 3*7/7500 = 0.0028
"""

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) #First obs not header
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) #Experimental

#Viz Results
results = list(rules) #already sorted by relevance
