"""
Multi Armed Bandit Problem
Single Armed Bandit = Slot machine
How do u play them to maxmimize returns?
Each machine has a distribution associated with it, goal is to figure that out
regret: non optimal method
Modern App: Advertising (find which method is best)

Example:
multiple ads avail to disp to user, reward given if user clicks on ad
we consider the average reward to decide which is best

At start assume a distribution for ads (emprical avg of confidence band)
Confidence band: Will include the actual distribution
the first few rounds are trial rounds to det actual dist
dist goes down, confidence band goes up and vice versa, also shrinks

Dataset tells us what happens on each round for all ads, not possbile IRL

Random Slection
select a rand ad at each row and calc reward
distribution is uniform as data is randomly selected
"""

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implement Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward += reward

#Viz the result (Histogram)
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

#Implement UCB (Without Package)
import math
N = 10000                           #total test
d=10                                #no of ads
ads_selected = []
numbers_of_selections = [0] * d      #vector of size d init to 0
sums_of_rewards = [0] * d           # each add
total_reward_ucb = 0

for n in range(N):
    ad =0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:    #first 10, we create upper bound for each ad
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
            
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward_ucb += reward
    
#Viz UCB
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


            
            