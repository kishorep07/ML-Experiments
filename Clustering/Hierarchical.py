"""
Heirarchical Clustering
Same as K Means but but diff process
2 types: Agglomerative & Divisive

Agglomerative:
Make each pt a single pt cluster (N Clusters)
Take 2 closest data pts and make them 1 cluster
Take 2 closest (closest pts/furthest pts/avg/centroid dist) clusters and make them 1 cluster
[repeat until only 1 cluster]

Dendogram:
Shows how dissmilar (euclidian dist) 2 pts (connected) are (by height)
We can set a dissmilarity threshold to limit K
Measure the vertical dist b/w joins at diff levels, highest pt, count and set levels
"""

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3,4]].values #income and score

#Plotting Dendogram to find K
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #ward method tries to minimize variance
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Dist')
plt.show()

#Fitting HC
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward') #K=5 from dendogram
y_hc = hc.fit_predict(X)

#Viz Clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of clients')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

