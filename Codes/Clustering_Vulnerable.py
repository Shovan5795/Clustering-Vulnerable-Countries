# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:53:25 2020

@author: shovon5795
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


dataset = pd.read_csv(r"C:\Users\shovon5795\Desktop\Research\Mamun sir saha reno\Liklihood Dataset.csv")

#Dataset column selection


X = dataset.iloc[:,0:9].values

#applying PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components=None)
X= pca1.fit_transform(X)
explained_variance1 = pca1.explained_variance_ratio_


from sklearn.decomposition import PCA
pca1 = PCA(n_components=2)
X= pca1.fit_transform(X)
explained_variance1 = pca1.explained_variance_ratio_

from sklearn.cluster import KMeans
#Within class sum of squares

WCSS = []

#The Elbow Method to detect number of clusters


for i in range(1,13):
    kmeans1=KMeans(n_clusters=i, init='k-means++', n_init = 10, max_iter=300, random_state=0)
    kmeans1.fit(X)
    WCSS.append(kmeans1.inertia_)

  


plt.plot(range(1,13), WCSS)
plt.title("Number of Iterations Needed for Scores")
plt.xlabel("No of clusters")
plt.ylabel("WCSS")
plt.show()

#Fitting dataset into clusters using kmeans
c=5

kmeans1=KMeans(n_clusters=c, init='k-means++', n_init = 10, max_iter=300, random_state=0)
ykmeans1=kmeans1.fit_predict(X)

plt.scatter(X[ykmeans1 == 4, 0], X[ykmeans1 == 4, 1], s=50, c='orange', label = 'Very Low')
plt.scatter(X[ykmeans1 == 1, 0], X[ykmeans1 == 1, 1], s=50, c='green', label = 'Low')
plt.scatter(X[ykmeans1 == 0, 0], X[ykmeans1 == 0, 1], s=50, c='red', label = 'Medium')
plt.scatter(X[ykmeans1 == 2, 0], X[ykmeans1 == 2, 1], s=50, c='magenta', label = 'High')
plt.scatter(X[ykmeans1 == 3, 0], X[ykmeans1 == 3, 1], s=50, c='blue', label = 'Very High')

plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=200, c='yellow', label = 'centroids')
plt.title("Vulnerability Cluster of Different Countries using K-means++ (Liklihood Scores)")
plt.xlabel("1st PCA of Survey Attributes")
plt.ylabel("2nd PCA of Survey Attributes")
plt.legend()
plt.show()

print(f'Silhouette Score(n=5): {silhouette_score(X, ykmeans1)}')

####Hierarchical Clustering####
#Cluster Count

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendogram of vulnerability clusters score")
plt.xlabel('No. of clusters')
plt.ylabel('Euclidian Distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc1 = AgglomerativeClustering(n_clusters = c, affinity = 'euclidean', linkage = 'ward')
yhc1 = hc1.fit_predict(X)


plt.scatter(X[yhc1 == 4, 0], X[yhc1 == 4, 1], s=50, c='orange', label = 'Very Low')
plt.scatter(X[yhc1 == 1, 0], X[yhc1 == 1, 1], s=50, c='green', label = 'Low')
plt.scatter(X[yhc1 == 0, 0], X[yhc1 == 0, 1], s=50, c='red', label = 'Medium')
plt.scatter(X[yhc1 == 3, 0], X[yhc1 == 3, 1], s=50, c='magenta', label = 'High')
plt.scatter(X[yhc1 == 2, 0], X[yhc1 == 2, 1], s=50, c='blue', label = 'Very High')


#plt.scatter(X[ykmeans1 == 5, 0], X[ykmeans1 == 5, 1], s=50, c='yellow', label = 'cluster6')

plt.title("Vulnerability Cluster of Different Countries using Agglomerative Clustering (Liklihood Scores)")
plt.xlabel("1st PCA of Survey Attributes")
plt.ylabel("2nd PCA of Survey Attributes")
plt.legend()
plt.show()
print(f'Silhouette Score(n=5): {silhouette_score(X, yhc1)}')


#BIRCH CLUSTERING
#from numpy import arange

from sklearn.cluster import Birch
brc=Birch(threshold=1.0, branching_factor=25,n_clusters=c)
ybrc = brc.fit_predict(X)
plt.scatter(X[ybrc == 1, 0], X[ybrc == 1, 1], s=50, c='orange', label = 'Very Low')
plt.scatter(X[ybrc == 4, 0], X[ybrc == 4, 1], s=50, c='green', label = 'Low')
plt.scatter(X[ybrc == 3, 0], X[ybrc == 3, 1], s=50, c='red', label = 'Medium')
plt.scatter(X[ybrc == 0, 0], X[ybrc == 0, 1], s=50, c='magenta', label = 'High')
plt.scatter(X[ybrc == 2, 0], X[ybrc == 2, 1], s=50, c='blue', label = 'Very High') 
plt.title("Vulnerability Cluster of Different Countries using BIRCH (Liklihood Scores) with Threshold 1.0")
plt.xlabel("1st PCA of Survey Attributes")
plt.ylabel("2nd PCA of Survey Attributes")
plt.legend()
plt.show()
print(f'Silhouette Score(n=5): {silhouette_score(X, ybrc)}')


dataset["Vulnerability Birch Score 1.0 Thrs"] = ybrc
dataset["Vulnerability Kmeans Score"] = ykmeans1
dataset["Vulnerability Agglomerative Score"] = yhc1
dataset.to_csv(r"C:\Users\shovon5795\Desktop\Research\Mamun sir saha reno\Liklihood Dataset.csv", index = False)


