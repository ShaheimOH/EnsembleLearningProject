#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 08:36:18 2019

@author: Ogbomo-Harmitt
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## READING AND UNPICKLING DATASET 

unpickled_df = pd.read_pickle("TRAIN_TEST_ga_regressionremove_myelinwrois_voronoi_with_labels.pk1")
y = unpickled_df.iloc[:,-1].values
X =  unpickled_df.iloc[:,:300].values
X = np.transpose(X)

# #DATA PREPROCESSING

sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)

## 2D-PCA IMPLEMENTATION

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(X_scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

## PLOTTING GRAPH

print(pca.explained_variance_ratio_)
plt.figure(1)
plt.scatter(principalComponents[:,0],principalComponents[:,1])
plt.title('PCA Brain surface data')
plt.ylabel('Principle Component 2')
plt.xlabel('Principle Component 1')
#plt.show()
plt.savefig('PCA_BRAIN_SURFACE.png')

## K-MEANS CLUSTERING 

# Use silhouette score

range_n_clusters = list (range(2,10))

best_score = 0 
best_num_cluster = 0

for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(principalDf)
    centers = clusterer.cluster_centers_
    score = silhouette_score (principalDf, preds, metric='euclidean')
    if score > best_score:
        best_score = score
        best_num_cluster = n_clusters
        

# Plotting K-means with 2 clusters
        
kmeans =  KMeans(n_clusters = 2,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(principalComponents)
X1 = principalComponents

plt.figure(3)
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.title('K-Means Brain surface data with 2 clusters')
plt.ylabel('Principle Component 2')
plt.xlabel('Principle Component 1')
#plt.show()
plt.savefig('K_MEANS_2_BRAIN_SURFACE.png')

# Plotting K-means with 4 clusters

kmeans =  KMeans(n_clusters = 4,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(principalComponents)
X1 = principalComponents

plt.figure(4)
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 3')
plt.scatter(X1[y_kmeans == 3, 0], X1[y_kmeans == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')
plt.title('K-Means Brain surface data with 4 clusters')
plt.ylabel('Principle Component 2')
plt.xlabel('Principle Component 1')
#plt.show()
plt.savefig('K_MEANS_4_BRAIN_SURFACE.png')