"""
Created on Thu Oct 17 13:57:27 2019

@author: Ogbomo-Harmitt
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#DATA PREPROCESSING 

dataset = pd.read_csv('DHCP_RF_Birth_age_model_dataset.txt')
Features =  dataset.iloc[:,2:].values
Labels = dataset.iloc[:,0].values

Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features,Labels,test_size= 0.2, random_state=42)

#FEATURE SELECTION - PCA

pca  = PCA()
param  = {'n_components': [10, 20, 30, 40, 50, 60, 70, 80, 85]}
search = GridSearchCV(pca,param, cv = 10)
search.fit(Features_train,Labels_train)
print('Best n components:', search.best_estimator_.get_params()['n_components'])
best_PCA = search.best_estimator_
trans_Features_train = best_PCA.fit_transform(Features_train)  
trans_Features_test = best_PCA.transform(Features_test) 

#REGRESSION MODEL 

clf = RandomForestRegressor(n_estimators=100, max_depth=2,random_state=0)
clf.fit(trans_Features_train,Labels_train)
pred = clf.predict(trans_Features_test)
print(mean_absolute_error(Labels_test, pred))

plt.scatter(Labels_test, pred,marker = 'x')
plt.plot(Labels_test,Labels_test,'r')
plt.xlabel('True Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('GA vs. Regressional RF Model Prediction of GA')
plt.grid()
#plt.savefig('RF_REGRESSION_PLOT_10_19.png')
plt.show()

#CLASSIFIER MODEL 

New_labels = Labels < 37
New_Labels = New_labels *1

Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features,New_Labels,test_size= 0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(trans_Features_train,Labels_train)
pred = clf.predict(trans_Features_test)

# PLOT CLASSIFIER MODEL

labels = ['Term', 'Preterm']
cm = confusion_matrix(Labels_test, pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap = 'gray')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.savefig('RF_CLASSIFICATION_PLOT_10_19.png')
plt.show()