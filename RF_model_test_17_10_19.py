#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:19:45 2019

@author: Ogbomo-Harmitt
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


### IMPORTING DATASET

boston = load_boston()

Features = boston.data

Labels = boston.target

dataset = pd.read_csv('DHCP_RF_Birth_age_model_dataset.txt')
Features =  dataset.iloc[:,2:].values
Labels = dataset.iloc[:,0].values

### IMPLEMENTING RGERSSION MODEL

Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features,Labels,test_size= 0.2, random_state=42)

# REGRESSION MODEL

clf = RandomForestRegressor(n_estimators=100, max_depth=2,random_state=0)
clf.fit(Features_train,Labels_train)
pred = clf.predict(Features_test)
importances =  clf.feature_importances_
indices = np.argsort(importances)[::-1]

print('Mean Absolute error: ', mean_absolute_error(Labels_test, pred))
print('R^2 Score: ', r2_score(Labels_test, pred))

# PLOT REGRESSION MODEL

plt.scatter(Labels_test, pred,marker = 'x')
plt.plot(Labels_test,Labels_test,'r')
plt.xlabel('True Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('GA vs. Regressional RF Model Prediction of GA')
plt.grid()
#plt.savefig('RF_REGRESSION_PLOT_10_19.png')
plt.show()

# FEATURE ANALYSIS 

plt.figure()
plt.title("Feature importances")
plt.bar(range(Features_train.shape[1]), importances[indices],
       color="r", align="center")
plt.xlim([-1, Features_train.shape[1]])
plt.xlabel("Feature Index")
#plt.show()
plt.savefig('RF_REGRESSION_FI.png')
