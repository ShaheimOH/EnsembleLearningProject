#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:21:22 2020

@author: Ogbomo-Harmitt
"""

from Stacked_Gen_Regression import *
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

from sklearn.model_selection import RepeatedKFold

## READING AND UNPICKLING DATASET / DATA PRE-PROCESSING 

unpickled_df = pd.read_pickle("TRAIN_TEST_ga_regressionremove_myelinwrois_voronoi_with_labels.pk1")
unpickled_df = unpickled_df.drop(unpickled_df.index[428])
Labels = unpickled_df.iloc[:,-1].values
Features =  unpickled_df.iloc[:,:299].values

# REPEATED STRATIFIED K-FOLD SPLIT

rkf = RepeatedKFold()

Results = np.zeros((50,4))

i = 0


Ridge_Model = Ridge()
RF_Model = RandomForestRegressor()
Boost_Model = ensemble.GradientBoostingRegressor()

for train_index, test_index in rkf.split(Features, Labels):

    X_train, X_test = Features[train_index], Features[test_index]
    y_train, y_test = Labels[train_index], Labels[test_index]

    # STACKED GENERALIZATION 
    
    Test_obj = Stacked_Gen_Regression(y_train,X_train)
    Test_obj.TrainModel()
    y_hat = Test_obj.ModelPredict(X_test)
    Results[i,0] = mean_absolute_error(y_test, y_hat)
    
    # RIDGE MODEL
    
    Ridge_Model.fit(X_train,y_train)
    y_pred = Ridge_Model.predict(X_test)
    Results[i,1] = mean_absolute_error(y_test, y_pred)
    
    # RANDOM FOREST MODEL
    
    RF_Model.fit(X_train,y_train)
    y_pred = RF_Model.predict(X_test)
    Results[i,2] = mean_absolute_error(y_test, y_pred)
    
    
    # BOOSTING REGRESSOR MODEL
    
    Boost_Model.fit(X_train,y_train)
    y_pred = Boost_Model.predict(X_test)
    Results[i,3] = mean_absolute_error(y_test, y_pred)
    
    i += 1

print('Stacked Generalisation Cross-Validation score: ',np.mean(Results[:,0]))
print('Ridge Cross-Validation score: ',np.mean(Results[:,1]))
print('Random Forest Cross-Validation score: ',np.mean(Results[:,2]))
print('Boost Cross-Validation score: ',np.mean(Results[:,3]))
