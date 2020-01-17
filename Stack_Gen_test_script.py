#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:11:28 2020

@author: Ogbomo-Harmitt
"""

from Stacked_Gen import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import numpy as np

## READING AND UNPICKLING DATASET / DATA PRE-PROCESSING 

unpickled_df = pd.read_pickle("TRAIN_TEST_ga_regressionremove_myelinwrois_voronoi_with_labels.pk1")
Data_labels = list(unpickled_df.columns.values)
Data_labels = Data_labels[0:300]
Data_labels = np.array(Data_labels)
Labels = unpickled_df.iloc[:,-1].values

# 1 = Preterm & 0 = Term 

New_Labels =  Labels < 37

Features =  unpickled_df.iloc[:,:300].values

X_train, X_test, y_train, y_test = train_test_split(Features, New_Labels, test_size=0.4)

# Stacked Generalisation Model

Test_obj = Stacked_Gen(y_train,X_train,Data_labels)

Test_obj.TrainModel()

y_hat = Test_obj.ModelPredict(X_test)

y_test = y_test*1
y_hat = y_hat*1

print("STACK GEN Accuracy Score = ",accuracy_score(y_test, y_hat))
print("R^2 Score = ",r2_score(y_test,y_hat))

# Random Forest Model 

RF_Model = RandomForestClassifier()
RF_Model.fit(X_train,y_train)
y_pred = RF_Model.predict(X_test)
y_pred = y_pred*1

print("RF Accuracy Score = ",accuracy_score(y_test, y_pred),"R^2 Score",r2_score(y_test, y_pred))

# Feature Ranking

c = Test_obj.FreatureRanking()
