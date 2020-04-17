#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:21:45 2020

@author: Ogbomo-Harmitt
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

class Stacked_Gen_Regression:
    
    # Intialises parameters for model
    
    def __init__(self,Train_Labels,Train_ALL_Features):
        
        self.Train_Labels = Train_Labels
        self.Train_ALL_Features = Train_ALL_Features
        self.All_anatomical_feats = self.Train_ALL_Features[:,600:774]

        # Intialising Sub models
        
        Stacked_Gen_Regression.Agg = Ridge()
        Stacked_Gen_Regression.Sub_RF = RandomForestRegressor(random_state=0)
        Stacked_Gen_Regression.Sub_Boost = ensemble.GradientBoostingRegressor(random_state=0)
        Stacked_Gen_Regression.Sub_Boost2 = ensemble.GradientBoostingRegressor(random_state=0)
        
        # Feature Importances       
        self.RF_Feature_Importances = None
        self.Boost_Feature_Importances = None
        self.Boost2_Feature_Importances = None
        
        # Training Model
        
        sub_models = [Stacked_Gen_Regression.Sub_Boost,
                      Stacked_Gen_Regression.Sub_RF,Stacked_Gen_Regression.Sub_Boost2]
        Features_train = [self.Train_ALL_Features,self.All_anatomical_feats,self.All_anatomical_feats]
        Sub_model_preds = np.zeros((self.Train_ALL_Features.shape[0],3))
        
        counter = 0
        
        for model in sub_models:
            model.fit(Features_train[counter],self.Train_Labels)
            Sub_model_preds[:,counter] = model.predict(Features_train[counter])
            counter += 1
        
        Stacked_Gen_Regression.Agg.fit(Sub_model_preds,self.Train_Labels)
        
        
    def ModelPredict(self,feat_a):
        
        feat_b = feat_a[:,600:774]
        Features = [feat_a,feat_b,feat_b]
        sub_models = [Stacked_Gen_Regression.Sub_Boost,Stacked_Gen_Regression.Sub_RF
                      ,Stacked_Gen_Regression.Sub_Boost2] 
        Sub_model_preds = np.zeros((feat_a.shape[0],3))
        
        counter = 0
        
        for model in sub_models:
            
            Sub_model_preds[:,counter] = model.predict(Features[counter])
            counter += 1
            
            if model == Stacked_Gen_Regression.Sub_RF:
                    
                self.RF_Feature_Importances = model.feature_importances_
                    
            elif model == Stacked_Gen_Regression.Sub_Boost:
            
                self.Boost_Feature_Importances = model.feature_importances_
                
            else:
        
                self.Boost2_Feature_Importances = model.feature_importances_
                
        return Stacked_Gen_Regression.Agg.predict(Sub_model_preds)
    
    def Get_Feature_Importance(self):
        
        return self.RF_Feature_Importances,self.Boost_Feature_Importances,self.Boost2_Feature_Importances