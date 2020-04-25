#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:55:56 2020

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

class Stacked_Gen_Regression:
    
    # Intialises parameters for model
    
    def __init__(self,Train_Labels,Train_ALL_Features):
        
        self.Train_Labels = Train_Labels
        self.Train_ALL_Features = Train_ALL_Features
        self.All_anatomical_feats = self.Train_ALL_Features[:,600:774]
        self.All_surface_feats = self.Train_ALL_Features[:,0:600]
        self.intensities = Train_ALL_Features[:,600:687]
        self.vols = Train_ALL_Features[:,687:]

        # Intialising Sub models
        
        Stacked_Gen_Regression.Agg = SVR(kernel = 'linear')
        Stacked_Gen_Regression.Sub_Boost = ensemble.GradientBoostingRegressor(random_state=0)
        Stacked_Gen_Regression.Sub_RF = RandomForestRegressor(random_state=0)
        Stacked_Gen_Regression.Sub_RF2 = RandomForestRegressor(random_state=0)
        Stacked_Gen_Regression.Sub_RF3 = RandomForestRegressor(random_state=0)
        
        
        # Feature Importances
        
        self.Agg_Feature_Importances = None
        self.Boost_Feature_Importances = None
        self.RF_Feature_Importances = None
        self.RF2_Feature_Importances = None
        self.RF3_Feature_Importances = None
    
        
        # Training Model
        
        sub_models = [Stacked_Gen_Regression.Sub_Boost,
                      Stacked_Gen_Regression.Sub_RF,Stacked_Gen_Regression.Sub_RF2,Stacked_Gen_Regression.Sub_RF3]
        
        Features_train = [self.Train_ALL_Features,self.intensities,self.vols,self.All_surface_feats]
        
        Sub_model_preds = np.zeros((self.Train_ALL_Features.shape[0],4))
        
        counter = 0
        
        for model in sub_models:
            model.fit(Features_train[counter],self.Train_Labels)
            Sub_model_preds[:,counter] = model.predict(Features_train[counter])
            counter += 1
        
        Stacked_Gen_Regression.Agg.fit(Sub_model_preds,self.Train_Labels)
        
        self.Agg_Feature_Importances = Stacked_Gen_Regression.Agg.coef_
        self.Boost_Feature_Importances = Stacked_Gen_Regression.Sub_Boost.feature_importances_
        self.RF_Feature_Importances =  Stacked_Gen_Regression.Sub_RF.feature_importances_
        self.RF2_Feature_Importances = Stacked_Gen_Regression.Sub_RF2.feature_importances_
        self.RF3_Feature_Importances = Stacked_Gen_Regression.Sub_RF3.feature_importances_
        
        
    def ModelPredict(self,feat_all):
        
        feat_intensity = feat_all[:,600:687]
        feat_vol = feat_all[:,687:]
        feat_surface = feat_all[:,0:600]
        
        Features = [feat_all,feat_intensity,feat_vol,feat_surface]
        
        sub_models = [Stacked_Gen_Regression.Sub_Boost,
                      Stacked_Gen_Regression.Sub_RF,Stacked_Gen_Regression.Sub_RF2,Stacked_Gen_Regression.Sub_RF3]
        
        Sub_model_preds = np.zeros((feat_all.shape[0],4))
        
        counter = 0
        
        for model in sub_models:
            
            Sub_model_preds[:,counter] = model.predict(Features[counter])
            counter += 1
                
        return Stacked_Gen_Regression.Agg.predict(Sub_model_preds)
    
    def Get_Feature_Importance(self):
        
        return self.Agg_Feature_Importances,self.Boost_Feature_Importances,self.RF_Feature_Importances,self.RF2_Feature_Importances,self.RF3_Feature_Importances