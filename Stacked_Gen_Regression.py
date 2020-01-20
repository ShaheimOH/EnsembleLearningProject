#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:48:08 2020

@author: Ogbomo-Harmitt

STACKED GENERALISATION MODEL VER 2.0 REGRESSION 

Inputs:
    
    Train_Labels - Labels for training data
    
    Train_Features - Features for training data 
    
    Test_Labels - Labels for test data (optional)
    
    Test_Features - Features for training data (optional)
    
    Data_Labels -  Labels of features 
    
    
Outputs:
    
    Trains model from training data
    
    Predicts from features
    
    Ranks features (combined recursive feature selection)

"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import numpy as np
from sklearn import ensemble

class Stacked_Gen_Regression:
    
    # Intialises parameters for model
    
    def __init__(self,Train_Labels,Train_Features):
        
        self.Train_Labels = Train_Labels
        self.Train_Features = Train_Features

        # Intialising Sub models
        
        Stacked_Gen_Regression.RF_Model = RandomForestRegressor()
        Stacked_Gen_Regression.Boost = ensemble.GradientBoostingRegressor()
        Stacked_Gen_Regression.Ridge = Ridge()
        Stacked_Gen_Regression.Agg = Ridge()
        
    # Trains model
    
    def TrainModel(self):

        # Train sub models
        
        X_sub, X_agg, y_sub, y_agg = train_test_split(self.Train_Features, self.Train_Labels, test_size=0.5)
        
        Stacked_Gen_Regression.RF_Model.fit(X_sub, y_sub)
        Stacked_Gen_Regression.Boost.fit(X_sub, y_sub)
        Stacked_Gen_Regression.Ridge.fit(X_sub, y_sub)
        

        # Training Aggregator
        
        Log_feats_agg  = np.zeros((len(y_agg),3))
        
        Log_feats_agg[:,0] = Stacked_Gen_Regression.RF_Model.predict(X_agg)
        Log_feats_agg[:,1] = Stacked_Gen_Regression.Boost.predict(X_agg)
        Log_feats_agg[:,2] = Stacked_Gen_Regression.Ridge.predict(X_agg)
        
        Stacked_Gen_Regression.Agg.fit(Log_feats_agg,y_agg)
        
        Stacked_Gen_Regression.RF_Model.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen_Regression.Boost.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen_Regression.Ridge.fit(self.Train_Features, self.Train_Labels)
        
        return 

        
    # Predicting with model 
    
    def ModelPredict(self,Features):
        
        Sub_model_preds  = np.zeros((Features.shape[0],3))
        
        Sub_model_preds[:,0] = Stacked_Gen_Regression.RF_Model.predict(Features)
        Sub_model_preds[:,1] = Stacked_Gen_Regression.Boost.predict(Features)
        Sub_model_preds[:,2] = Stacked_Gen_Regression.Ridge.predict(Features)
        
        return Stacked_Gen_Regression.Agg.predict(Sub_model_preds)
    
    # Feature ranking
    
    def RFE_ranking(self,estimator):
        
        selector = RFE(estimator)
        selector = selector.fit(self.Train_Features,self.Train_Labels)
        return selector.ranking_
    
    def FreatureRanking(self):
        
        # Sub Models Feature Rankings 

        Model_RFE_Rankings = np.zeros((self.Train_Features.shape[1],3))
        Model_RFE_Rankings[:,0] = self.RFE_ranking(Stacked_Gen_Regression.RF_Model)
        Model_RFE_Rankings[:,1] = self.RFE_ranking(Stacked_Gen_Regression.Boost)
        Model_RFE_Rankings[:,2] = self.RFE_ranking(Stacked_Gen_Regression.Ridge)
        
        # Combining Feature Rankings
        
        Combined_Feature_Ranking = np.zeros((self.Train_Features.shape[1],1))
        
        for i in range(self.Train_Features.shape[1]):
            Sum = 0 
            for k in range(3):
                Sum += Model_RFE_Rankings[i,k]
                
            Combined_Feature_Ranking[i] = Sum
            
            sorted_index = np.argsort(Combined_Feature_Ranking, axis=None, kind='mergesort')
            
            #New_feat_labels = self.Data_Labels(sorted_index)
            
        return  sorted_index

