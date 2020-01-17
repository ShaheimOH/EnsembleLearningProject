#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:30:55 2020

@author: Ogbomo-Harmitt

STACKED GENERALISATION MODEL VER 1.0

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import numpy as np

class Stacked_Gen:
    
    # Intialises parameters for model
    
    def __init__(self,Train_Labels,Train_Features,Data_Labels,Test_Labels = None,Test_Features = None):
        
        self.Train_Labels = Train_Labels
        self.Train_Features = Train_Features
        self.Test_Features = Test_Features
        self.Test_Labels = Test_Labels
        self.Data_Labels = Data_Labels
        # Intialising Sub models
        
        Stacked_Gen.RF_Model = RandomForestClassifier()
        Stacked_Gen.SVC_lin = SVC(kernel = 'linear')
        Stacked_Gen.SVC_Sig = SVC(kernel = 'sigmoid')
        Stacked_Gen.SVC_RBF = SVC(kernel = 'rbf')
        Stacked_Gen.SVC_Poly = SVC(kernel = 'poly')
        Stacked_Gen.Boosting = AdaBoostClassifier()
        Stacked_Gen.Agg_Logreg = LogisticRegression()
        
        
    # Trains model
    
    def TrainModel(self):

        # Train sub models
        
        Stacked_Gen.RF_Model.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen.SVC_lin.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen.SVC_Sig.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen.SVC_RBF.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen.SVC_Poly.fit(self.Train_Features, self.Train_Labels)
        Stacked_Gen.Boosting.fit(self.Train_Features, self.Train_Labels)
        
        # Training Aggregator
        
        Log_feats_agg  = np.zeros((len(self.Train_Labels),3))
        
        Log_feats_agg[:,0] = Stacked_Gen.RF_Model.predict(self.Train_Features)
        Log_feats_agg[:,1] = Stacked_Gen.SVC_lin.predict(self.Train_Features)
        Log_feats_agg[:,2] = Stacked_Gen.Boosting.predict(self.Train_Features)
        
        Stacked_Gen.Agg_Logreg.fit(Log_feats_agg,self.Train_Labels)
        
        
        return 

        
    # Predicting with model 
    
    def ModelPredict(self,Features):
        
        Sub_model_preds  = np.zeros((Features.shape[0],3))
        
        Sub_model_preds[:,0] = Stacked_Gen.RF_Model.predict(Features)
        Sub_model_preds[:,1] = Stacked_Gen.SVC_lin.predict(Features)
        Sub_model_preds[:,2] = Stacked_Gen.Boosting.predict(Features)
        
        
        return Stacked_Gen.Agg_Logreg.predict(Sub_model_preds)
    
    # Feature ranking
    
    def RFE_ranking(self,estimator):
        
        selector = RFE(estimator)
        selector = selector.fit(self.Train_Features,self.Train_Labels)
        return selector.ranking_
    
    def FreatureRanking(self):
        
        # Sub Models Feature Rankings 

        Model_RFE_Rankings = np.zeros((self.Train_Features.shape[1],3))
        Model_RFE_Rankings[:,0] = self.RFE_ranking(Stacked_Gen.RF_Model)
        Model_RFE_Rankings[:,1] = self.RFE_ranking(Stacked_Gen.SVC_lin)
        Model_RFE_Rankings[:,2] = self.RFE_ranking(Stacked_Gen.Boosting)
        
        # Combining Feature Rankings
        
        Combined_Feature_Ranking = np.zeros((self.Train_Features.shape[1],1))
        
        for i in range(self.Train_Features.shape[1]):
            Sum = 0 
            for k in range(3):
                Sum += Model_RFE_Rankings[i,k]
                
            Combined_Feature_Ranking[i] = Sum
            
        return Combined_Feature_Ranking