
"""
Created on Mon Oct  7 15:16:16 2019

@author: Ogbomo-Harmitt

Supervised learning - optimal model script

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math 

class Supervised_Learning_Trial:
    
    Process_data = 0
    
    def __init__(self,X,Y):
        self.X = X
        self.Y = y

    #DUMMY CLASS FUNCTION 
        
    def Pefrorm_trial(self):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
        
        for i in range(0,5):
            
            if i == 0:
                
                # MULTI-LINEAR REGRESSION 
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_name = 'MULTI-LINEAR REGRESSION'

            if i == 1:
                
                # RANDOM-FORREST
                model = RandomForestRegressor(n_estimators = 1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_name = 'RANDOM FORREST'
                
            if i == 2:
                
                model = SVR(kernel = 'linear')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_name = 'SUPPORT VECTOR MACHINE (LINEAR)'
            
            if i == 3:
                
                model = SVR(kernel = 'RBF')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_name = 'SUPPORT VECTOR MACHINE (RBF)'

            if i == 4:
                
                model = SVR(kernel = 'Polynomial')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_name = 'SUPPORT VECTOR MACHINE (POLYNOMIAL)'
                
            print(model_name,' MODEL IS FITTED')
            
            Score = r2_score(y_test,y_pred)
            RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
            
            print(model_name,' R^2 SCORE = ',Score)
            print(model_name,' RMSE = ',RMSE)
            
            
            
            
            
        
            
            
    
                
            
    