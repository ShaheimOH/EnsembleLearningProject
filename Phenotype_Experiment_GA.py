#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:06:26 2020

@author: Ogbomo-Harmitt
"""
import pandas as pd
import numpy as np
from Data_Processing_Ver_1_1 import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# FUNCTIONS

def Get_Mean_Filled_Dataset(Phenotype,Phenotype_IDs,IDs_Array,Features):
    
    Data_Processing_Obj = Data_Processing(IDs_Array,Phenotype,Phenotypes_IDs_Full)
    Features_mean_filled_array = Data_Processing_Obj.Missing_Data(Features,'mean')
    Features_mean_filled_array,Labels_mean = Data_Processing_Obj.Get_Labels_and_Features()
    
    return Features_mean_filled_array,Labels_mean


def Kfold_CV(Features,Labels,Model):
    
    kf = KFold(n_splits=5)
        
    error = []

    for train_index, test_index in kf.split(Features):
            
        X_train, X_test = Features[train_index], Features[test_index]
        y_train, y_test = Labels[train_index], Labels[test_index]
        Model.fit(X_train,y_train)
        pred = Model.predict(X_test)
        error.append(mean_absolute_error(y_test,pred))
        
    return np.mean(error)


# DATA PREPROCESSING 

T1_T2_All_Info = pd.read_csv("T1_T2_info.csv")
Myelin_All_Info = pd.read_csv("Myelin_info_LR.csv")
Curvature_All_Info = pd.read_csv("Curvature_info_LR.csv")
Sulc_All_Info = pd.read_csv("Sulc_info_LR.csv")
SC_All_Info = pd.read_csv("SC_Full_IDs.csv")
Vols_All_Info = pd.read_csv("Vols_info.csv")

T1_T2_Full_IDs = T1_T2_All_Info.iloc[:,1].tolist()
Myelin_Full_IDs = Myelin_All_Info.iloc[:,1].tolist()
Curvature_Full_IDs = Curvature_All_Info.iloc[:,1].tolist()
Sulc_Full_IDs = Sulc_All_Info.iloc[:,1].tolist()
SC_Full_IDs = SC_All_Info.iloc[:,1].tolist()
Vols_Full_IDs = Vols_All_Info.iloc[:,1].tolist()
Diff_Full_IDs = pd.read_csv("Diff_Full_IDs.csv").iloc[:,1].tolist()

T1_T2_Features = pd.read_csv("T1_T2_data_v2.csv").iloc[:,1:].values
Myelin_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withmyelin_rois_LR.pk1").iloc[:,:-1].values
Curvature_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withcurvature_rois_LR.pk1").iloc[:,:-1].values
Sulc_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withsulc_rois_LR.pk1").iloc[:,:-1].values
SC_Features = pd.read_csv("SC_FS_Birth_Features.csv").iloc[:,1:].values
Vols_Features = pd.read_csv("Vol_data_v2.csv").iloc[:,1:].values
Diff_Features = pd.read_csv("Diff_Data_V1.csv").iloc[:,1:].values

# Phenotypes 

Phenotypes_Info = pd.read_excel("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga.xlsx")
Phenotypes_IDs = Phenotypes_Info.iloc[:,0].tolist()
GA_at_Birth = Phenotypes_Info.iloc[:,10].values
GA_at_Scan = Phenotypes_Info.iloc[:,11].values
scanner_validation = Phenotypes_Info.iloc[:,4].values

Phenotypes_IDs_Full = []

for k,ID in enumerate(Phenotypes_IDs):
    
    full_ID = 'sub-' + ID + '_ses-' + str(scanner_validation[k])
    full_ID = full_ID[:-2]
    
    Phenotypes_IDs_Full.append(full_ID)
    
    
IDs_Array = [Myelin_Full_IDs,Curvature_Full_IDs,Sulc_Full_IDs,T1_T2_Full_IDs,SC_Full_IDs,Vols_Full_IDs,Diff_Full_IDs]

Features = [Myelin_Features,Curvature_Features,Sulc_Features,T1_T2_Features,SC_Features,Vols_Features,Diff_Features]


# GA AT BIRTH 

Features_mean_filled_array,Labels = Get_Mean_Filled_Dataset(GA_at_Birth,Phenotypes_IDs_Full,IDs_Array,Features)

T1_T2_Features_Mean_Fil = Features_mean_filled_array[3]
Myelin_Features_Mean_Fil = Features_mean_filled_array[0]
Curvature_Features_Mean_Fil = Features_mean_filled_array[1]
Sulc_Features_Mean_Fil = Features_mean_filled_array[2]
SC_Features_Mean_Fil = Features_mean_filled_array[4]
Vols_Features_Mean_Fil = Features_mean_filled_array[5]
Diff_Features_Mean_Fil = Features_mean_filled_array[6]

All_features_mean_filled  = np.concatenate((Features_mean_filled_array[0],Features_mean_filled_array[1],
                                        Features_mean_filled_array[2],Features_mean_filled_array[3],
                                        Features_mean_filled_array[4],Features_mean_filled_array[5],Features_mean_filled_array[6]),axis = 1)

Feature_Labels = ["T1/T2","Myelin","Curvature","Sulc","SC","Vols","Diff","All"]

Feats = [T1_T2_Features_Mean_Fil,Myelin_Features_Mean_Fil,Curvature_Features_Mean_Fil,
         Sulc_Features_Mean_Fil,SC_Features_Mean_Fil,Vols_Features_Mean_Fil,Diff_Features_Mean_Fil,All_features_mean_filled]

Models =  [GradientBoostingRegressor(random_state=0),RandomForestRegressor(random_state=0)]

Model_Labels = ["Boost","RF"]

Data = []

# Cross-Validation

for Feat_index,Features in enumerate(Feats):
    
    for Model_index,Model in enumerate(Models):
        
        Error = Kfold_CV(Features,Labels,Model)
        
        Model_Label = Model_Labels[Model_index]
        Feature_Label = Feature_Labels[Feat_index]
        
        Current_data = [Feature_Label,Model_Label,Error]
        
        Data.append(Current_data)
        
        print(Feature_Label,"-",Model_Label,": ",Error)


GA_at_Birth_df = pd.DataFrame(Data, columns = ['Modality', 'Model','Error']) 

GA_at_Birth_df.to_csv("Phenotype_Exp_Mean_Fil_GA_birth.csv")


# GA AT SCAN

Features = [Myelin_Features,Curvature_Features,Sulc_Features,T1_T2_Features,SC_Features,Vols_Features,Diff_Features]

Features_mean_filled_array,Labels = Get_Mean_Filled_Dataset(GA_at_Scan,Phenotypes_IDs_Full,IDs_Array,Features)

T1_T2_Features_Mean_Fil = Features_mean_filled_array[3]
Myelin_Features_Mean_Fil = Features_mean_filled_array[0]
Curvature_Features_Mean_Fil = Features_mean_filled_array[1]
Sulc_Features_Mean_Fil = Features_mean_filled_array[2]
SC_Features_Mean_Fil = Features_mean_filled_array[4]
Vols_Features_Mean_Fil = Features_mean_filled_array[5]
Diff_Features_Mean_Fil = Features_mean_filled_array[6]

Feats = [T1_T2_Features_Mean_Fil,Myelin_Features_Mean_Fil,Curvature_Features_Mean_Fil,
         Sulc_Features_Mean_Fil,SC_Features_Mean_Fil,Vols_Features_Mean_Fil,Diff_Features_Mean_Fil,All_features_mean_filled]

All_features_mean_filled  = np.concatenate((Features_mean_filled_array[0],Features_mean_filled_array[1],
                                        Features_mean_filled_array[2],Features_mean_filled_array[3],
                                        Features_mean_filled_array[4],Features_mean_filled_array[5],Features_mean_filled_array[6]),axis = 1)

Data = []

# Cross-Validation

for Feat_index,Features in enumerate(Feats):
    
    for Model_index,Model in enumerate(Models):
        
        Error = Kfold_CV(Features,Labels,Model)
        
        Model_Label = Model_Labels[Model_index]
        Feature_Label = Feature_Labels[Feat_index]
        
        Current_data = [Feature_Label,Model_Label,Error]
        
        Data.append(Current_data)
        
        print(Feature_Label,"-",Model_Label,": ",Error)


GA_at_Scan_df = pd.DataFrame(Data, columns = ['Modality', 'Model','Error']) 

GA_at_Scan_df.to_csv("Phenotype_Exp_Mean_Fil_GA_Scan.csv")