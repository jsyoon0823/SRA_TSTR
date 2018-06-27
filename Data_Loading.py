# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split


def Data_Loading_MAGGIC(data_name, it, test_fraction, k, year):
    
    if data_name == 'maggic':
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/Synthetic_Data/Data/Maggic_extracted_imputed_'+str(it+1) + '.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                
        Time_hor = 365*year

        Test = Test[(Test['days_to_fu']>Time_hor) | ((Test['days_to_fu']<=Time_hor) & (Test['death_all']==1))]
        N  = Test.shape[0]
        Test['Label_death_horizon']=[0]*N
        Test.loc[((Test['days_to_fu']<=Time_hor) & (Test['death_all']==1)), 'Label_death_horizon'] =1   
        X_test = Test.drop(['death_all', 'days_to_fu', 'Label_death_horizon'], axis=1)
        Y_test = Test['Label_death_horizon']
            
        ### 2. Training Data Generation            
        Train = Train[(Train['days_to_fu']>Time_hor) | ((Train['days_to_fu']<=Time_hor) & (Train['death_all']==1))]
        N  = Train.shape[0]
        Train['Label_death_horizon']=[0]*N
        Train.loc[((Train['days_to_fu']<=Time_hor) & (Train['death_all']==1)), 'Label_death_horizon'] =1   
        X_train = Train.drop(['death_all', 'days_to_fu', 'Label_death_horizon'], axis=1)
        Y_train = Train['Label_death_horizon']            
        
        bin_idx = 9   
        
    return X_test, Y_test, X_train, Y_train, bin_idx
        
def Data_Loading_Credit(data_name, it, test_fraction, k, year):
            
    #%%    
    if data_name == 'credit':
        
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/Synthetic_Data/Data/kaggle_credit.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                
        Y_test = Test['Class']
        X_test = Test.drop(['Class'], axis=1)
                    
        Y_train = Train['Class']
        X_train = Train.drop(['Class'], axis=1)            
                   
        bin_idx = 0      
                              
        return X_test, Y_test, X_train, Y_train, bin_idx

def Data_Loading_Cancer(data_name, it, test_fraction, k, year):               
    #%%  
    if data_name == 'cancer':
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/Synthetic_Data/Data/kaggle_cancer_'+str(it+1) + '.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                
        Y_test = Test['Biopsy']
        X_test = Test.drop(['Biopsy'], axis=1)
                    
        Y_train = Train['Biopsy']
        X_train = Train.drop(['Biopsy'], axis=1)            
                   
        bin_idx = 0     
                
        return X_test, Y_test, X_train, Y_train, bin_idx
        
    #%% 
def Data_Loading_Heart(data_name, it, test_fraction, k, year): 
    if data_name == 'heart':
    
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/Synthetic_Data/Data/Heart_Trans_Final'+str(it+1) + '.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                
        Time_hor = 365*year

        Test = Test[(Test["'Survival'"]>Time_hor) | ((Test["'Survival'"]<=Time_hor) & (Test["'Censor'"]==0))]
        N  = Test.shape[0]
        Test['Label_death_horizon']=[0]*N
        Test.loc[((Test["'Survival'"]<=Time_hor) & (Test["'Censor'"]==0)), 'Label_death_horizon'] =1   
        X_test = Test.drop(["'Censor'", "'Survival'", 'Label_death_horizon'], axis=1)
        Y_test = Test['Label_death_horizon']
            
        ### 2. Training Data Generation            
        Train = Train[(Train["'Survival'"]>Time_hor) | ((Train["'Survival'"]<=Time_hor) & (Train["'Censor'"]==0))]
        N  = Train.shape[0]
        Train['Label_death_horizon']=[0]*N
        Train.loc[((Train["'Survival'"]<=Time_hor) & (Train["'Censor'"]==0)), 'Label_death_horizon'] =1   
        X_train = Train.drop(["'Censor'", "'Survival'", 'Label_death_horizon'], axis=1)
        Y_train = Train['Label_death_horizon']            
        
        bin_idx = 15    
        
        return X_test, Y_test, X_train, Y_train, bin_idx
            
            
                #%% 
def Data_Loading_Heart_Wait(data_name, it, test_fraction, k, year): 
    if data_name == 'heart_wait':
    
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/Synthetic_Data/Data/Heart_Wait_Final.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                
        Time_hor = 365*year

        Test = Test[(Test["'Survival'"]>Time_hor) | ((Test["'Survival'"]<=Time_hor) & (Test["'Censor'"]==0))]
        N  = Test.shape[0]
        Test['Label_death_horizon']=[0]*N
        Test.loc[((Test["'Survival'"]<=Time_hor) & (Test["'Censor'"]==0)), 'Label_death_horizon'] =1   
        X_test = Test.drop(["'Censor'", "'Survival'", 'Label_death_horizon'], axis=1)
        Y_test = Test['Label_death_horizon']
            
        ### 2. Training Data Generation            
        Train = Train[(Train["'Survival'"]>Time_hor) | ((Train["'Survival'"]<=Time_hor) & (Train["'Censor'"]==0))]
        N  = Train.shape[0]
        Train['Label_death_horizon']=[0]*N
        Train.loc[((Train["'Survival'"]<=Time_hor) & (Train["'Censor'"]==0)), 'Label_death_horizon'] =1   
        X_train = Train.drop(["'Censor'", "'Survival'", 'Label_death_horizon'], axis=1)
        Y_train = Train['Label_death_horizon']            
        
        bin_idx = 2
        
        return X_test, Y_test, X_train, Y_train, bin_idx
    #%%            