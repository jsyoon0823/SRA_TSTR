"""
Jinsung Yoon (05/13/2018)
Synthetic Data Generation & Test on the Predictive Model &
"""
#%% Import packages
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBRegressor

#%% Import Function
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/2018_Research/KDD')

# Data Loading File
import Data_Loading
# GAN Framework to generate sample

#%% Parameters
# Survival cutoff
year = 1
# k fold cross validation
num_folds = 4
test_fraction = 0.25

# Diverse Predictive models
models = ['logisticregression','randomforest', 'gaussiannb','bernoullinb','svmlin',
          'Extra Trees','LDA', 'AdaBoost','Bagging','gbm','NN','xgb']

# Number of models
L = len(models)

# Diverse datasets
datasets = ['maggic','heart','credit','cancer','heart_wait']
data_name = datasets[0]

## Imputation Number
M=1

for pp in range(11):
    prob = 0.05*pp

#######

    #%% Output Initialization
    AUC_ar   = [[0 for x in range(num_folds)] for y in range(L)]
    AUPRC_ar   = [[0 for x in range(num_folds)] for y in range(L)]
    
    AUC_ar_New   = [[0 for x in range(num_folds)] for y in range(L)]
    AUPRC_ar_New   = [[0 for x in range(num_folds)] for y in range(L)]
    
    # Just for Feature_No extraction
    
    if data_name == 'maggic':
        X_test, Y_test, X_train, Y_train, _ = Data_Loading.Data_Loading_MAGGIC(data_name, 0, test_fraction, 0, year)
    elif data_name == 'heart':
        X_test, Y_test, X_train, Y_train, _ = Data_Loading.Data_Loading_Heart(data_name, 0, test_fraction, 0, year)
    elif data_name == 'credit':
        X_test, Y_test, X_train, Y_train, _ = Data_Loading.Data_Loading_Credit(data_name, 0, test_fraction, 0, year)
    elif data_name == 'cancer':
        X_test, Y_test, X_train, Y_train, _ = Data_Loading.Data_Loading_Cancer(data_name, 0, test_fraction, 0, year)
    elif data_name == 'heart_wait':
        X_test, Y_test, X_train, Y_train, _ = Data_Loading.Data_Loading_Heart_Wait(data_name, 0, test_fraction, 0, year)
    
    Feature_No = len(X_train.columns)
    
    
    #%% Training Testing Results
    ##### Iteration Start
    #### For each CV Folds
    for k in range(num_folds):
         
        ### For each imputed data matrix
        for i in range(M):
            
                
            ### 1. Testing Data Generation
            if data_name == 'maggic':
                X_test, Y_test, X_train, Y_train, bin_idx = Data_Loading.Data_Loading_MAGGIC(data_name, i, test_fraction, k, year)
            elif data_name == 'heart':
                X_test, Y_test, X_train, Y_train, bin_idx = Data_Loading.Data_Loading_Heart(data_name, i, test_fraction, k, year)
            elif data_name == 'credit':
                X_test, Y_test, X_train, Y_train, bin_idx = Data_Loading.Data_Loading_Credit(data_name, i, test_fraction, k, year)
            elif data_name == 'cancer':
                X_test, Y_test, X_train, Y_train, bin_idx = Data_Loading.Data_Loading_Cancer(data_name, i, test_fraction, k, year)
            elif data_name == 'heart_wait':
                X_test, Y_test, X_train, Y_train, bin_idx = Data_Loading.Data_Loading_Heart_Wait(data_name, i, test_fraction, k, year)
            
            #%% 2. New Data Generation
            Train_No = len(Y_train)
            Test_No = len(Y_test)
            
            Train_Y_Add = np.random.binomial(1, prob, Train_No)
            Y_train_New = np.abs(Y_train - Train_Y_Add)
            
            Test_Y_Add = np.random.binomial(1, prob, Test_No)
            Y_test_New = np.abs(Y_test - Test_Y_Add)
    
            #%% Prediction  
    
            if(i==0):
                
                Predict_av = np.zeros([len(Y_test), L])   
                Predict_av_New = np.zeros([len(Y_test), L])   
            
            for j in range(L):
                
                print('num_fold: ' + str(k+1) + '  Algorithm' + str(j+1))
        
                model_name = models[j]            
            
                if model_name == 'logisticregression':
                    model         = LogisticRegression()
       
                if model_name == 'NN':    
                    model        = MLPClassifier(hidden_layer_sizes=(200,200))
    
                if model_name == 'randomforest':      
                    model        = RandomForestClassifier()
                   
                if model_name == 'gaussiannb':  
                    model        = GaussianNB()
    
                if model_name == 'bernoullinb':  
                    model        = BernoulliNB()
    
                if model_name == 'multinb':  
                    model        = MultinomialNB()
          
                if model_name == 'svmlin':         
                    model        = svm.LinearSVC()
           
                if model_name == 'gbm':         
                    model         = GradientBoostingClassifier()    
    
                if model_name == 'Extra Trees':
                    model =  ExtraTreesClassifier(n_estimators=20)
                    
                if model_name == 'LDA':
                    model =  LinearDiscriminantAnalysis() 
    
                if model_name == 'Passive Aggressive':
                    model =   PassiveAggressiveClassifier()
     
                if model_name == 'AdaBoost':
                    model =   AdaBoostClassifier()
    
                if model_name == 'Bagging':
                    model =   BaggingClassifier()
                                        
                if model_name == 'xgb':
                    model =   XGBRegressor()                                    
                                            
                if(model_name=='svmlin' or model_name=='Passive Aggressive'): 
                    model.fit(X_train, Y_train_New)
                    Predict = model.decision_function(X_test)
                    
                elif (model_name =='xgb'):
                    model.fit(np.asarray(X_train), Y_train_New)
                    Predict = model.predict(np.asarray(X_test))
                    
                else:
                    model.fit(X_train, Y_train_New)
                    Predict = model.predict_proba(X_test)[:,1]
                
                Predict_av[:,j] = Predict_av[:,j] + Predict
            
        for j in range(L):
            AUC_ar[j][k] = metrics.roc_auc_score(Y_test, Predict_av[:,j]/M)
            AUPRC_ar[j][k] = metrics.average_precision_score(Y_test, Predict_av[:,j]/M)
                            
    #%% AUROC / AUPRC Output
    Output = np.zeros([L,4])
    Output_New = np.zeros([L,4])
    
    for j in range(L):
        Output[j,0] = round(np.mean(AUC_ar[j]),4)
        Output[j,1] = round((2*np.std(AUC_ar[j])/np.sqrt(num_folds)),4)
        
        Output[j,2] = round(np.mean(AUPRC_ar[j]),4)
        Output[j,3] = round((2*np.std(AUPRC_ar[j])/np.sqrt(num_folds)),4)
        
    #%%# Save Results
    # AUROC/AUPRC
    # Setting B
    file_name = '/home/jinsung/Documents/Jinsung/2018_Research/KDD/Result/' + data_name + '_p_' + str(int(100*prob)) + '.csv'
    np.savetxt(file_name, Output)
