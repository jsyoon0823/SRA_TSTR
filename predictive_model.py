"""Predictive model.

Reference: J. Jordon, J. Yoon, M. van der Schaar, 
           "Measuring the quality of Synthetic data for use in competitions," 
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
import numpy as np

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor


def predictive_model (train_x, train_y, test_x, model_name):
  """Predictive model define, train, and test.
  
  Args:
    - train_x: training features
    - train_y: training labels
    - test_x: testing features
    - model_name: predictive model name
    
  Returns:
    - model: trained predictive model
    - test_y_hat: prediction on testing set
  """
  
  assert model_name in ['logisticregression', 'nn', 'randomforest',
                        'gaussiannb', 'bernoullinb', 'multinb',
                        'svmlin', 'gbm', 'extra trees',
                        'lda','passive aggressive', 'adaboost',
                        'bagging', 'xgb']
  
  # Define model
  if model_name == 'logisticregression':
    model = LogisticRegression()
  elif model_name == 'nn':    
    model = MLPClassifier(hidden_layer_sizes=(200,200))
  elif model_name == 'randomforest':      
    model = RandomForestClassifier()
  elif model_name == 'gaussiannb':  
    model = GaussianNB()
  elif model_name == 'bernoullinb':  
    model = BernoulliNB()
  elif model_name == 'multinb':  
    model = MultinomialNB()
  elif model_name == 'svmlin':         
    model = svm.LinearSVC()
  elif model_name == 'gbm':         
    model = GradientBoostingClassifier()    
  elif model_name == 'extra trees':
    model = ExtraTreesClassifier(n_estimators=20)
  elif model_name == 'lda':
    model = LinearDiscriminantAnalysis() 
  elif model_name == 'passive aggressive':
    model = PassiveAggressiveClassifier()
  elif model_name == 'adaboost':
    model = AdaBoostClassifier()
  elif model_name == 'bagging':
    model = BaggingClassifier()
  elif model_name == 'xgb':
    model = XGBRegressor()                                    
  
  # Train & Predict
  if model_name in ['svmlin', 'Passive Aggressive']: 
    model.fit(train_x, train_y)
    test_y_hat = model.decision_function(test_x)
    
  elif model_name == 'xgb':
    model.fit(np.asarray(train_x), train_y)
    test_y_hat = model.predict(np.asarray(test_x))
                
  else:
    model.fit(train_x, train_y)
    test_y_hat = model.predict_proba(test_x)[:,1]
    
  return model, test_y_hat
            