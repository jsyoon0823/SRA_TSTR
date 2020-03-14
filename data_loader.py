"""Data loader.

Reference: J. Jordon, J. Yoon, M. van der Schaar, 
           "Measuring the quality of Synthetic data for use in competitions," 
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
----------------------------------------
Loads two public Kaggle datasets with median imputation and MinMax normalization.

(1) Kaggle Credit Card Fraud Detection:
  - https://www.kaggle.com/mlg-ulb/creditcardfraud
(2) Kaggle Cervical Cancer Risk Classification: 
  - https://www.kaggle.com/loveall/cervical-cancer-risk-classification
"""

# Necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def data_loader(data_name, test_fraction):
  """Load datasets and divide into training and testing sets.
  
  Args:
    - data_name: 'cancer' or 'credit'
    - test_fraction: the fraction of testing data (between 0 and 1)
    
  Returns:
    - train_x: training features
    - train_y: training labels
    - test_x: testing features
    - test_y: testing labels
  """
  
  assert data_name in ['cancer','credit']
  
  # Define file_name and label_name
  if data_name == 'cancer':
    file_name = 'data/kag_risk_factors_cervical_cancer.csv'
    label_name = 'Biopsy'    
  elif data_name == 'credit':
    file_name = 'data/creditcard.csv' # Need to download before
    label_name = 'Class'
    
  # Load data
  ori_data = pd.read_csv(file_name, sep = ',')
  
  # Median imputation
  if data_name == 'cancer':
    ori_data = ori_data.replace('?', np.nan)
    ori_data = ori_data.fillna(ori_data.median())
    ori_data = ori_data.astype(float)
    
  # Normalization (MinMaxScaler)
  for col_name in ori_data.columns:
    ori_data[col_name] = ori_data[col_name] - np.min(ori_data[col_name])
    ori_data[col_name] = ori_data[col_name] / np.max(ori_data[col_name] + 1e-8)
    
  # Train / Test division
  idx = range(len(ori_data))
  train_data, test_data, train_idx, test_idx = \
  train_test_split(ori_data, idx, test_size=test_fraction)  
        
  # Define feature / label
  train_y = train_data[label_name]
  train_x = train_data.drop([label_name], axis = 1)
  
  test_y = test_data[label_name]
  test_x = test_data.drop([label_name], axis = 1)
  
  # Rounding and convert to numpy array
  train_x = np.asarray(train_x)
  train_y = np.asarray(train_y.round())
  
  test_x = np.asarray(test_x)
  test_y = np.asarray(test_y.round())    
                              
  return train_x, train_y, test_x, test_y 