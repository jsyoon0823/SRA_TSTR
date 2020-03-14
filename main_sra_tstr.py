"""SRA-TSTR Main function

Reference: J. Jordon, J. Yoon, M. van der Schaar, 
           "Measuring the quality of Synthetic data for use in competitions," 
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
------------------------------------
(1) Load data
(2) Generate synthetic data
(3) Train model on 3 settings:
  - TRTR: Train on Real, Test on Real
  - TSTR: Train on Synthetic, Test on Real
  - TSTS: Train on Synthetic, Test on Synthetic
(4) Compute two metrics:
  - TSTR: Compare the performance with TRTR
  - SRA: Synthetic ranking agreement (Compare ranking agreement between TRTR and TSTR)
"""


# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_loader import data_loader
from synthetic_data_generator import synthetic_data_generator
from predictive_model import predictive_model
from utils import synthetic_ranking_agreement
from utils import performance, output_visualization


def main (args):  
  """SRA-TSTR Main function.
  
  Args:
    - data_name: data to be used in this experiment
    - num_fold: the number of fold (interations)
    - test_fraction: the fraction of testing data
    - prob_flip: the fraction of testing labels to be flipped
    - metric_name: metrics to evaluate the predictive models
  """
    
  # Diverse Predictive models
  models = ['logisticregression','randomforest', 'gaussiannb',
            'bernoullinb','svmlin','extra trees','lda', 'adaboost',
            'bagging','gbm','nn','xgb']
  
  performance_trtr = np.zeros([len(models), args.num_fold])
  performance_tstr = np.zeros([len(models), args.num_fold])
  performance_tsts = np.zeros([len(models), args.num_fold])
  
  for i in range(args.num_fold):
                   
    # Load original data
    train_x_real, train_y_real, test_x_real, test_y_real = \
    data_loader(args.data_name, args.test_fraction)  
    # Generate synthetic data
    train_x_synth, train_y_synth, test_x_synth, test_y_synth = \
    synthetic_data_generator(train_x_real, train_y_real, 
                             test_x_real, test_y_real, args.prob_flip)
        
    for j in range(len(models)):
      
      model_name = models[j]            
      
      print('num fold: ' + str(i+1) + ', predictive model: ' + model_name)
          
      _, test_y_hat_trtr = predictive_model(train_x_real, train_y_real, 
                                            test_x_real, model_name)
      _, test_y_hat_tstr = predictive_model(train_x_synth, train_y_synth, 
                                            test_x_real, model_name)
      _, test_y_hat_tsts = predictive_model(train_x_synth, train_y_synth, 
                                            test_x_synth, model_name)
                  
      performance_trtr[j, i] = performance(test_y_real, test_y_hat_trtr, 
                                           args.metric_name)
      performance_tstr[j, i] = performance(test_y_real, test_y_hat_tstr, 
                                           args.metric_name)
      performance_tsts[j, i] = performance(test_y_synth, test_y_hat_tsts, 
                                           args.metric_name)
                        
  #%%
  print('TSTR Performance (' + args.metric_name + ')')
  output_visualization (performance_trtr, performance_tstr, models)
  
  result = synthetic_ranking_agreement(np.mean(performance_trtr, 1),
                                       np.mean(performance_tsts, 1))
  print('SRA Performance: ' + str(result))
  
    
##  
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['credit','cancer'],
      default='cancer',
      type=str)
  parser.add_argument(
      '--num_fold',
      help='the number of fold (interations)',
      default=4,
      type=int)
  parser.add_argument(
      '--test_fraction',
      help='the fraction of testing data',
      default=0.25,
      type=float)
  parser.add_argument(
      '--prob_flip',
      help='the fraction of testing labels to be flipped',
      default=0.2,
      type=float)
  parser.add_argument(
      '--metric_name',
      help='metrics to evaluate the predictive models',
      choices=['auc','apr'],
      default='auc',
      type=str)
  
  args = parser.parse_args() 
  
  # Call main function  
  main(args)