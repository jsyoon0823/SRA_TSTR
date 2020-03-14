"""Utility functions.

Reference: J. Jordon, J. Yoon, M. van der Schaar, 
           "Measuring the quality of Synthetic data for use in competitions," 
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
------------------------------------
(1) performance: evaluate predictive model performance.
(2) synthetic_ranking_agreement: SRA in the paper. 
(3) output_visualization: visualize the TSTR metric.
"""

# Necessary packages
from texttable import Texttable
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def performance(test_y, test_y_hat, metric_name):
  """Evaluate predictive model performance.
  
  Args:
    - test_y: original testing labels
    - test_y_hat: prediction on testing data
    - metric_name: 'auc' or 'apr'
    
  Returns:
    - score: performance of the predictive model
  """
  
  assert metric_name in ['auc', 'apr']
  
  if metric_name == 'auc':
    score = roc_auc_score(test_y, test_y_hat)
  elif metric_name == 'apr':
    score = average_precision_score(test_y, test_y_hat)
    
  return score


def synthetic_ranking_agreement (performance_trtr, performance_tsts):
  """Ranking agreement between real and synthetic data.
  
  Args:
    - performance_trtr: performance of various predictive models on TRTR
    - performance_tsts: performance of various predictive models on TSTS
    
  Returns:
    - sra: synthetic ranking agreement
  """
  
  no = len(performance_trtr)  
  nom, denom = 0, 0
  
  for i in range(no):    
    for j in range(no):      
      if j != i:
        order_real = performance_trtr[i] - performance_trtr[j]
        order_synth = performance_tsts[i] - performance_tsts[j]
        
        if order_real * order_synth >= 0:
          nom = nom + 1        
        denom = denom + 1
                    
  sra = round(float(nom)/float(denom), 3)
    
  return sra


def output_visualization (performance_trtr, performance_tstr, models):
  """Visualize the TRTR and TSTR results.
  
  Args:
    - performance_trtr: performance of various predictive models on TRTR
    - performance_tstr: performance of various predictive models on TSTR
    - models: the name of predictive models
  """
    
  # Initialize table
  perf_table = Texttable()    
  first_row = ['models', 'TRTR', 'TSTR']    
  perf_table.set_cols_align(["c" for _ in range(len(first_row))])
  multi_rows = [first_row]
  
  for i in range(len(models)):
    
    curr_row = [models[i], np.mean(performance_trtr[i, :]),
                np.mean(performance_tstr[i, :])]
    multi_rows = multi_rows + [curr_row]
    
  curr_row = ['Average', np.mean(performance_trtr), 
              np.mean(performance_tstr)]
  multi_rows = multi_rows + [curr_row]
        
  perf_table.add_rows(multi_rows)
  print(perf_table.draw())
