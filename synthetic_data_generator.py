"""Synthetic data generator.

Reference: J. Jordon, J. Yoon, M. van der Schaar, 
           "Measuring the quality of Synthetic data for use in competitions," 
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
------------------------------------
Just label flip in here as the example in paper.
Users can replace this function to any other synthetic data generator
such as GAN framework.
"""

# Necessary packages
import numpy as np


def synthetic_data_generator (train_x, train_y, test_x, test_y, prob_flip):
  """Small portion of label flip.
  
  Args:
    - train_x: training features
    - train_y: training labels
    - test_x: testing features
    - test_y: testing labels
    - prob_flip: the portion of labels to be flipped
    
  Returns:
    - train_x_new: generated training features
    - train_y_new: generated training labels
    - test_x_new: generated testing features
    - test_y_new: generated testing labels
  """
  # Training data
  # Determine the indice for flipping
  train_no, _ = train_x.shape
  train_idx = np.random.permutation(train_no)[:int(train_no*prob_flip)]
  
  train_x_new = train_x.copy()
  train_y_new = train_y.copy()
  # Flip labels
  train_y_new[train_idx] = 1-train_y[train_idx]
  
  # Testing data
  # Determine the indice for flipping
  test_no, _ = test_x.shape
  test_idx = np.random.permutation(test_no)[:int(test_no*prob_flip)]
  
  test_x_new = test_x.copy()
  test_y_new = test_y.copy()
  # Flip labels
  test_y_new[test_idx] = 1-test_y[test_idx] 
  
  return train_x_new, train_y_new, test_x_new, test_y_new