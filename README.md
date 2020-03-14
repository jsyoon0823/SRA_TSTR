# Codebase for "SRA-TSTR (Synthetic Randking Agreement - Train on Synthetic, Test on Real)"

Authors: James Jordon, Jinsung Yoon, Mihaela van der Schaar

Paper: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
       "Measuring the quality of Synthetic data for use in competitions," 
       KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
       (https://arxiv.org/abs/1806.11345)
       
Contact: jsyoon0823@gmail.com

This directory contains implementations of SRA-TSTR framework.
To run the pipeline, simply run python3 -m main_sra_tstr.py.

Note that any model architecture can be used as the synthetic data generator 
such as GAN. 

## Stages of SRA-TSTR framework:

-   Load dataset
-   Generate synthetic data
-   Train model on 3 settings:
    (1) TRTR: Train on Real, Test on Real
    (2) TSTR: Train on Synthetic, Test on Real
    (3) TSTS: Train on Synthetic, Test on Synthetic
-   Compute two metrics:
    (1) TSTR: Compare the performance with TRTR
    (2) SRA: Synthetic ranking agreement (Compare ranking agreement between TRTR and TSTR)

### Command inputs:

-   data_name: data to be used ('credit' or 'cancer')
-   num_fold: the number of fold (interations)
-   test_fraction: the fraction of testing data
-   prob_flip: the fraction of testing labels to be flipped
-   metric_name: metrics to evaluate the predictive models ('auc' or 'apr')

### Example command

```shell
$ python3 main_sra_tstr.py 
--data_name cancer --num_fold 4 --test_fraction 0.25 --prob_flip 0.2
--metric_name auc
```

### Outputs

-   TRTR Performance
-   TSTR Performance
-   SRA Performance