import pandas as pd
import numpy as np
import torch
import os

from metrics import *

"""
This file contains the methods used to remove training examples from the dataset. Each method takes in the config,
the dataframe of the results, and the number of examples to remove. It returns the indexes of the examples to remove.
"""

def random(cfg, df, num_examples):
    """ Randomly remove a percentage(or number) of the training data point """
    len_dataset = len(df['image_id'].unique())
    idxs = np.random.choice(len_dataset, num_examples, replace=False)
    return idxs, []

def high_loss(cfg, df, num_examples):
    """ Remove the examples with the highest loss """
    cumulative_loss = df.groupby('image_id', as_index=False).apply(calc_cumulative_loss)
    cumulative_loss.columns = ['image_id', 'cumulative_loss']
    idxs = np.argsort(df['cumulative_loss'])[::-1][:num_examples]
    return idxs['image_id'].values, []

def num_flips(cfg, df, num_examples):
    """ Remove the examples with the highest number of flips """
    num_flips = df.groupby('image_id', as_index=False).apply(calc_num_flips)
    num_flips.columns = ['image_id', 'num_flips']
    idxs = np.argsort(df['num_flips'])[::-1][:num_examples]
    return idxs['image_id'].values, []

def high_fslt_low_ssft(cfg, first_df, second_df, num_examples):
    """ Second Split Forgetting: remove the examples with the highest FSLT and lowest SSFT """
    fslt = first_df.groupby('image_id', as_index=False).apply(calc_fslt)
    fslt.columns = ['image_id', 'fslt']
    ssft = second_df.groupby('image_id', as_index=False).apply(calc_ssft)
    ssft.columns = ['image_id', 'ssft']
    results = pd.concat([fslt, ssft], axis=1)
    results = results.loc[:,~results.columns.duplicated()].copy()
    return results.sort_values('fslt', ascending=False).sort_values('ssft', ascending=True)[:num_examples]['image_id'].values, []