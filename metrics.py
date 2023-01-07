import os
import pandas as pd
import numpy as np

def fslt(df):
    """ First-Split Learning Time """
    df = df.sort_values('epoch', ascending=False)
    incorrect = df[df['prediction'] != df['label']]
    if len(incorrect) == 0:
        return 0
    else:
        return incorrect.iloc[0]['epoch'] + 1 # +1 because we want the first epoch where the model was correct

def ssft(df):
    """ Second-Split Forgetting Time """
    df = df.sort_values('epoch', ascending=False)
    correct = df[df['prediction'] == df['label']]
    if len(correct) == 0:
        return 0
    else:
        return correct.iloc[0]['epoch'] + 1 # +1 because we want the first epoch where the model was incorrect

def num_forgetting(df):
    """ Number of Forgetting Events (when acc of example descreases b/t epochs)"""
    last_pred = -1
    num_forgetting = 0
    for i, row in df.iterrows():
        if i == 0:
            last_pred = row['prediction']
        elif row['prediction'] != last_pred and last_pred == row['label']:
            num_forgetting += 1
        last_pred = row['prediction']
    return num_forgetting

def num_flips(df):
    """ Number of Flip Events (when pred of example changes b/t epochs)"""
    last_pred = -1
    num_forgetting = 0
    for i, row in df.iterrows():
        if i == 0:
            last_pred = row['prediction']
        elif row['prediction'] != last_pred:
            num_forgetting += 1
        last_pred = row['prediction']
    return num_forgetting

def cumulative_learning(df):
    """ Cumulative Learning Accuracy (sum of acc of example over epochs) """
    return len(df[df['prediction'] == df['label']])

def cumulative_confidence(df):
    """ Cumulative Learning Confidence (sum of confidence of example over epochs) """
    return df['conf'].sum()

def cumulative_loss(df):
    """ Cumulative Loss (sum of loss of example over epochs) """
    return df['loss'].sum()

def conf_delta(df):
    """ Sum of Differences in prediction confidence b/t epochs """
    last_conf = -1
    conf_delta = 0
    for i, row in df.iterrows():
        if i == 0:
            last_conf = row['conf']
        else:
            conf_delta += abs(row['conf'] - last_conf)
        last_conf = row['conf']
    return conf_delta

def num_loss_flips(df):
    """ Number of time the loss goes up b/t epochs """
    last_loss = -1
    num_loss_flips = 0
    for i, row in df.iterrows():
        if i == 0:
            last_loss = row['loss']
        elif row['loss'] > last_loss:
            num_loss_flips += 1
        last_loss = row['loss']
    return num_loss_flips

def max_loss_epoch(df):
    """ Epoch with the highest loss """
    return df[df['loss'] == df['loss'].max()]['epoch'].iloc[0]