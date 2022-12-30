import os
import sys
import time
import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import ast
from sklearn.metrics import confusion_matrix
from scipy import stats
from tqdm import tqdm

import omegaconf

def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            #print(running_key_temp, value)
            flattened_dict[running_key_temp] = value
    return flattened_dict

def read_unknowns(unknown_list):
    """
    input is of form ['--METHOD.MODEL.LR=0.001758722642964502', '--METHOD.MODEL.NUM_LAYERS=1']
    """
    ret = {}
    for item in unknown_list:
        key, value = item.split('=')
        try:
            value = ast.literal_eval(value)
        except:
            print("MALFORMED ", value)
        k = key[2:]
        ret[k] = value
    return ret

def nest_dict(flat_dict, sep='.'):
    """Return nested dict by splitting the keys on a delimiter.

    >>> from pprint import pprint
    >>> pprint(nest_dict({'title': 'foo', 'author_name': 'stretch',
    ... 'author_zipcode': '06901'}))
    {'author': {'name': 'stretch', 'zipcode': '06901'}, 'title': 'foo'}
    """
    tree = {}
    for key, val in flat_dict.items():
        t = tree
        prev = None
        for part in key.split(sep):
            if prev is not None:
                t = t.setdefault(prev, {})
            prev = part
        else:
            t.setdefault(prev, val)
    return tree

def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

def evaluate(predictions, labels, groups=[], label_names=None, num_augmentations=1):
    """
    Gets the evaluation metrics given the predictions and labels. 
    num_augmentations is for test-time augmentation, if its set >1, we group predictions by 
    num_augmentations and take the consesus as the label
    """
    if num_augmentations > 1:
        predictions_aug = predictions.reshape((int(len(predictions)/num_augmentations), num_augmentations))
        print("aug shape ", predictions_aug.shape, " label shape ", labels.shape)
        majority_pred = []
        for i, group in enumerate(predictions_aug):
            majority_pred.append(stats.mode(group)[0])
        predictions = np.array(majority_pred)
        print("new pred shape ", predictions.shape)

    cf_matrix = confusion_matrix(labels, predictions, labels=label_names)
    class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
    accuracy = np.mean((labels == predictions).astype(np.float)) * 100.
    balanced_acc = class_accuracy.mean()
    if len(groups) == 0:
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy])
    else:
        group_acc = np.array([get_per_group_acc(value, predictions, labels, groups) for value in np.unique(groups)])
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy]), np.array([round(g,2) for g in group_acc])

def get_per_group_acc(value, predictions, labels, groups):
    indices = np.array(np.where(groups == value))
    return np.mean((labels[indices] == predictions[indices]).astype(np.float)) * 100

def get_resnet_features(model, dataset, device='cuda'):
    """
    Gets the features of pretrained resnet model
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=2)

    features = []
    def hook(model, input, output):
        features.append(input[0].detach())
        return hook

    h = model.fc.register_forward_hook(hook)

    model.eval()
    all_features, all_labels = [], []
    all_groups, all_idxs = [], []
    
    with torch.no_grad():
        for img, label, group, idx in tqdm(loader):
            out = model(img.to(device))
            all_labels.append(label)
            all_groups.append(group)
            all_idxs.append(idx)
    all_features = features
    h.remove()
    print(torch.cat(all_features).cpu().numpy().shape)

    return torch.cat(all_features).cpu(), torch.cat(all_labels).cpu(), torch.cat(all_groups).cpu(), torch.cat(all_idxs).cpu()


def get_run_name(args):
    """
    Creates a run name based on the arguments
    """
    run = f"{args.exp.run}-debug" if args.exp.debug else args.exp.run
    if args.noise.method != "noop":
        run = f"{run}-bs{args.data.batch_size}-lr{args.hps.lr}-wd{args.hps.weight_decay}-epochs{args.exp.num_epochs}-seed{args.seed}"
    else:
        run = f"{run}-bs{args.data.batch_size}-{args.noise.method}-{args.noise.p}-lr{args.hps.lr}-wd{args.hps.weight_decay}-epochs{args.exp.num_epochs}-seed{args.seed}"
    
    if args.profile.method != "profile":
        run = f"{run}-{args.profile.method}"
        
    if args.model.save_emb:
        run = f"{run}-lp"

    if args.exp.oracle:
        run = f"{run}-oracle"

    return run

def compareModelWeights(model_a, model_b):
    module_a = model_a._modules
    module_b = model_b._modules
    if len(list(module_a.keys())) != len(list(module_b.keys())):
        return False
    a_modules_names = list(module_a.keys())
    b_modules_names = list(module_b.keys())
    for i in range(len(a_modules_names)):
        layer_name_a = a_modules_names[i]
        layer_name_b = b_modules_names[i]
        if layer_name_a != layer_name_b:
            return False
        layer_a = module_a[layer_name_a]
        layer_b = module_b[layer_name_b]
        if (
            (type(layer_a) == nn.Module) or (type(layer_b) == nn.Module) or
            (type(layer_a) == nn.Sequential) or (type(layer_b) == nn.Sequential)
            ):
            if not compareModelWeights(layer_a, layer_b):
                return False
        if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
            if not torch.equal(layer_a.weight.data, layer_b.weight.data):
                return False
    return True