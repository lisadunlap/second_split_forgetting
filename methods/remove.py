import torch
import torch.utils.data as data
import numpy as np

def noop(cfg, labels, p=0.1):
    """
    Do nothing to the labels
    """
    return labels, []

def random(cfg, labels, p=0.1):
    """
    Randomly mislabel some of the labels with probability p
    """
    noisy_labels = list(labels).copy() #copy labels list, this is will be the new list with noisy labels
    num_labels = len(labels) #number of labels
    num_classes = len(set(labels)) #number of unique labels
    noisy_idxs = [] #this is the list of indices where the labels will be switched
    indices = np.random.permutation(num_labels) #randomly permute the indices
    for i, idx in enumerate(indices): 
        if i < p * num_labels: # only change the first pct_noise% of the permuted labels
            noisy_idxs.append(idx) #append to noisy_idxs
            before_label = noisy_labels[idx] 
            while noisy_labels[idx] == before_label: #ensure that the new label isn't the same
                new_label = np.random.choice(num_classes) #randomly select a new label
                noisy_labels[idx] = new_label  #assign new label
    return noisy_labels, noisy_idxs

def confusion_matrix(cfg, labels, p=0.1):
    """
    Mislabels in proportion to the confusion matrix of the validation set.
    Goal is to mislabel classes that would be commonly mislabled.
    """
    assert cfg.noise.confusion_matrix is not False, "Must provide a confusion matrix"
    confusion_matrix = np.load(cfg.noise.confusion_matrix)
    # remove the diagonal of the confusion matrix
    confusion_matrix = confusion_matrix - np.diag(np.diag(confusion_matrix))
    # normalize the confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1).reshape(-1,1)
    noisy_labels = list(labels).copy() #copy labels list, this is will be the new list with noisy labels
    num_labels = len(labels) #number of labels
    num_classes = len(set(labels)) #number of unique labels
    noisy_idxs = [] #this is the list of indices where the labels will be switched
    indices = np.random.permutation(num_labels) #randomly permute the indices
    for i, idx in enumerate(indices): 
        if i < p * num_labels: # only change the first pct_noise% of the permuted labels
            noisy_idxs.append(idx) #append to noisy_idxs
            before_label = noisy_labels[idx] 
            while noisy_labels[idx] == before_label: #ensure that the new label isn't the same
                new_label = np.random.choice(num_classes, p=confusion_matrix[before_label]) #randomly select a new label
                noisy_labels[idx] = new_label  #assign new label
    return noisy_labels, noisy_idxs
