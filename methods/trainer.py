# import torch
# import torch.nn as nn
# import numpy as np
# import os
# from tqdm import tqdm
# import copy
# import pandas as pd

# import torchvision
# from torchvision.models import resnet50, ResNet50_Weights
# import torchvision.models as models
# import torchvision.transforms as transforms
# import torch.utils.data as data

# import omegaconf
# from omegaconf import OmegaConf
# import argparse
# import wandb
# import time
# import copy

# from utils import read_unknowns, nest_dict, flatten_config, get_resnet_features
# from datasets.get_dataset import get_dataset
# from datasets.base import BaseDataset, EmbeddingDataset, get_sampler
# from utils import evaluate, get_run_name

# def train_val_loop(args, model, optimizer,
#     loader, weights, epoch, best_acc=0, phase="train", stage='first-split'):
#     """
#     One epoch of train-val loop given loss weights for each sample.
#     Returns of dict of metrics to log
#     """
#     total_loss, cls_correct, total = 0,0,0
#     if phase == "train":
#         model.train()
#     else:
#         model.eval()
#     total_loss, cls_correct, total = 0, 0, 0
#     cls_true, cls_pred, cls_groups, cls_conf, cls_losses, idxs = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
#     with torch.set_grad_enabled(phase == 'train'):
#         with tqdm(total=len(loader), desc=f'{phase} epoch {epoch}') as pbar:
#             for i, (inp, cls_target, cls_group, idx) in enumerate(loader):
#                 inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
#                 if phase == "train":
#                     optimizer.zero_grad()
#                 out = model(inp)
#                 conf, pred = torch.max(m(out), dim=-1)
#                 cls_loss = class_criterion(out.float(), cls_target)
#                 if phase == 'train':
#                     cls_loss *= torch.Tensor(weights[idx]).cuda()
#                 loss = cls_loss.mean()
#                 if phase == "train":
#                     loss.backward()
#                     optimizer.step()

#                 total_loss += loss.item()
#                 total += cls_target.size(0)
#                 cls_correct += pred.eq(cls_target).sum().item()

#                 cls_true = np.append(cls_true, cls_target.cpu().numpy())
#                 cls_pred = np.append(cls_pred, pred.cpu().numpy())
#                 cls_groups = np.append(cls_groups, cls_group.cpu().numpy())
#                 cls_losses = np.append(cls_losses, cls_loss.detach().cpu().numpy())
#                 cls_conf = np.append(cls_conf, conf.detach().cpu().numpy())
#                 idxs = np.append(idxs, idx.cpu().numpy())
#                 pbar.update(i)
            
#     accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)
#     metrics = {f"{phase} loss": total_loss, f"{phase} acc": accuracy, f"{phase} balanced class acc": balanced_acc, 
#                 f"{phase} class acc": class_accuracy, f"{phase} group acc": group_accuracy, "epoch": epoch}
#     if phase == 'val' and balanced_acc > best_acc:
#         best_acc = balanced_acc
#         wandb.summary["best val balanced acc"] = best_acc
#         wandb.summary["best val epoch"] = epoch
#         wandb.summary["best val group acc"] = group_accuracy
#         wandb.summary["best val class acc"] = class_accuracy
#         wandb.summary["best val acc"] = accuracy
#     if epoch % args.model.save_every == 0:
#     # save predictions 
#         save_predictions(idxs, cls_pred, cls_true, cls_groups, cls_losses, cls_conf, epoch, phase)
#     return metrics, best_acc

# class Trainer:

#     """
#     Wrapper class for trainnig a model. (note, not he pytorch trainer)
#     """
#     def __init__(self, cfg, trainloader, valloader):
#         pass

#     def save_predictions(self, idxs, preds, labels, groups, losses, confs, epoch, phase='val-split'):
#         predictions = []
#         for (i, p, l, g, loss, c) in zip(idxs, preds, labels, groups, losses, confs):
#             predictions += [{
#                 "image_id": int(i),
#                 "epoch": epoch,
#                 "label": l,
#                 "prediction": p,
#                 "loss": loss,
#                 "conf": c,
#                 "group": g,
#                 "phase": phase
#             }]
#         predictions_dir = f'./predictions/{args.data.dataset}/{run}/{phase}'
#         if not os.path.exists(predictions_dir):
#             os.makedirs(predictions_dir)
#         print(f'Saving predictions to {predictions_dir}/epoch_{epoch}.csv')
#         df = pd.DataFrame(predictions)
#         df.to_csv(f'{predictions_dir}/epoch_{epoch}.csv', index=False)
#         results_df = pd.concat([results_df, df])
#         wandb.save(f'{predictions_dir}/epoch_{epoch}.csv')
