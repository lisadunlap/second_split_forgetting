import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import copy
import pandas as pd

import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data

import omegaconf
from omegaconf import OmegaConf
import argparse
import wandb
import time
import copy

from utils import read_unknowns, nest_dict, flatten_config, get_resnet_features
from datasets.get_dataset import get_dataset
from datasets.base import BaseDataset, EmbeddingDataset, get_sampler
from utils import evaluate, get_run_name, compareModelWeights
import removal_methods
import methods.profiling as profiling
# from main import save_checkpoint, load_checkpoint, save_predictions

parser = argparse.ArgumentParser(description='Train/Val')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
# flags = parser.parse_args()
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not args.exp.wandb:
    os.environ['WANDB_SILENT']="true"

run = get_run_name(args)
wandb.init(entity='lisadunlap', project=args.proj, name=run, config=flatten_config(args))
print(f"DATASET {args.data.dataset}")

train_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='train'), args)
val_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='val'), args, clean=args.noise.clean_val)
test_set = BaseDataset(get_dataset(args.data.test_dataset, cfg=args, split='test'), args, clean=True)
weights = train_set.class_weights

labels = np.array(train_set.labels) == np.array(train_set.clean_labels)
p = args.noise.p if args.noise.method != 'noop' else 0
print(f"Training {run} on {args.data.dataset} with {p*100}% {args.noise.method} noise ({len(labels[labels == False])}/{len(labels)})")

if args.exp.oracle:
    assert hasattr(train_set.dataset, 'get_upweight_samples') or hasattr(train_set.dataset.datasets[1], 'get_upweight_samples'), f"Oracle doesn't work with dataset {args.data.dataset}"
    if args.data.dataset in ['ExpandedImagenette', 'WaterbirdsDiffusion']:
        idxs, groups = train_set.dataset.datasets[1].get_upweight_samples()
    else:
        idxs, groups = train_set.dataset.get_upweight_samples()
    removed_idxs, upweight_idxs = train_set.noisy_idxs, idxs
    print(f"Removing {len(removed_idxs)} samples and upweight {len(upweight_idxs)}")
else:
    removed_idxs, upweight_idxs = [], []

def get_model(args):
    # Load model
    pretrained_weights = args.model.weights if args.model.ft and args.model.weights != 'None' else None
    model = getattr(models, args.model.arch)(weights=pretrained_weights).cuda()
    num_classes, dim = model.fc.weight.shape
    model.fc = nn.Linear(dim, len(train_set.classes)).cuda()
    for name, param in model.named_parameters():
        if "fc" in name or not args.model.ft:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # model = nn.DataParallel(model).cuda()
    if args.model.save_emb:
        # change model to linear layer only
        return model, torchvision.ops.MLP(dim, [dim // 2, len(train_set.classes)]).cuda()
    return model, model

base_model, model = get_model(args)

## BETA: save emb and put lin layer on top
if args.model.save_emb:
    if args.model.load_emb: # load saved embeddings
        print("Loading embeddings...")
        save_dict = torch.load(f"{args.exp.save_dir}/{args.data.dataset}/emb_{args.model.arch}_{args.noise.method}_{args.noise.p}.pt")
        assert save_dict['seed'] == args.seed, f"Seed {args.seed} doesn't match saved seed {save_dict['seed']}"
    else: # compute embeddings
        print("Computing embeddings...")
        # assert args.noise.method == 'noop', "Can't save embeddings with noise"
        train_features, train_labels, train_groups, train_idxs = get_resnet_features(base_model, train_set)
        val_features, val_labels, val_groups, val_idxs = get_resnet_features(base_model, val_set)
        test_features, test_labels, test_groups, test_idxs = get_resnet_features(base_model, test_set)
        save_dict = {'train': {'features': train_features, 'labels': train_labels, 'groups': train_groups, 'idxs': train_idxs},
                    'val': {'features': val_features, 'labels': val_labels, 'groups': val_groups, 'idxs': val_idxs},
                    'test': {'features': test_features, 'labels': test_labels, 'groups': test_groups, 'idxs': test_idxs},
                    'seed': args.seed}
        if not os.path.exists(f"{args.exp.save_dir}/{args.data.dataset}"):
            os.makedirs(f"{args.exp.save_dir}/{args.data.dataset}")
        torch.save(save_dict, f"{args.exp.save_dir}/{args.data.dataset}/emb_{args.model.arch}_{args.noise.method}_{args.noise.p}.pt")
    train_set = EmbeddingDataset(train_set, save_dict['train']['features'], save_dict['train']['labels'], save_dict['train']['groups'], save_dict['train']['idxs'])
    val_set = EmbeddingDataset(val_set, save_dict['val']['features'], save_dict['val']['labels'], save_dict['val']['groups'], save_dict['val']['idxs'])
    test_set = EmbeddingDataset(test_set, save_dict['test']['features'], save_dict['test']['labels'], save_dict['test']['groups'], save_dict['test']['idxs'])
    print("...done!")
    
model = nn.DataParallel(model).cuda()

# Load dataset
first_split_trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=args.data.batch_size, num_workers=2)

trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.data.batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(
        val_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)

if args.exp.class_weights:
    weights = train_set.class_weights.cuda()
    print("class weights: ", weights)
    class_criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
else:
    class_criterion = nn.CrossEntropyLoss(reduction='none')
m = nn.Softmax(dim=1)
def get_optimizer(args, model):
    if args.hps.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.hps.lr, weight_decay=args.hps.weight_decay)
    elif args.hps.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.hps.lr, weight_decay=args.hps.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {args.hps.optimizer}")
    return optimizer
optimizer = get_optimizer(args, model)

results_df = pd.DataFrame(columns=['image_id', 'epoch', 'label', 'prediction', 'loss', 'conf', 'group', 'phase'])

def train_val_loop(args, model, optimizer,
    loader, weights, epoch, best_acc=0, phase="train", stage='first-split'):
    """
    One epoch of train-val loop given loss weights for each sample.
    Returns of dict of metrics to log
    """
    total_loss, cls_correct, total = 0,0,0
    if phase == "train":
        model.train()
    else:
        model.eval()
    total_loss, cls_correct, total = 0, 0, 0
    cls_true, cls_pred, cls_groups, cls_conf, cls_losses, idxs = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    with torch.set_grad_enabled(phase == 'train'):
        with tqdm(total=len(loader), desc=f'{phase} epoch {epoch}') as pbar:
            for i, (inp, cls_target, cls_group, idx) in enumerate(loader):
                inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
                if phase == "train":
                    optimizer.zero_grad()
                out = model(inp)
                conf, pred = torch.max(m(out), dim=-1)
                cls_loss = class_criterion(out.float(), cls_target)
                if phase == 'train':
                    cls_loss *= torch.Tensor(weights[idx]).cuda()
                loss = cls_loss.mean()
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total += cls_target.size(0)
                cls_correct += pred.eq(cls_target).sum().item()

                cls_true = np.append(cls_true, cls_target.cpu().numpy())
                cls_pred = np.append(cls_pred, pred.cpu().numpy())
                cls_groups = np.append(cls_groups, cls_group.cpu().numpy())
                cls_losses = np.append(cls_losses, cls_loss.detach().cpu().numpy())
                cls_conf = np.append(cls_conf, conf.detach().cpu().numpy())
                idxs = np.append(idxs, idx.cpu().numpy())
                pbar.update(i)
            
    accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)
    metrics = {f"{phase} loss": total_loss, f"{phase} acc": accuracy, f"{phase} balanced class acc": balanced_acc, 
                f"{phase} class acc": class_accuracy, f"{phase} group acc": group_accuracy, "epoch": epoch}
    if phase == 'val' and balanced_acc > best_acc:
        best_acc = balanced_acc
        wandb.summary["best val balanced acc"] = best_acc
        wandb.summary["best val epoch"] = epoch
        wandb.summary["best val group acc"] = group_accuracy
        wandb.summary["best val class acc"] = class_accuracy
        wandb.summary["best val acc"] = accuracy
        wandb.summary["best worst group acc"] = min(group_accuracy)
    if epoch % args.model.save_every == 0:
    # save predictions 
        save_predictions(idxs, cls_pred, cls_true, cls_groups, cls_losses, cls_conf, epoch, phase)
    return metrics, best_acc

def save_predictions(idxs, preds, labels, groups, losses, confs, epoch, phase='val-split'):
    global results_df
    predictions = []
    for (i, p, l, g, loss, c) in zip(idxs, preds, labels, groups, losses, confs):
        predictions += [{
            "image_id": int(i),
            "epoch": epoch,
            "label": l,
            "prediction": p,
            "loss": loss,
            "conf": c,
            "group": g,
            "phase": phase
        }]
    predictions_dir = f'./predictions/{args.data.dataset}/{run}/{phase}'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    print(f'Saving predictions to {predictions_dir}/epoch_{epoch}.csv')
    df = pd.DataFrame(predictions)
    df.to_csv(f'{predictions_dir}/epoch_{epoch}.csv', index=False)
    results_df = pd.concat([results_df, df])

def profile_upweights(cfg, weights, model, optimizer, train_loader, val_loader, epoch, samples_to_upweight, best_acc=0):
    """
    Run profiler
    """
    profiler = getattr(profiling, cfg.profile.method)(cfg, train_loader, val_loader)
    np.random.seed(epoch)
    # sample_ids = train_loader.dataset.idxs
    # samples_to_upweight = np.random.choice(sample_ids, size=int(len(sample_ids) * args.data.upweight_fraction), replace=False)
    # samples_to_upweight = sample_ids
    model.eval()
    torch.save(model.module.state_dict(), 'cur_model_weights.pth')
    train_metrics, _ = train_val_loop(cfg, model, optimizer, train_loader, weights, epoch=epoch, phase='train')
    val_metrics, best_acc = train_val_loop(cfg, model, optimizer, val_loader, weights, epoch=epoch, best_acc=best_acc, phase='val')
    diffs = np.zeros(len(train_loader.dataset.idxs))
    for idx in samples_to_upweight:
        new_weights = profiler.get_new_weights(weights, idx)
        print(f"## weight diff {np.sum(weights - new_weights)}")
        _, new_model = get_model(args)
        new_model.load_state_dict(torch.load('cur_model_weights.pth'))
        new_model = nn.DataParallel(new_model).cuda()
        new_optimizer = get_optimizer(args, new_model)
        new_train_metrics, _ = train_val_loop(cfg, new_model, new_optimizer, train_loader, new_weights, epoch=epoch, phase='train', stage='profile-upweights')
        new_val_metrics, best_acc = train_val_loop(cfg, new_model, new_optimizer, val_loader, new_weights, epoch=epoch, best_acc=best_acc, phase='val', stage='profile-upweights')
        weights, diff, changed = profiler.profile_step(val_metrics, new_val_metrics, weights, new_weights, idx)
        print(diff)
        diffs[idx] = diff
        np.save(f"diffs-{epoch}.npy", diffs)
    return weights, model, train_metrics, val_metrics, best_acc

def get_weights(cfg, dataset, samples_to_remove=[], samples_to_upweight=[], split='train', weights=None):
    """
    Returns a sampler for the given dataset. (weighted, random, etc.)
    Upweight is dictated by cfg.data.upweight_factor
    """
    assert all([r != u for r, u in zip(samples_to_remove, samples_to_upweight)]), 'samples to remove and upweight must be disjoint'
    image_ids = [i for i in range(len(dataset))]
    if weights is None:
        weights = np.ones(len(image_ids))
    weights[samples_to_remove] = 0
    weights[samples_to_upweight] *= args.data.upweight_factor
    # normalize by class counts
    labels = [l for l in dataset.labels if weights[l] > 0]
    groups = [g for g in dataset.dataset.groups if weights[g] > 0]
    for i, (label, group) in enumerate(zip(labels, groups)):
        label_subset = [i for i,l in enumerate(labels) if l == label]
        weights[i] /= len(label_subset)
    weights /= np.max(weights)
    return weights

# the magic happens here
best_val_acc, best_test_acc, best_val_epoch = 0, 0, 0
stage = 'first-split' if args.data.train_first_split != 'all' else 'all'
num_epochs = args.exp.num_epochs if not args.exp.debug else 1
weighter, selector = getattr(profiling, args.profile.method)(args, trainloader, valloader), getattr(profiling, args.select.method)(args, trainloader, valloader)
weights = get_weights(cfg, trainloader.dataset, samples_to_remove=removed_idxs, samples_to_upweight=upweight_idxs) # get inital weights
profile_steps = [args.hps.warmup] if (args.hps.profile_freq == 0 or args.profile.method == 'Profiler') else list(range(args.hps.warmup, num_epochs, args.hps.profile_freq))
print("profile steps: ", profile_steps)
if args.exp.class_weights:
    weights = np.ones(len(trainloader.dataset))
predictions_dir = f'./predictions/{args.data.dataset}/{run}'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)
np.save(f'{predictions_dir}/weights_0.npy', weights)
wandb.save(f'{predictions_dir}/weights_0.npy')
if args.profile.method == 'LeaveOneOut':
    print("entering new loop")
    for epoch in range(num_epochs):
        if epoch in profile_steps:
            print(f"******************* profiling step {epoch} *******************")
            samples = selector.get_interesting_samples(results_df[results_df['phase'] == 'train'])
            # profile_upweights(cfg, weights, model, optimizer, train_loader, val_loader, split='train', num_iterations=10)
            new_weights, model, train_metrics, val_metrics, best_val_acc = profile_upweights(args, weights, model, optimizer, trainloader, valloader, epoch, samples, best_acc=best_val_acc)
            # train and val metrics should be the same 
            # assert all([old_train_metrics[k] == train_metrics[k] for k in old_train_metrics.keys()]), 'train metrics should be the same'
            print(f"weight difference {np.sum(np.abs(new_weights - weights))}")
            weights = new_weights
            # save weights metrix
            np.save(f'{predictions_dir}/weights_{epoch}.npy', weights)
            wandb.save(f'{predictions_dir}/weights_{epoch}.npy')
        else:
            train_metrics, _ = train_val_loop(args, model, optimizer, trainloader, weights, epoch, phase="train", stage='profile-upweights')
            val_metrics, best_val_acc = train_val_loop(args, model, optimizer, valloader, weights, epoch, best_acc=best_val_acc, phase="val", stage='profile-upweights')
        wandb.log(train_metrics)
        wandb.log(val_metrics)
else:
    for epoch in range(num_epochs):
        if epoch in profile_steps:
            samples = selector.get_interesting_samples(results_df[results_df['phase'] == 'train'])
            print("samples to upweight: ", len(samples))
            new_weights = weighter.get_new_weights(weights, samples)
            print(f"weight difference {np.sum(np.abs(new_weights - weights))}")
            weights = new_weights
            # save weights metrix
            np.save(f'{predictions_dir}/weights_{epoch}.npy', weights)
            wandb.save(f'{predictions_dir}/weights_{epoch}.npy')
        train_metrics, _ = train_val_loop(args, model, optimizer, trainloader, weights, epoch, phase="train", stage='profile-upweights')
        val_metrics, best_val_acc = train_val_loop(args, model, optimizer, valloader, weights, epoch, best_acc=best_val_acc, phase="val", stage='profile-upweights')
        wandb.log(train_metrics)
        wandb.log(val_metrics)

test_metrics, best_test_acc = train_val_loop(args, model, optimizer, testloader, weights, epoch, best_acc=best_test_acc, phase="test")
wandb.summary['best test balanced acc'] = test_metrics['test balanced class acc']
wandb.summary['best test acc'] = test_metrics['test acc']
wandb.summary['best test group acc'] = test_metrics['test group acc']
results_df.to_csv(f'{predictions_dir}/results.csv')
wandb.save(f'{predictions_dir}/results.csv')