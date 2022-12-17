import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import copy
import pandas as pd

from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data

import omegaconf
from omegaconf import OmegaConf
import argparse
import wandb

from utils import read_unknowns, nest_dict, flatten_config
from datasets.get_dataset import get_dataset
from datasets.base import BaseDataset, get_sampler
from utils import evaluate
import removal_methods

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

run = f"{args.exp.run}-debug" if args.exp.debug else args.exp.run
if args.exp.debug:
    run = 'debug'
wandb.init(entity='lisadunlap', project='second_split', name=run, config=flatten_config(args))
print(f"DATASET {args.data.dataset}")

train_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='train'), args)
val_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='val'), args, clean=args.noise.clean_val)
test_set = BaseDataset(get_dataset(args.data.test_dataset, cfg=args, split='test'), args, clean=True)
weights = train_set.class_weights

labels = np.array(train_set.labels) == np.array(train_set.clean_labels)
p = args.noise.p if args.noise.method != 'noop' else 0
print(f"Training {args.exp.run} on {args.data.dataset} with {p*100}% {args.noise.method} noise ({len(labels[labels == False])}/{len(labels)})")

# Load model
pretrained_weights = 'IMAGENET1K_V1' if args.model.ft else None
model = getattr(models, args.model.arch)(weights=pretrained_weights).cuda()
num_classes, dim = model.fc.weight.shape
model.fc = nn.Linear(dim, len(train_set.classes)).cuda()
for name, param in model.named_parameters():
    if "fc" in name or not args.model.ft:
        param.requires_grad = True
    else:
        param.requires_grad = False
model = nn.DataParallel(model).cuda()



# Remove samples from the train set
if args.data.remove:
    assert os.path.exists(args.data.results_dir), f"Results directory {args.data.results_dir} does not exist"
    df = pd.concat([pd.read_csv(f) for f in os.listdir(args.data.results_dir)])
    num_samples = args.data.num_samples_to_remove if args.data.num_samples_to_remove > 0 else int(args.data.num_samples_to_remove * len(train_set))
    removed_idxs, upweight_idxs = getattr(removal_methods, args.data.removal_method)(args, df, num_samples)
    print(f"Removing {len(removed_idxs)} samples from first split via {args.data.removal_method}")
    print(f"Upweighting {len(upweight_idxs)} samples from second split via {args.data.removal_method}")
if args.exp.oracle:
    assert args.data.dataset == 'ExpandedImagenette', "Oracle only works with ExpandedImagenette"
    idxs, groups = train_set.dataset.datasets[1].get_upweight_samples()
    removed_idxs, upweight_idxs = train_set.noisy_idxs, idxs
else:
    removed_idxs, upweight_idxs = [], []

# get sampler (for removing/upweighting samples)
sampler = get_sampler(args, train_set, removed_idxs, upweight_idxs)


# Load dataset
assert args.data.train_first_split in ['odd', 'even', 'all'], "train_first_split must be 'odd' or 'even'"
if args.data.train_first_split == 'all':
    first_split_trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.data.batch_size, num_workers=2, sampler=sampler)
else:
    first_split_trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=args.data.batch_size, num_workers=2, sampler=sampler)
    second_split_trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=args.data.batch_size, num_workers=2, sampler=sampler)

trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)
valloader = torch.utils.data.DataLoader(
        val_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)

class_criterion = nn.CrossEntropyLoss()
class_criterion_per = nn.CrossEntropyLoss(reduction='none')
m = nn.Softmax(dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=args.hps.lr, weight_decay=args.hps.weight_decay, momentum=0.9)
# iters_per_epoch = len(first_split_trainloader)+1
# T_max = args.exp.num_epochs*iters_per_epoch
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False)

def train_val_loop(loader, epoch, phase="train", best_acc=0, stage='first-split'):
    """
    One epoch of train-val loop.
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
                loss = cls_loss
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                else:
                    cls_loss = class_criterion_per(out.float(), cls_target)

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
    print("group accuracy", group_accuracy)

    wandb.log({f"{stage} {phase} loss": total_loss, f"{stage} {phase} cls acc": accuracy, f"{stage} {phase} balanced class acc": balanced_acc, 
                f"{stage} {phase} class acc": class_accuracy, f"{stage} {phase} group acc": group_accuracy, "epoch": epoch})

    if phase == 'val' and balanced_acc > best_acc:
        print(stage, balanced_acc, best_acc)
        best_acc = balanced_acc
        save_checkpoint(model, balanced_acc, epoch, stage)
        wandb.summary[f'{stage} best {phase} acc'] = accuracy
        wandb.summary[f'{stage} best {phase} balanced acc'] = balanced_acc
        wandb.summary[f'{stage} best {phase} group acc'] = group_accuracy
        wandb.summary[f'{stage} best {phase} epoch'] = epoch
    elif phase == 'test':
        wandb.summary[f'{stage} {phase} acc'] = accuracy
        wandb.summary[f'{stage} {phase} balanced acc'] = balanced_acc
        wandb.summary[f'{stage} {phase} group acc'] = group_accuracy
    if phase != 'train' and epoch % args.model.save_every == 0:
    # save predictions 
        save_predictions(idxs, cls_pred, cls_true, cls_groups, cls_losses, cls_conf, epoch, stage, phase)
    return best_acc if phase == 'val' else balanced_acc

def save_predictions(idxs, preds, labels, groups, losses, confs, epoch, stage='first-split', phase='val-split'):
    predictions = []
    for (i, p, l, g, loss, c) in zip(idxs, preds, labels, groups, losses, confs):
        predictions += [{
            "image_id": int(i),
            "stage": stage,
            "epoch": epoch,
            "label": l,
            "prediction": p,
            "loss": loss,
            "conf": c,
            "group": g
        }]
    predictions_dir = f'./predictions/{args.data.dataset}/{args.exp.run}/{stage}/{phase}'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    print(f'Saving predictions to {predictions_dir}/predictions-{args.data.train_first_split}-epoch_{epoch}.csv')
    pd.DataFrame(predictions).to_csv(f'{predictions_dir}/predictions-{args.data.train_first_split}-epoch_{epoch}.csv', index=False)
    wandb.save(f'{predictions_dir}/predictions-{args.data.train_first_split}-epoch_{epoch}.csv')

def save_checkpoint(model, acc, epoch, stage='base'):
    state = {
        "acc": acc,
        "epoch": epoch,
        "net": model.module.state_dict()
    }
    print("SAVING CHECKPOINT")
    checkpoint_dir = f'./checkpoint/{args.data.dataset}/{args.exp.run}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(f'Saving checkpoint with acc {acc} to {checkpoint_dir}/{stage}-{args.data.train_first_split}-model_best.pth')
    torch.save(state, f'{checkpoint_dir}/{stage}-{args.data.train_first_split}-model_best.pth')
    wandb.save(f'{checkpoint_dir}/{stage}-{args.data.train_first_split}-model_best.pth')

def load_checkpoint(model, stage='base'):
    path = f'./checkpoint/{args.data.dataset}/{args.exp.run}/{stage}-{args.data.train_first_split}-model_best.pth'
    checkpoint = torch.load(path)
    model.module.load_state_dict(checkpoint['net'])
    print(f"...loaded checkpoint with acc {checkpoint['acc']}")

if args.model.eval:
    print("------ EVAL ONLY ------")
    load_checkpoint(model, 'all')
    best_val_acc = train_val_loop(valloader, 0, phase="val", stage='best')
    best_test_acc = train_val_loop(testloader, 0, phase="test", stage='best')
    wandb.summary['best val acc'] = best_val_acc
    wandb.summary['test acc'] = best_test_acc
 
if not args.model.eval:
    # first split
    best_val_acc, best_test_acc, best_val_epoch = 0, 0, 0
    stage = 'first-split' if args.data.train_first_split != 'all' else 'all'
    num_epochs = args.exp.num_epochs if not args.exp.debug else 1
    for epoch in range(num_epochs):
        train_acc = train_val_loop(first_split_trainloader, epoch, phase="train", stage=stage)
        if epoch % args.model.save_every == 0:
            _ = train_val_loop(trainloader, epoch, phase="eval-train", best_acc=best_val_acc, stage=stage)
        best_val_acc = train_val_loop(valloader, epoch, phase="val", best_acc=best_val_acc, stage=stage)
        print(f"Epoch {epoch} val acc: {best_val_acc}")

    # load_checkpoint(model, 'first-split')
    train_val_loop(testloader, epoch, phase="test", stage=stage)

    if args.data.train_first_split != 'all' and not args.exp.train_first_split_only:
        # second split
        best_val_acc, best_test_acc, best_val_epoch = 0, 0, 0
        for epoch in range(num_epochs):
            train_acc = train_val_loop(second_split_trainloader, epoch, phase="train", stage='ft')
            if epoch % args.model.save_every == 0:
                _ = train_val_loop(first_split_trainloader, epoch, phase="eval-first-split", best_acc=best_val_acc, stage='second-split')
            best_val_acc = train_val_loop(valloader, epoch, phase="val", best_acc=best_val_acc, stage='second-split')
            print(f"Epoch {epoch} val acc: {best_val_acc}")

        # load_checkpoint(model, stage='second-split')
        train_val_loop(testloader, epoch, phase="test", stage='second-split')