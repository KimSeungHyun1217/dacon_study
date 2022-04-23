import argparse
import glob,time
import json
import multiprocessing
import os
import random,math
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm
import os.path as osp

from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import DaconDataset,AlbumAugmentation
from loss import create_criterion
from model import PretrainedModel

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def train(data_dir,model_dir,args):
    torch.cuda.empty_cache()

    save_dir = increment_path(os.path.join(model_dir, args.name))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    dataset_module = getattr(import_module("dataset"), args.dataset)  
    train_dataset = dataset_module(
        path=data_dir,
    )
    num_class = train_dataset.get_num_class()
    print(num_class)
    
    train_dataset.set_transform(AlbumAugmentation)

    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        num_workers = multiprocessing.cpu_count() //2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last =True
    )

    model = PretrainedModel('vit_l_16',num_class)()
    model = model.to(device)
    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    num_batches = math.ceil(len(train_dataset) / args.batch_size)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model.train()
    for epoch in range(args.epochs):
        matches = 0
        with tqdm(total=num_batches) as pbar:
            epoch_loss, epoch_start = 0, time.time()
            
            for idx , train_batch in enumerate(train_loader):
                image,label = train_batch
                image = image.to(device)
                label = label.to(device)

                outs = model(image)
                outs = outs.to(device)

                preds = torch.argmax(outs, dim=-1)
                matches += (preds == label).sum().item()

                loss = criterion(outs,label)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'loss' : loss_val, 'accuracy' : matches
                }
                pbar.set_postfix(val_dict)
        scheduler.step()

        print(f"epoch : {epoch} Mean loss : {epoch_loss/num_batches}")

        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        ckpt_fpath = osp.join(model_dir, 'latest.pth')
        torch.save(model.state_dict(), ckpt_fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='DaconDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/open/train_df.csv'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)