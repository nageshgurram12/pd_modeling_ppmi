# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os.path
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from mypath import Path
from dataloaders.preprocessing import preprocess_data
from dataloaders.dataset import PPMIDataset
from custom_symbols import SYMBOLS

def prepare_dataloaders(args):        
    final_file = Path.get_path('merged_and_padded')
    if not os.path.exists(final_file):
        preprocess_data(args.pred_type, pad_missing_visits=True)
        
    data = pd.read_csv(final_file)
    # Create patient ids array
    pat_ids = data[SYMBOLS.PAT_COL].unique()
    # Creating data indices for training and validation splits:
    dataset_size = len(pat_ids)
    indices = list(range(dataset_size))
    val_split = args.val_split
    split = int(np.floor(val_split * dataset_size))
    if args.shuffle_data :
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    dataset = PPMIDataset(args, data=data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                           sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=valid_sampler)
    return (train_loader, val_loader)