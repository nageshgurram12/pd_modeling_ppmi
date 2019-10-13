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
        preprocess_data(pred_types=args.pred_types, pad_missing_visits=True)
        
    data = pd.read_csv(final_file)
    # Create patient ids array
    pat_ids = data[SYMBOLS.PAT_COL].unique()
    
    # Creating data indices for training and validation splits
    dataset_size = len(pat_ids)
    indices = list(range(dataset_size))
    val_split = args.val_split
    test_split = args.test_split
    
    val_split = int(np.floor(val_split * dataset_size))
    test_split = int(np.floor(test_split * dataset_size))
    
    if args.shuffle_data :
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    test_indices, val_indices, train_indices = \
    (indices[:test_split], indices[:(val_split+test_split)], \
    indices[(val_split+test_split):])
    
    # sampler to pick train and val dataset
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    dataset = PPMIDataset(args, data=data)
    
    # create torch data loaders
    common_args = {'batch_size' : args.batch_size, 
                   'num_workers' : args.num_workers,
                   'drop_last' : True}
    train_loader = torch.utils.data.DataLoader(dataset,sampler=train_sampler,
                                               **common_args)
    
    val_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler,
                                             **common_args)
    
    test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler,
                                              **common_args)
    
    return (train_loader, val_loader, test_loader)