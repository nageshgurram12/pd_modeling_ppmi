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
    # get the data we want to handle
    final_file = get_final_file_path(args)
    
    if not os.path.exists(final_file):
        preprocess_args = {"pred_types" : args.pred_types, 
                           SYMBOLS.MISSING_VAL_STRATEGY: args.missing_val_strategy}
        preprocess_data(**preprocess_args)
        
    data = pd.read_csv(final_file, dtype='float64')
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
    (test_indices, val_indices, train_indices) = \
    (indices[:test_split], indices[test_split:(val_split+test_split)], \
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
    
    # collate the data if missing NAs are removed
    if args.missing_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.REMOVE:
        common_args['collate_fn'] = pad_seq
        
    train_loader = torch.utils.data.DataLoader(dataset,sampler=train_sampler,
                                               **common_args)
    
    val_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler,
                                             **common_args)
    
    test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler,
                                              **common_args)
    
    return (train_loader, val_loader, test_loader)

'''
Get the file path based on missing val strategy and others
'''
def get_final_file_path(args):
    '''
    final_file = Path.get_path(SYMBOLS.DATA_TYPES.MERGED_PADDED)
    if args.missing_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.REMOVE:
        final_file = Path.get_path(SYMBOLS.DATA_TYPES.MERGED_NA_REMOVED)
    elif args.missing_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.PAD:
        final_file = Path.get_path(SYMBOLS.DATA_TYPES.MERGED_PADDED)
    elif args.missing_val_strategy == SYMBOLS.MISSING_VAL_STRATEGIES.MA:
        final_file = Path.get_path(SYMBOLS.DATA_TYPES.MERGED_MA)
    '''
    return Path.get_preprocessed(args.preprocessed_file_name)
    #return final_file

'''
This is to match the API for pad_seq 
default torch collate_fn won't return lengths
'''
def collate_fn(batch_data):
    src_seqs, trg_seqs = zip(*batch_data)
    
    def seq_len(sequences):
        lengths = [len(seq) for seq in sequences]
        return lengths
    
    return torch.FLoatTensor(src_seqs), seq_len(src_seqs), \
           torch.FloatTensor(trg_seqs), seq_len(trg_seqs)

'''
If sequences have variable length, then pad them with 0s
'''
def pad_seq(batch_data):

    # pad seqs with 0s    
    def pad_seqs(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), \
                                           max(lengths), len(sequences[0][0]))
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end, :] = seq[:end, :]
        return padded_seqs, lengths
    
    # sort by sequence length (descending order) to use pack_padded_sequence
    batch_data.sort(key=lambda x: len(x[0]), reverse=True)
    
    # seperate source and target sequences
    src_seqs = [sample[0] for sample in batch_data]
    trg_seqs = [sample[1] for sample in batch_data]
    
    src_seqs, src_lens = pad_seqs(src_seqs)
    trg_seqs, trg_lens = pad_seqs(trg_seqs)
    
    return src_seqs, src_lens, trg_seqs, trg_lens