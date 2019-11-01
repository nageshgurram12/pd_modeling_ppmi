# -*- coding: utf-8 -*-
import torch.utils.data as data
import os.path
import pandas as pd
import torch

from custom_symbols import SYMBOLS

class PPMIDataset(data.Dataset):
    def __init__(self, args, data):
        super().__init__()
        
        # Create patient ids array
        self.pat_ids = data[SYMBOLS.PAT_COL].unique()
        # complete data
        self.data = data
        
        # set input size, output size in args
        args.input_size = (data.shape[1]) # patno, event id are removed
        args.output_size = len(args.pred_types)
            
        # command args
        self.args = args
            
    def __getitem__(self, index):
        # get patient id for the index
        pat_id = self.pat_ids[index]
        
        # get all visits for the patient
        all_visits = self.data.loc[self.data[SYMBOLS.PAT_COL] == pat_id]
        #all_visits = all_visits.drop(columns=[SYMBOLS.PAT_COL, SYMBOLS.EVENT_COL])

        # take the seq len of target and type of target(like UPDRS3, AMBUL score etc..)
        pred_seq_len = self.args.pred_seq_len
        pred_types = self.args.pred_types
        total_visits = all_visits.shape[0]
        
        # Do some data validations
        if self.args.missing_val_strategy != SYMBOLS.MISSING_VAL_STRATEGIES.REMOVE:
            if total_visits != SYMBOLS.TOTAL_VISITS:
                raise ValueError

        # if pred seq length has smaller then entire thorw error
        if not(pred_seq_len <= all_visits.shape[0]):
            raise ValueError
            
        # If any prediction var is missing in data, then throw error
        if not(all(pred_var in all_visits.columns for pred_var in pred_types)):
            raise ValueError
        
        # Take input sequence by removing pred sequence
        input_seq_len = total_visits-pred_seq_len
        
        #input_seq_filter = all_visits[SYMBOLS.EVENT_COL] <= input_seq_len
        input_seq = all_visits.iloc[:input_seq_len]
        input_seq = torch.tensor(input_seq.values, dtype=torch.float, \
                                 device= self.args.device)
        
        # Take prediction (output) score sequence of only target outcomes
        #pred_seq_filter = all_visits[SYMBOLS.EVENT_COL] > input_seq_len
        pred_seq = all_visits[pred_types].iloc[input_seq_len:]
        pred_seq = torch.tensor(pred_seq.values, dtype=torch.float, \
                                device= self.args.device)

        return input_seq, pred_seq
            
    def __len__(self):
        return len(self.pat_ids)
        