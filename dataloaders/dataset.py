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
        args.input_size = (data.shape[1]-2) # patno, event id are removed
        if type(args.pred_type) is str:
            args.output_size = 1
            
        # command args
        self.args = args
            
    def __getitem__(self, index):
        # get patient id for the index
        pat_id = self.pat_ids[index]
        
        # get all visits for the patient
        all_visits = self.data.loc[self.data[SYMBOLS.PAT_COL] == pat_id]
        all_visits.drop(columns=[SYMBOLS.PAT_COL, SYMBOLS.EVENT_COL], inplace=True)

        # take the seq len of target and type of target(like UPDRS3, AMBUL score etc..)
        pred_seq_len = self.args.pred_seq_len
        pred_type = self.args.pred_type
        total_visits = all_visits.shape[0]
        
        # Do some data validations
        if total_visits != SYMBOLS.TOTAL_VISITS:
            raise ValueError

        if not(pred_seq_len <= all_visits.shape[0] and pred_type in all_visits.columns):
            raise ValueError
        
        # Take input sequence by removing pred sequence from all visits
        input_seq = all_visits.iloc[:(total_visits-pred_seq_len), :].values
        input_seq = torch.tensor(input_seq, dtype=torch.float)
        
        # Take prediction (output) score sequence
        pred_seq = all_visits.iloc[(total_visits-pred_seq_len):, :][pred_type].values
        pred_seq = torch.tensor(pred_seq, dtype=torch.float)

        return input_seq, pred_seq
        
        
    def __len__(self):
        return len(self.pat_ids)
        