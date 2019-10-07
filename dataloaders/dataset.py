# -*- coding: utf-8 -*-
import torch.utils.data as data
import os.path
import pandas as pd

from custom_symbols import SYMBOLS

class PPMIDataset(data.Dataset):
    def __init__(self, args, data):
        super().__init__()
        
        # Create patient ids array
        self.pat_ids = data[SYMBOLS.PAT_COL].unique()
        # complete data
        self.data = data
        
        # command args
        self.args = args
        
    def __getitem__(self, index):
        # get patient id for the index
        pat_id = self.pat_ids[index]
        
        # get all visits for the patient
        all_visits = self.data.loc[SYMBOLS.PAT_COL == pat_id]
        all_visits.drop(columns=[SYMBOLS.PAT_COL, SYMBOLS.EVENT_COL], inplace=True)

        # take the seq len of target and type of target(like UPDRS3, AMBUL score etc..)
        pred_seq_len = args.pred_seq_len
        pred_type = args.pred_type
        total_visits = all_vists.shape[0]
        
        # Do some data validations
        if total_visits != SYMBOLS.TOTAL_VISITS:
            raise ValueError

        if pred_seq_len <= all_vists.shape[0] and pred_type in all_visits.columns:
            raise ValueError
        
        all_visits = all_visits.to_numpy() # convert to numpy array
        
        input_seq = all_visits[:]
        
        # Take prediction (output) score for the sequence
        pred_seq = 
        
    def __len__(self):
        return len(self.pat_ids)
        