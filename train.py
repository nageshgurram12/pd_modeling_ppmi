# -*- coding: utf-8 -*-
import numpy as np

import argparse
import torch

from modeling.encoder_decoder import Model
from dataloaders import prepare_dataloaders
from custom_symbols import SYMBOLS

def train(args):    
    (train_loader, val_loader) = prepare_dataloaders(args)

    # set hidden size, each visit feature size and number of prediction scores
    args.hidden_size = 256
    model = Model(args)
        
    for (input_seq, pred_seq) in train_loader:
        print(input_seq.shape, pred_seq.shape)
        
        pred_seq = pred_seq.unsqueeze(2)
        model.train(input_seq, pred_seq)
    
    
def main():
    parser = argparse.ArgumentParser(description="PD Modeling")
    
    parser.add_argument('--val-split', type=int, default=0.2,
                        help="Validation split")
    
    parser.add_argument('--shuffle-data', type=bool, default=True)
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    
    parser.add_argument('--pred-seq-len', type=int, default=5,
                        help='Prediction sequence length')
    
    parser.add_argument('--pred-type', type=str, default=SYMBOLS.TOTAL_UPDRS3,
                        choices=[SYMBOLS.TOTAL_UPDRS3, SYMBOLS.AMBUL_SCORE],
                        help='Prediction sequence length')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(args)
    
if __name__ == "__main__":
    main()