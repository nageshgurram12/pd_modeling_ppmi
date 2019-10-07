# -*- coding: utf-8 -*-
import numpy as np

import argparse
import torch

from dataloaders import prepare_dataloaders

def train(args):    
    prepare_dataloaders(args)
    pass
    
    
def main():
    parser = argparse.ArgumentParser(description="PD Modeling")
    
    parser.add_argument('--val-split', type=int, default=0.2,
                        help="Validation split")
    
    parser.add_argument('--shuffle-data', type=bool, default=True)
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    
    parser.add_argument('--pred-seq-len', type=int, default=5,
                        help='Prediction sequence length')
    
    parser.add_argument('--pred-type', type=str, default='UPDRS3',
                        choices=['UPDRS3', 'AMBUL_SCORE'],
                        help='Prediction sequence length')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    train(args)
    
if __name__ == "__main__":
    main()