# -*- coding: utf-8 -*-
import numpy as np

import argparse
import torch

from modeling.encoder_decoder import Model
from dataloaders import prepare_dataloaders
from custom_symbols import SYMBOLS

class Trainer(object):
    def __init__(self, args):
        super().__init__()
        
        (self.train_loader, self.val_loader, self.test_loader) = \
        prepare_dataloaders(args)
        
        print("Dataloaders are ready")
        ###----- Begin hyper parameter space ------ ##
        # set hidden size, each visit feature size and number of prediction scores
        args.hidden_size = 256
        
        ###----- End hyper parameter space ------ ##
        
        self.args = args
        
        # get model
        self.model = Model(args)
        
        # define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        # define loss - as these are scores define MSE 
        self.criterion = torch.nn.MSELoss()

    def train(self):
        # set model to train mode
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        
        loss_val = 0
        
        # each sample is  of shape [batch, seq, feature]
        for (input_seq, target_seq) in self.train_loader:
            
            # clear gradients
            self.model.zero_grad()
            
            # feed forward the inputs to model and target
            pred_seq = self.model(input_seq, target_seq)
            
            # If target is filled with missing value, then ignore them while 
            # calculating loss
            missing_index = np.where(target_seq == SYMBOLS.FILL_NA)
            pred_seq[missing_index] = 0
            target_seq[missing_index] = 0
            
            # calculate loss
            loss = self.criterion(pred_seq, target_seq)
            
            # calculate gradients
            loss.backward()
            
            # update model parameters
            self.optimizer.step()
            
            loss_val += loss.item()
            
        # return total loss per seq
        return loss_val / len(self.train_loader)
            
    def validate_test(self, data_loader, type='val'):
        # set model to eval mode
        self.model.eval()
        
        loss_val = 0
        total_samples = len(data_loader)
        
        # turn off gradient accumulation
        with torch.no_grad():
            
            # turn off teacher force
            self.model.teacher_force_ratio = 0
            iter = 0
            
            for (input_seq, target_seq) in data_loader:
                # feed forward the inputs to model and target
                pred_seq = self.model(input_seq, target_seq)
                
                if iter % 10 == 0 and type == 'test':
                    self.print_results(input_seq, target_seq, pred_seq)
                
                # If target is filled with missing value, then ignore them while 
                # calculating loss
                missing_index = np.where(target_seq == SYMBOLS.FILL_NA)
                pred_seq[missing_index] = 0
                target_seq[missing_index] = 0
            
                # calculate loss
                loss = self.criterion(pred_seq, target_seq)
                
                loss_val += loss.item()
                iter += 1
        
        return loss_val / len(self.val_loader)
    
    def print_results(self, input_seq, target_seq, pred_seq):
        pat_no = input_seq[:, 0, 0]
        
        
    
def main():
    parser = argparse.ArgumentParser(description="PD Modeling")
    
    parser.add_argument('--val-split', type=float, default=0.2,
                        help="Validation split (num patients ratio)")
    
    parser.add_argument('--test-split', type=float, default=0.1,
                        help="Test split")
    
    parser.add_argument('--shuffle-data', type=bool, default=True)
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    
    parser.add_argument('--pred-seq-len', type=int, default=5,
                        help='Prediction sequence length for test')
    
    parser.add_argument('--pred-types', type=str, default=SYMBOLS.TOTAL_UPDRS3,
                        choices=[SYMBOLS.TOTAL_UPDRS3, SYMBOLS.AMBUL_SCORE],
                        help='Specify outcomes to predict (space seperated)')
    
    parser.add_argument('--teacher-force-ratio', type=float, default=0.5,
                        help='Ratio of using original outcomes in decoder')
    
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Ratio of using original outcomes in decoder')
    
    parser.add_argument('--results-file', type=str,
                        default='./results/test_results.csv',
                        help="Write val/test results to file")
    
    parser.add_argument('--num-epochs', type=int, default=1,
                        help="Number of epochs")
    
    args = parser.parse_args()
    
    # Convert outcomes variables into list
    args.pred_types = args.pred_types.split(" ")
    torch.manual_seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(args)
    
    best_val_loss = float('inf')
    for epoch in range(0, args.num_epochs):
       train_loss = trainer.train() 
       print("\n epoch : %d, train loss: %f" % (epoch, train_loss))
       
       valid_loss = trainer.validate_test(trainer.val_loader, type='val')
       print("valid loss: %f" % (valid_loss))
       if valid_loss < best_val_loss:
           best_val_loss = valid_loss
           print("best val loss: %f, epoch : %d" % \
                 (best_val_loss, epoch))

    test_loss = trainer.validate_test(trainer.test_loader, type='test')
     
    print("test loss: %f" % (test_loss))
    
if __name__ == "__main__":
    main()