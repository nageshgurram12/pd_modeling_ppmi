# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
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
        #torch.autograd.set_detect_anomaly(True)
        
        loss_val = 0
        self.model.teacher_force_ratio = self.args.teacher_force_ratio
        
        # input, target of shape [batch, seq, feature]
        for samples in self.train_loader:
            if len(samples) == 4:
                (input_seq, ip_seq_lengths, target_seq, tg_seq_lengths) = samples
                input_seq = torch.nn.utils.rnn.pack_padded_sequence(input_seq, \
                                             ip_seq_lengths, batch_first=True)
            else:
                (input_seq, target_seq) = samples
                
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
            #loss.requires_grad(True)
            
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
        
        if type == 'test':
            results = self.create_results_df()
            
        # turn off gradient accumulation
        with torch.no_grad():
            
            # turn off teacher force
            self.model.teacher_force_ratio = 0
            iter = 0
            
        # each sample is  of shape [batch, seq, feature]
        for samples in data_loader:
            if len(samples) == 4:
                (input_seq, ip_seq_lengths, target_seq, tg_seq_lengths) = samples
                input_seq = torch.nn.utils.rnn.pack_padded_sequence(input_seq, \
                                             ip_seq_lengths, batch_first=True)
            else:
                (input_seq, target_seq) = samples
                
                # feed forward the inputs to model and target
                pred_seq = self.model(input_seq, target_seq)
                
                if iter % 10 == 0 and type == 'test':
                    self.store_results(results, input_seq, target_seq, pred_seq)
                
                # If target is filled with missing value, then ignore them while 
                # calculating loss
                missing_index = np.where(target_seq == SYMBOLS.FILL_NA)
                pred_seq[missing_index] = 0
                target_seq[missing_index] = 0
            
                # calculate loss
                loss = self.criterion(pred_seq, target_seq)
                
                loss_val += loss.item()
                iter += 1
        
        # write new results to the file
        if type == 'test':
            results.to_csv(self.args.results_file)
        
        return loss_val / len(self.val_loader)
    
    '''
    Create a results file to store predictions on test
    return a dataframe to write 
    '''
    def create_results_df(self):
        # create a dataframe to store the results into file
        columns = [SYMBOLS.PAT_COL, SYMBOLS.EVENT_COL]
        pred_types = self.args.pred_types
        pred_outputs = ["pred_" + type for type in pred_types]
        target_outputs = ["target_" + type for type in pred_types]
        columns.extend(target_outputs); columns.extend(pred_outputs)
        results = pd.DataFrame(columns=columns, index=None)
        
        return results
    
    '''
    Write random test pred to results dataframe
    '''
    def store_results(self, results, input_seq, target_seq, pred_seq):           
        # input_seq - (batch, seq, features)
        pat_ids = input_seq[:, 0, 0].numpy()
        visit_num = input_seq_len = input_seq.shape[1]
        target_seq = target_seq.detach().numpy(); 
        pred_seq = pred_seq.detach().numpy()
        
        for pat_ix in range(len(pat_ids)):
            pat_id = int(pat_ids[pat_ix])
            visit_num = input_seq_len
            # ix - (batch_id, seq_ix, output_ix)
            for seq_ix in range(target_seq.shape[1]):
                outs = target_seq[pat_ix, seq_ix, :] # targets for visit seq_ix
                if outs != SYMBOLS.FILL_NA:
                    preds = pred_seq[pat_ix, seq_ix, :]
                    visit_num += seq_ix
                    outs = np.append(outs, preds) # append preds and pat ids, events
                    outs = np.insert(outs, 0, [pat_id, visit_num])
                    results.loc[len(results)] = list(outs.astype(int))
        
    
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
    
    parser.add_argument('--pred-seq-len', type=int, default=4,
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
    
    parser.add_argument('--num-epochs', type=int, default=10,
                        help="Number of epochs")
    
    parser.add_argument('--missing-val-strategy', type=str,
                        default=SYMBOLS.MISSING_VAL_STRATEGIES.PAD,
                        help="Specify missing val removal strategy")
    
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
       
       '''
       valid_loss = trainer.validate_test(trainer.val_loader, type='val')
       print("valid loss: %f" % (valid_loss))
       if valid_loss < best_val_loss:
           best_val_loss = valid_loss
           print("best val loss: %f, epoch : %d" % \
                 (best_val_loss, epoch))

        '''
    test_loss = trainer.validate_test(trainer.test_loader, type='test')
     
    print("test loss: %f" % (test_loss))
    
if __name__ == "__main__":
    main()