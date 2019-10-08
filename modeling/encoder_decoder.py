# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

'''
Encoder returns context vector and hidden states at each timestep
after passing through model.
This takes entire sequence of visits and returns the output for entire seq
'''
class Encoder(nn.Module):
    '''
    input_size represents the feature vector of each visit by patient
    hidden_size represents the model produced hidden state size at time step (t)
    '''
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
    '''
    input_seqs - batch of multiple patient visits [pat_id, event_id, features]
    hidden - initial hidden state
    
    return:
        output_seqs - output state from each cell in gru
        hidden - default 0
    '''
    def forward(self, input_seqs):
        output_seqs, hidden = self.gru(input_seqs)
        return output_seqs, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
'''
Decoder predicts the target label at each step in output seq.
forward takes single visit feature vector and gives its only hidden state
'''
class Decoder(nn.Module):
    '''
    hidden_size - hidden state size at each time step t
    output_size - mostly will be 1, as we predict only one target score
    '''
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    '''
    prev_score - previous output prediction (0 for initial step)
    hidden - previous time step hidden state
    '''
    def forward(self, prev_score, hidden):
        prev_score = prev_score.unsqueeze(1)
        output, hidden = self.gru(prev_score, hidden)
        pred_score = self.linear(output.squeeze(1))
        
        return pred_score, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
class Model():
    def __init__(self, args):
        
        self.hidden_size = args.hidden_size
        if 'input_size' in args:
            self.input_size = args.input_size # input size at in_seq[t] 
        
        if 'output_size' in args:
            self.output_size = args.output_size # output size at out_seq[t]
        self.batch_size = args.batch_size

        # Instantiate decoder and encoder        
        self.encoder = Encoder(self.input_size, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.output_size)
        
    '''
    pats_visit_seqs - batch of sequence of visits 
    [batch, in_seq_len, features]
    prediction_scores - batch of sequence of prediction scores 
    [batch, pred_seq_len, score]
    '''
    def train(self, pats_visit_seqs, prediction_scores):
        pred_seq_len = prediction_scores.shape[1]
        
        #tensor to store decoder outputs
        outputs = torch.zeros(pred_seq_len, self.batch_size, self.output_size)
        
        encoder_outs, last_out = self.encoder(pats_visit_seqs)
        
        # For decoder first cell, prev pred score is 0
        prev_score = torch.zeros(self.batch_size, self.output_size)
        hidden = last_out # initial hidden state for decoder
        
        for t in range(0, pred_seq_len):
            output, hidden = self.decoder(prev_score, hidden)
            outputs[t] = prev_score = output
            
        return outputs