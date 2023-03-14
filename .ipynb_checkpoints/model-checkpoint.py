import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import dataset
import numpy as np

from lib.positional_encoding import *

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class TransformerModel(torch.nn.Module):
    def __init__(self, d_input, d_model=64, nlayers=3, d_out=None, 
                 dropout=0.15, num_players=1000, max_action=1, phi=0.05, **kwargs):
        
        super().__init__()
        
        self.d_model = d_model
        self.player_embedding_dim = d_model
        self.nlayers = nlayers
        self.d_out = d_out
        self.max_action = max_action
        self.phi = phi
        
        if d_out is None:
            self.d_out = d_input
            
        self.positional_encoder = PositionalEncoding()
        
        self.p_embeddings = torch.nn.Embedding(num_players, d_model, scale_grad_by_freq=True, max_norm=None, sparse=False)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_mdel, nhead=nhead, batch_first=True, dim_feedforward=4*d_model)
        self.temporal = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_model, bias=True),
            )
        
        self.out = torch.nn.Sequential(
                torch.nn.Linear(d_model, self.d_out, bias=True),
                torch.nn.Tanh()
                )
        
        
    def forward(self, src, player_list, src_mask, noise=None 
                tr_mask=None, player_mask=None, **kwargs):
        player_ids = player_list + 2
        
        player_embeddings = self.poi_embeddings((player_list).flatten().long()).view(src.size(0), src.size(1), -1)
        
        if noise is not None:
            src[tr_mask==1] = noise

        src_enc = self.shot_encoder(src) * math.sqrt(self.d_model) 
                
        src_enc = src_enc + player_embeddings
        
        src_enc = self.positional_encoder(src_enc.transpose(0, 1)).transpose(0, 1)
        
        auto_mask = generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        o = self.temporal(src=src_enc, mask=player_mask, src_key_padding_mask=auto_mask)
    
        preds = self.out(o)
        
        return preds