#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
#from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, embed_size, dropout_rate = 0.3):
        """
        Parameters:
            embed_size = size of word embedding
            dropout_rate = probability of setting weights to zeros
        Layers:
            L_proj: Projection layer (batch_size, embed_size, embed_size)
            L_gate: Gate layer (batch_size, embed_size, embed_size)
        """
        super(Highway, self).__init__()
        self.L_proj = nn.Linear(embed_size, embed_size)
        self.L_gate = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p = dropout_rate)
        
        self.L_proj.bias.data.fill_(-0.1)
        self.L_gate.bias.data.fill_(-0.1)
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the Highway layer.
        
        @param X (Tensor): input is output of embedding conv layer. dimension of one word
        should be size of embedding word. (batch_size, embed_size)
        size > (batch_size, embed_size)
        @returns X (Tensor): Highwayed X  (batch_size, embed_size)
        """
        X_proj = F.relu(self.L_proj(X))
        X_gate = torch.sigmoid(self.L_gate(X))
        X_highway = X_gate*X_proj + (1 - X_gate)*X
        X = self.dropout(X_highway)
        
        return X
### END YOUR CODE 

