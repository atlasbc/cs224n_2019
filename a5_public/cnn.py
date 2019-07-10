#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A CNN that consists of Conv1d
    
    @param input_size: (char_embedding_size, maximum_word_length) > 1 word
    @param num_filters: Number of filters. Should be equal to Word embedding size.
    @param kernel_size: Default is 5. How many characters is looked in a word at one time.
    batch size is number of sentences. 
    """  
    def __init__(self, char_embedding_size, num_filters, kernel_size = 5):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(char_embedding_size, num_filters, kernel_size)
    
    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        """
        Takes batches of words that consists of character embeddings.
        Returns word embedding that combines character embeddings with CNN for each word.
        @param X_reshaped: Batches of words consists of character embeddings. (batch_size, )
        
        """
#        print("X_reshaped size", X_reshaped.size())
        X_conv = self.cnn(X_reshaped)
        X_conv_out = F.relu(X_conv)
#        print("Size after CNN_applied", X_conv_out.size())
        X_conv_out = torch.max(X_conv_out, dim=2)[0] #torch max returns(max_value, max_value_indices)
#        print("Size after max pool applied",X_conv_out.size())
        return X_conv_out
        

### END YOUR CODE

