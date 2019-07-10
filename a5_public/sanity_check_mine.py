# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:55:46 2019

@author: Sony
"""
import torch
from torch import nn
from cnn import CNN


#wtest = torch.ones((10, 8, 18)) #max_sentence_length(# of words), char_embedding_size, max_word_length
#cnn_test = CNN(8, 50)
#cnn_result = cnn_test(wtest)
#print(50*"*")
#print(cnn_result.size())

#test with embedding 
#input (sentence_length, batch_size, max_word_length)
#input after embedding (sentence_length, batch_size, max_word_length, embed_size)

etest = torch.ones((10, 4, 18, 8))
cnn_test = CNN(8, 50)

#reshaping should be (40, 8, 18)
etest = etest.permute((0, 1, 3, 2))
print(etest.size())
etest = etest.view(-1, 8, 18)
#after cnn should be (40, 50)
cnn_result = cnn_test(etest)
print(cnn_result.size())
#reshape again (10, 4, 50)
cnn_result = cnn_result.view(10, 4, -1)
print("cnn_result", cnn_result.size())

