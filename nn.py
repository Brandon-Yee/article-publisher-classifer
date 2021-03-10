# -*- coding: utf-8 -*-
"""
Article Classifier Neural Network Baseline

A NN-based classifier for classifying article publishers based on title and 
article.

"""

import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    """
    Article Publisher Classifier Neural Network Definition
    
    PARAMETERS:
    in_size - int: size of the input vector
    hidden_sizes - list: hidden layer sizes
    
    Word2Vec Embedding Layer Example:
    https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

    """
    def __init__(self, in_size, hidden_sizes, device):
        super(NN, self).__init__()
        #self.embeddings = word2vec #model #nn.Embedding.from_pretrained(weights)
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_size, hidden_sizes[0]))
        self.device = device
        for i in range(1, len(hidden_sizes)):
            self.linear.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        self.linear.append(nn.Linear(hidden_sizes[-1], 11))
        
    def forward(self, X):
        for i in range(len(self.linear)-1):
            X = F.relu(self.linear[0](X))
        X = F.log_softmax(self.linear[-1](X), dim=1)
        return X
        
    