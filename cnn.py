import math
import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 10000
MAX_LEN = 200

KERN_CONV1 = (5, 300)
STRIDE_CONV1 = 1
NUM_C_OUT_CONV1 = 10

KERN_POOL1 = 20
STRIDE_POOL1 = KERN_POOL1

convOut_D = (math.floor((MAX_LEN - (KERN_CONV1[0] - 1) - 1) / STRIDE_CONV1 + 1),
             math.floor((300 - (KERN_CONV1[1] - 1) - 1) / STRIDE_CONV1 + 1))
max2lin_L = math.floor((convOut_D[0] - (KERN_POOL1 - 1) - 1) / STRIDE_POOL1 + 1)


class CNN(nn.Module):
    """
    Article Publisher Classifier Neural Network Definition
    
    PARAMETERS:
    in_size - int: size of the input vector
    hidden_sizes - list: hidden layer sizes
    
    Word2Vec Embedding Layer Example:
    https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

    some examples: https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f
    """
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(2, NUM_C_OUT_CONV1, KERN_CONV1)
        self.pool1 = nn.MaxPool1d(KERN_POOL1)
        self.lin = nn.Linear(max2lin_L*NUM_C_OUT_CONV1, 11)

    def forward(self, x):
        x = self.conv1(x)
        #print('after conv1:', x.shape)
        x = F.relu(torch.squeeze(x))
        #print('after squeeze, and relu:', x.shape)
        x = self.pool1(x)
        #print('after pool1:', x.shape)
        x = torch.flatten(x, start_dim=1)
        #print('after flatten:', x.shape)
        x = self.lin(x)
        #print('after linear:', x.shape)
        x = F.log_softmax(x, dim=1)

        return x
