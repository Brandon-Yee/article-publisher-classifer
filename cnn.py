import math
import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 10000
MAX_LEN = 500

KERN_CONV1 = (7, 300)
STRIDE_CONV1 = 1
NUM_C_OUT_CONV1 = 16

KERN_POOL1 = KERN_CONV1[0]
STRIDE_POOL1 = KERN_POOL1

conv1_out_D = (math.floor((MAX_LEN - (KERN_CONV1[0] - 1) - 1) / STRIDE_CONV1 + 1),
               math.floor((300 - (KERN_CONV1[1] - 1) - 1) / STRIDE_CONV1 + 1))
max1_out_L = math.floor((conv1_out_D[0] - (KERN_POOL1 - 1) - 1) / STRIDE_POOL1 + 1)


KERN_CONV2 = (8, 300)
STRIDE_CONV2 = 1
NUM_C_OUT_CONV2 = 16

KERN_POOL2 = KERN_CONV2[0]
STRIDE_POOL2 = KERN_POOL2

conv2_out_D = (math.floor((MAX_LEN - (KERN_CONV2[0] - 1) - 1) / STRIDE_CONV2 + 1),
               math.floor((300 - (KERN_CONV2[1] - 1) - 1) / STRIDE_CONV2 + 1))
max2_out_L = math.floor((conv2_out_D[0] - (KERN_POOL2 - 1) - 1) / STRIDE_POOL2 + 1)


KERN_CONV3 = (9, 300)
STRIDE_CONV3 = 1
NUM_C_OUT_CONV3 = 16

KERN_POOL3 = KERN_CONV3[0]
STRIDE_POOL3 = KERN_POOL3

conv3_out_D = (math.floor((MAX_LEN - (KERN_CONV3[0] - 1) - 1) / STRIDE_CONV3 + 1),
               math.floor((300 - (KERN_CONV3[1] - 1) - 1) / STRIDE_CONV3 + 1))
max3_out_L = math.floor((conv3_out_D[0] - (KERN_POOL3 - 1) - 1) / STRIDE_POOL3 + 1)


KERN_CONV4 = (10, 300)
STRIDE_CONV4 = 1
NUM_C_OUT_CONV4 = 16

KERN_POOL4 = KERN_CONV4[0]
STRIDE_POOL4 = KERN_POOL4

conv4_out_D = (math.floor((MAX_LEN - (KERN_CONV4[0] - 1) - 1) / STRIDE_CONV4 + 1),
               math.floor((300 - (KERN_CONV4[1] - 1) - 1) / STRIDE_CONV4 + 1))
max4_out_L = math.floor((conv4_out_D[0] - (KERN_POOL4 - 1) - 1) / STRIDE_POOL4 + 1)


lin_in_D = max1_out_L * NUM_C_OUT_CONV1 + max2_out_L * NUM_C_OUT_CONV2 \
    + max3_out_L * NUM_C_OUT_CONV3 + max4_out_L * NUM_C_OUT_CONV4


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

        self.conv2 = nn.Conv2d(2, NUM_C_OUT_CONV2, KERN_CONV2)
        self.pool2 = nn.MaxPool1d(KERN_POOL2)

        self.conv3 = nn.Conv2d(2, NUM_C_OUT_CONV3, KERN_CONV3)
        self.pool3 = nn.MaxPool1d(KERN_POOL3)

        self.conv4 = nn.Conv2d(2, NUM_C_OUT_CONV4, KERN_CONV4)
        self.pool4 = nn.MaxPool1d(KERN_POOL4)

        self.lin = nn.Linear(lin_in_D, 11)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(torch.squeeze(x1))
        x1 = self.pool1(x1)
        x1 = torch.flatten(x1, start_dim=1)
        print('after flatten:', x1.shape)

        x2 = self.conv2(x)
        x2 = F.relu(torch.squeeze(x2))
        x2 = self.pool2(x2)
        x2 = torch.flatten(x2, start_dim=1)
        print('after flatten:', x2.shape)

        x3 = self.conv3(x)
        x3 = F.relu(torch.squeeze(x3))
        x3 = self.pool3(x3)
        x3 = torch.flatten(x3, start_dim=1)
        print('after flatten:', x3.shape)

        x4 = self.conv4(x)
        x4 = F.relu(torch.squeeze(x4))
        x4 = self.pool4(x4)
        x4 = torch.flatten(x4, start_dim=1)
        print('after flatten:', x4.shape)

        combined = torch.cat((x1, x2, x3, x4), dim=1)
        combined = self.lin(combined)
        combined = F.log_softmax(combined, dim=1)

        return combined
