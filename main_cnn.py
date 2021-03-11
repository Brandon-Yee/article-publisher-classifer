import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from classifier import generate_data_sets, save_data, load_data
from feature_extraction import get_vocab, tokenize_cnn
import torch
import torch.nn as nn
from cnn import CNN
from cnn import MAX_LEN
from cnn import VOCAB_SIZE
from sklearn import preprocessing


#VOCAB_SIZE = 10000
#MAX_LEN = 1000
STEP_SIZE = .01

"""
KERN_CONV1 = 4
STRIDE_CONV1 = 1
KERN_POOL1 = 5
STRIDE_MAX1 = 5

conv2max_L = math.floor((MAX_LEN - (KERN_CONV1 - 1) - 1) / STRIDE_CONV1 + 1)
max2lin_L = math.floor((conv2max_L - (KERN_POOL1 - 1) - 1) / STRIDE_MAX1 + 1)
"""

"""
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
"""

#def main():
BATCH_SIZE = 200
NUM_EPOCHS = 10

tgt_classes = [
        'Reuters',
        'TechCrunch',
        'Economist',
        'CNN',
        'CNBC',
        'Fox News',
        'Politico',
        'The New York Times',
        'Vox',
        'Business Insider',
        'Other'
        ]

le = preprocessing.LabelEncoder()
le.fit(tgt_classes)


train, val, test = load_data()
train_sub = train.iloc[:500].copy()

num_batches_in_train = math.ceil(train.shape[0] / BATCH_SIZE)

vectorizer, data = get_vocab(train['title'], train['publication'], length=VOCAB_SIZE, vocab_type='tf-idf')
print(len(vectorizer.vocabulary_))

embed_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

out_of_vocab = []
for word in vectorizer.vocabulary_.keys():
    if word not in embed_model:
        out_of_vocab.append(word)
    
for word in out_of_vocab:
    vectorizer.vocabulary_.pop(word)
    
print(len(vectorizer.vocabulary_))
tokenizer = vectorizer.build_tokenizer()

cnn = CNN()
#cnn = cnn.to('cuda')
#cnn = cnn.cuda()

loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(cnn.parameters(), lr=STEP_SIZE)


acc_train = 0.0
num_epochs = 0

train_loss_list = []
val_loss_list = []

train_acc_list = []
val_acc_list = []

for number in range(NUM_EPOCHS):

    epoch_loss = 0.0
    running_loss = 0.0
    running_count = 0

    tot_right = 0
    tot = 0


    ## start batch processing
    for i in range(num_batches_in_train):
        # grab a batch
        if (i+1)*BATCH_SIZE > train.shape[0]:
            X_batch = train.iloc[i*BATCH_SIZE:]
            y_batch = train['publication'].iloc[i*BATCH_SIZE:]
        else:
            X_batch = train.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = train['publication'].iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        y_coded = le.transform(y_batch.values)
        y_coded = torch.from_numpy(y_coded)
        y_coded = torch.as_tensor(y_coded, dtype=torch.long)
        #y_coded = y_coded.to(device)

        # Get the tokenized title + article
        sub_tokenized = tokenize_cnn(X_batch, tokenizer,
                                     vectorizer.vocabulary_.keys(),
                                     max_title_len=MAX_LEN, max_article_len=MAX_LEN)

        batch_tensor = torch.zeros((len(sub_tokenized), 2, MAX_LEN, 300))
        for i in range(len(sub_tokenized)):
            batch_tensor[i, 0, :, :] = torch.from_numpy(embed_model[sub_tokenized[i][0]])
            batch_tensor[i, 1, :, :] = torch.from_numpy(embed_model[sub_tokenized[i][1]])

        #batch_tensor = batch_tensor.to(device)
        # get publication target

        optim.zero_grad()
        probs = cnn(batch_tensor)
        y_hat = probs.argmax(axis=1)

        loss = loss_func(probs, y_coded)
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        running_loss += loss.item()
        running_count += len(y_coded)

        tot_right += torch.count_nonzero(y_hat == y_coded).item()
        tot += len(y_coded)

        if i % (num_batches_in_train // 4) == ((num_batches_in_train // 4) - 1):
            print('[%d, %d]\trunning_loss:\t%f' %
                  (num_epochs + 1, i + 1, running_loss / running_count))
            running_loss = 0.0
            running_count = 0

    acc_train = tot_right / tot

    train_loss_list.append(epoch_loss / BATCH_SIZE)
    train_acc_list.append(acc_train)


    # check validation
    loss_val = 0.0
    vtot_right = 0
    vtot = 0
    val_n = val.shape[0]
    each = int(val_n // 10)

    for i in range(10):
        X_batch_val = val.iloc[i*each:(i+1)*each]
        y_batch_val = train['publication'].iloc[i*each:(i+1)*each]

        y_coded_val = le.transform(y_batch_val.values)
        y_coded_val = torch.from_numpy(y_coded_val)
        y_coded_val = torch.as_tensor(y_coded_val, dtype=torch.long)
        #y_coded_val = y_coded_val.to(device)

        # Get the tokenized title + article
        sub_tokenized_val = tokenize_cnn(X_batch_val, tokenizer,
                                         vectorizer.vocabulary_.keys(),
                                         max_title_len=MAX_LEN, max_article_len=MAX_LEN)

        batch_tensor_val = torch.zeros((len(sub_tokenized_val), 2, MAX_LEN, 300))
        for i in range(len(sub_tokenized)):
            batch_tensor_val[i, 0, :, :] = torch.from_numpy(embed_model[sub_tokenized_val[i][0]])
            batch_tensor_val[i, 1, :, :] = torch.from_numpy(embed_model[sub_tokenized_val[i][1]])

        #batch_tensor_val = batch_tensor_val.to(device)

        probs_val = cnn(batch_tensor_val)
        y_hat_val = probs_val.argmax(axis=1)
        print(y_hat_val[:100])

        loss_val += loss_func(probs_val, y_coded_val).item()
        vtot_right += torch.count_nonzero(y_hat_val == y_coded_val).item()
        vtot += len(y_coded_val)

    acc_val = vtot_right / vtot

    val_loss_list.append(loss_val / vtot)
    val_acc_list.append(acc_val)


    print()
    print('End of Epoch\t%d -------' % (number+1))
    print('\tTraining Loss:\t\t%f' % (epoch_loss))
    print('\tValidation Loss:\t%f' % (loss_val))
    print('\tTraining Accuracy:\t%f' % (acc_train))
    print('\tValidation Accuracy:\t%f' % (acc_val))
    print()


plt.plot(range(1, 11), train_acc_list, label='Train')
plt.plot(range(1, 11), val_acc_list, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch for 4 layer Conv CNN')
plt.legend()
plt.grid()

plt.show()

