import numpy as np
import pandas as pd
import gensim
from classifier import generate_data_sets, save_data, load_data
from feature_extraction import get_vocab, tokenize_cnn
import torch
#from cnn import CNN


VOCAB_SIZE = 10000
MAX_LEN = 1000

def main():

train, val, test = load_data()
train_sub = train.iloc[:500].copy()

vectorizer, data = get_vocab(train['title'], train['publication'], length=VOCAB_SIZE, vocab_type='tf-idf')
print(len(vectorizer.vocabulary_))

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

out_of_vocab = []
for word in vectorizer.vocabulary_.keys():
    if word not in model:
        out_of_vocab.append(word)
    
for word in out_of_vocab:
    vectorizer.vocabulary_.pop(word)
    
print(len(vectorizer.vocabulary_))
tokenizer = vectorizer.build_tokenizer()


## start batch processing

# Get the tokenized title + article
sub_tokenized = tokenize_cnn(train_sub, tokenizer,
                             vectorizer.vocabulary_.keys(),
                             max_title_len=MAX_LEN, max_article_len=MAX_LEN)

batch_tensor = torch.zeros((len(sub_tokenized), MAX_LEN, 300, 2))
for i in range(len(sub_tokenized)):
    print(i)
    batch_tensor[i, :, :, 0] = torch.from_numpy(model[sub_tokenized[i][0]])
    batch_tensor[i, :, :, 1] = torch.from_numpy(model[sub_tokenized[i][1]])
    
