# -*- coding: utf-8 -*-
"""
Dataset Class for Article Classifier

adapted from https://discuss.pytorch.org/t/data-processing-as-a-batch-way/14154/4
"""
import torch
from torch.utils.data import Dataset
from feature_extraction import tokenize, get_label_idx

class ArticleDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, chunksize, word2vec, publications,
                 max_article_len=1000, max_title_len=50):
        self.df = df
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.chunksize = chunksize
        self.embeddings = word2vec
        self.publications = publications
        self.max_article_len = max_article_len
        self.max_title_len = max_title_len
        self.len = len(df) // chunksize
        
    def __getitem__(self, index):
        tokenized_batch = tokenize(self.df.iloc[index*self.chunksize:index*self.chunksize+self.chunksize], 
                     self.tokenizer, 
                     self.vocab,
                     self.max_title_len,
                     self.max_article_len)
        
        embedded_batch = torch.FloatTensor(self.embeddings[tokenized_batch]).view(1, -1)
        labels = torch.as_tensor(get_label_idx(self.df['publication'].iloc[index], self.publications)).long()
        return embedded_batch.squeeze(), labels
    
    def __len__(self):
        return self.len

