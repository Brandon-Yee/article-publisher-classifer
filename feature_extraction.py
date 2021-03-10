# -*- coding: utf-8 -*-
"""
Article Classifier Pre-Processing Functions
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK lemmatization/Stemming of the words before processing
# https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
word_transforms = ['no stem', 'stem']
vocab_types = ['mutual info', 'tf-idf']
BOG_types   = ['binary count', 'log count', 'raw count']

# Scikit Learn Feature Extraction
# https://scikit-learn.org/stable/modules/feature_extraction.html
# Part of speech tagger
# http://www.nltk.org/book/ch05.html
features    = ['article length', 'title length', 'BOW', 'POS']

# Bigram features?
# word 2 vec

# Compute mutual info for vocab
def mutual_info(X_train, Y_train, vect=None):
    """
    p(xi) = proportion of documents that contain word xi
    p(xi, y) = proportion of documents that contain xi in class y
    p(y) = proportion of documents that are class y
    """
    print('Computing mututal information')
    if vect is None:
        vect = CountVectorizer(binary=True)
    
    doc_word_counts = vect.fit_transform(X_train).toarray()
    n, d = doc_word_counts.shape
    
    px = np.sum(doc_word_counts, axis=0) / n
    py = Y_train.value_counts() / n
    info = np.zeros(d)
    for y in range(py.size):
        print('{}/{}'.format(y+1, py.size))
        pxy = (doc_word_counts[Y_train == py.index[y]].sum(axis=0)+1) / (n+1)
        info += np.multiply(pxy, np.log2(np.divide(pxy, np.multiply(px, py[y]))))
            
    return info, vect

# Generates a vocabulary from the training data 
def get_vocab(X_train, Y_train, length=5000, vocab_type='mutual info'):
    if vocab_type == 'mutual_info':
        info, vectorizer = mutual_info(X_train, Y_train)
        sorted_idx = np.argsort(info)[-length:]
        vocab = np.array(vectorizer.get_feature_names())[sorted_idx]
        sorted_info = info(sorted_idx)
        return vocab, sorted_info
    else:
        vectorizer = TfidfVectorizer(smooth_idf=True, max_features=length)
        sample_tf_idf = vectorizer.fit_transform(X_train)
        return vectorizer, sample_tf_idf

def tokenize(batch, tokenizer, vocabulary, max_title_len=50, max_article_len=1000):
    for i in range(len(batch)):
        title_tokens = tokenizer(batch['title'].iloc[i])
        for j, token in enumerate(title_tokens):
            if token not in vocabulary:
                title_tokens[j] = 'unk'
                
        title_tokens.insert(0, 's')
        if (max_title_len - len(title_tokens)) > 0:
            title_pad = ['e'] * (max_title_len - len(title_tokens))
            title_tokens.extend(title_pad)
        else:
            for i in range(len(title_tokens) - max_title_len + 1):
                title_tokens.pop()
                
            title_tokens.append('e')
                        
        article_tokens = tokenizer(batch['article'].iloc[i])
        for j, token in enumerate(article_tokens):
            if token not in vocabulary:
                article_tokens[j] = 'unk'
                
        article_tokens.insert(0, 's')
        if (max_article_len - len(article_tokens)) > 0:
            article_pad = ['e'] * (max_article_len - len(article_tokens))
            article_tokens.extend(article_pad)
        else:
            for i in range(len(article_tokens) - max_article_len + 1):
                article_tokens.pop()
                
            article_tokens.append('e')
            
        title_tokens.extend(article_tokens)
        
    return title_tokens

def get_label_idx(Y, publications):
    indices = []
    if type(Y) is not list:
        return publications.index(Y)
    else:
        for y in Y:
            indices.append(publications.index(y))
    
    return np.array(indices)

def tokenize_cnn(batch, tokenizer, vocabulary, max_title_len=5000, max_article_len=5000):
    processed = []
    for i in range(len(batch)):
        title_tokens = tokenizer(batch['title'].iloc[i])
        for j, token in enumerate(title_tokens):
            if token not in vocabulary:
                title_tokens[j] = 'unk'

        title_tokens.insert(0, 's')
        if (max_title_len - len(title_tokens)) > 0:
            title_pad = ['e'] * (max_title_len - len(title_tokens))
            title_tokens.extend(title_pad)
        else:
            for i in range(len(title_tokens) - max_title_len + 1):
                title_tokens.pop()
                
            title_tokens.append('e')

        article_tokens = tokenizer(batch['article'].iloc[i])
        for j, token in enumerate(article_tokens):
            if token not in vocabulary:
                article_tokens[j] = 'unk'

        article_tokens.insert(0, 's')
        if (max_article_len - len(article_tokens)) > 0:
            article_pad = ['e'] * (max_article_len - len(article_tokens))
            article_tokens.extend(article_pad)
        else:
            for i in range(len(article_tokens) - max_article_len + 1):
                article_tokens.pop()
                
            article_tokens.append('e')

        each_entry = [title_tokens, article_tokens]
        processed.append(each_entry)

    return processed
