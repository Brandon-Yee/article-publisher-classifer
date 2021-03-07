# -*- coding: utf-8 -*-
"""
Main Script to Run the Classifier
"""
import numpy as np
from classifier import generate_data_sets, save_data, load_data
from feature_extraction import get_vocab, tokenize
from nn import NN

VOCAB_SIZE = 10000

if __name__ == "__main__":
    #train, val, test = generate_data_sets('./article_data.csv')
    #save_data(train, val, test)
    train, val, test = load_data()
    
    # Generate the vocabulary based on the top TF-IDF weighted words
    model = NN(10, [1000])
    vectorizer, data = get_vocab(train['title'], train['publication'], length=VOCAB_SIZE, vocab_type='tf-idf')
    print(len(vectorizer.vocabulary_))
    out_of_vocab = []
    for word in vectorizer.vocabulary_.keys():
        if word not in model.embeddings.vocab.keys():
            out_of_vocab.append(word)
    
    for word in out_of_vocab:
        vectorizer.vocabulary_.pop(word)
    
    print(len(vectorizer.vocabulary_))
    tokenizer = vectorizer.build_tokenizer()

    # Get the tokenized title + article
    val_tokenized = tokenize(val, tokenizer, vectorizer.vocabulary_.keys(), max_article_len=1000)
    
    # Get the word embedding
    d = len(val_tokenized[0])
    model.embeddings[val_tokenized[0]]
    