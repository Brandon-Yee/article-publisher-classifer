# -*- coding: utf-8 -*-
"""
Main Script to Run the Classifier
"""
import numpy as np
from classifier import generate_data_sets, save_data, load_data
from feature_extraction import get_vocab, tokenize

VOCAB_SIZE = 10000

if __name__ == "__main__":
    #train, val, test = generate_data_sets('./article_data.csv')
    #save_data(train, val, test)
    train, val, test = load_data()
    
    # Generate the vocabulary based on the top TF-IDF weighted words
    vectorizer, data = get_vocab(train['title'], train['publication'], length=VOCAB_SIZE, vocab_type='tf-idf')
    vocab = vectorizer.vocabulary_
    tokenizer = vectorizer.build_tokenizer()
    
    # Get the tokenized title + article
    val_tokenized = tokenize(val, tokenizer)
    
    # Get the word embedding