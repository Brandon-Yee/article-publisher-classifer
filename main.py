# -*- coding: utf-8 -*-
"""
Main Script to Run the Classifier
"""
import numpy as np
from classifier import generate_data_sets, save_data, load_data
from feature_extraction import mutual_info, get_vocab

if __name__ == "__main__":
    #train, val, test = generate_data_sets('./article_data.csv')
    #save_data(train, val, test)
    train, val, test = load_data()
    
    BOW_word_length = [1000, 2000, 5000, 10000]
    vectorizer, data = get_vocab(train['title'], train['publication'], length=5000, vocab_type='tf-idf')
    vocab = vectorizer.vocabulary_