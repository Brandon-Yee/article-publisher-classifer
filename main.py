# -*- coding: utf-8 -*-
"""
Main Script to Run the Classifier
"""
#from classifier import generate_data_sets
from classifier import generate_data_sets, load_data
from feature_extraction import mutual_info

if __name__ == "__main__":
    #train, val, test = generate_data_sets('./article_data.csv')
    train, val, test = load_data()
    info, vect = mutual_info(train['title'], train['publication'])