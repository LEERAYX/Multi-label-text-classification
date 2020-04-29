# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:35:08 2020

@author: ruixuanl
"""

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.sentiment.util import mark_negation


import pandas as pd
import os


class Config:
    
    # Data and output directory config
    data_path = r'./Data'
    output_path = r'./NaiveBayesOutput'
    
    tone_train = r'tone_train.txt'
    tone_test = r'tone_test.txt'
    code_train = r'code_train.txt'
    code_test = r'code_test.txt'
    
    # Setting config
    is_lower = True
    is_neg_mark = False
    is_stop = True
    num_gram = 1
    
    
    # Set stop word list
    stop_list = set(stopwords.words("english"))
    
    
    
def create_ngrams_features(words, n=1):
    """

    Parameters
    ----------
    words : list
        List of tokens.
    n : int, optional
        n-gram. The default is 1.

    Returns
    -------
    Bag of words in dictionary format.

    """
    
    if Config.is_lower:
        words = [word.lower() for word in words]
    
    if Config.is_stop:
        words = [word for word in words if word not in Config.stop_list]
    
    vocab_ngram = ngrams(words, n)
    
    
    
    word_dict = dict([(ngs, True) for ngs in vocab_ngram])
    
    return word_dict


def tokenize_text(text):
    """
    

    Parameters
    ----------
    text : string
        Text to be tokenized.

    Returns
    -------
    Tokenized tokens.

    """
    
    words = nltk.word_tokenize(text)
    
    if Config.is_neg_mark:
        words = mark_negation(words)
    
    return words

def text_process(X, n):
    
    output = []
    for text in X:
        
        try:
            words = tokenize_text(text)
            output.append(create_ngrams_features(words, n))
        except:
            output.append({})
        
    return output


if __name__ == "__main__":
    
    train_dt = pd.read_csv(os.path.join(Config.data_path, Config.tone_train), sep = '\t', encoding='cp1252')
    test_dt = pd.read_csv(os.path.join(Config.data_path, Config.tone_test), sep = '\t', encoding='cp1252')
    
    print('----- Tone classifier -----')
    for n in range(1, 6):
        
        X_train = text_process(list(train_dt['text']), n)
        X_test = text_process(list(test_dt['text']), n)
        y_train = list(train_dt['label'])
        y_test = list(test_dt['label'])
    
        train_set = []
        for i in range(len(X_train)):
            train_set.append((X_train[i], y_train[i]))
    
        test_set = []
        for i in range(len(X_test)):
            test_set.append((X_test[i], y_test[i]))
        
    
        classifier = NaiveBayesClassifier.train(train_set)
    
        accuracy = nltk.classify.util.accuracy(classifier, test_set)
        
        print('{}-gram accuracy {:.4f}'.format(n, accuracy))
        
        
        
    print('----- Code classifier -----')
    
    train_dt = pd.read_csv(os.path.join(Config.data_path, Config.code_train), sep = '\t', encoding='cp1252')
    test_dt = pd.read_csv(os.path.join(Config.data_path, Config.code_test), sep = '\t', encoding='cp1252')
    
    
    for n in range(1, 6):
        
        X_train = text_process(list(train_dt['text']), n)
        X_test = text_process(list(test_dt['text']), n)
        y_train = list(train_dt['label'])
        y_test = list(test_dt['label'])
    
        train_set = []
        for i in range(len(X_train)):
            train_set.append((X_train[i], y_train[i]))
    
        test_set = []
        for i in range(len(X_test)):
            test_set.append((X_test[i], y_test[i]))
        
    
        classifier = NaiveBayesClassifier.train(train_set)
    
        accuracy = nltk.classify.util.accuracy(classifier, test_set)
        
        print('{}-gram accuracy {:.4f}'.format(n, accuracy))
    
    
