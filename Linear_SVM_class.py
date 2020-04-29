# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:35:08 2020

@author: ruixuanl
"""


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

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
    


if __name__ == "__main__":
    
    print('--------Tone classification--------\n')
    
    train_dt = pd.read_csv(os.path.join(Config.data_path, Config.tone_train), sep = '\t', encoding='cp1252')
    test_dt = pd.read_csv(os.path.join(Config.data_path, Config.tone_test), sep = '\t', encoding='cp1252')
    
    train_text = ['' if type(t) == float else t for t in train_dt['text'].values]
    test_text = ['' if type(t) == float else t for t in test_dt['text'].values]
    
    tfidf = TfidfVectorizer(
        sublinear_tf=True, 
        min_df=3, norm='l2', 
        encoding='latin-1', 
        ngram_range=(1, 2), 
        stop_words='english')
    
    X = tfidf.fit_transform(train_text)
    
    y = list(train_dt['label'] - 1)
    
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.33, random_state = 1004)
    
    model = LinearSVC()
    model.fit(X_train, y_train)
    
    y_pred_dev = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred_dev, normalize=True, sample_weight=None)
    
    print("Development accuracy: {:.4f}".format(accuracy))
    
    X_test = tfidf.transform(test_text)
    y_test = list(test_dt['label'] - 1)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    
    test_dt['pred'] = y_pred + 1
    
    print("Test Accuracy: {:.4f}".format(accuracy))
    
    
    print('--------Code classification--------\n')
    
    train_dt = pd.read_csv(os.path.join(Config.data_path, Config.code_train), sep = '\t', encoding='cp1252')
    test_dt = pd.read_csv(os.path.join(Config.data_path, Config.code_test), sep = '\t', encoding='cp1252')
    
    train_text = ['' if type(t) == float else t for t in train_dt['text'].values]
    test_text = ['' if type(t) == float else t for t in test_dt['text'].values]
    
    tfidf = TfidfVectorizer(
        sublinear_tf=True, 
        min_df=4, norm='l2', 
        encoding='latin-1', 
        ngram_range=(1, 2), 
        stop_words='english')
    
    X = tfidf.fit_transform(train_text)
    
    y = list(train_dt['label'] - 1)
    
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.33, random_state = 1004)
    
    model = LinearSVC(random_state=1004)
    model.fit(X_train, y_train)
    
    y_pred_dev = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred_dev, normalize=True, sample_weight=None)
    
    print("Development accuracy: {:.4f}".format(accuracy))
    
    X_test = tfidf.transform(test_text)
    y_test = list(test_dt['label'] - 1)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    
    print("Test Accuracy: {:.4f}".format(accuracy))
    
    test_dt['pred'] = y_pred + 1
