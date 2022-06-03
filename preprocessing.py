# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:19:14 2022

@author: Konstantinos Tsiamitros
"""

import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

from Bag_of_words import load_dataset

from scipy.sparse import coo_matrix

from Apply_GA import Apply_GA

def standardize(X):
    return (X - X.mean()) / X.std() 

def plot(coo, xs2, ys2):
  
    # plain data
    plt.scatter(coo.row, coo.data)
    plt.show()

    xs2 = sorted(list(xs2))
    #print(xs2)
    lst = dict(sorted(ys2.items(), key=lambda item: item[1], reverse=True))
    #print(lst.keys())
    #print(lst.values())

    lst = np.array(list(lst.values()))

    # centering
    #mean = sum(lst) / len(lst)
    #ys = lst - mean

    # normalization
    #lst = np.array(lst, dtype=int)
    #ys = preprocessing.normalize(lst.reshape(1,-1))

    # standardization
    ys = standardize(lst)

    # plain data
    plt.scatter(range(0, ys.size), ys)
    plt.show()
    
    return lst, ys

def make_train_dataset(coo):
    data = []
    cols = []
    cnt = 0
    temp = []
    temp2 = []
    for r in range(0, len(coo.row)):
        if coo.row[r] == cnt:
            temp.append(coo.col[r])
            temp2.append(coo.data[r])
        else:
            cols.append(temp)
            data.append(temp2)
            cnt = cnt + 1
            temp = []
            temp2 = []
            temp.append(coo.col[r])
            temp2.append(coo.data[r])
    data.append(temp2)
    cols.append(temp)
    
    train = np.zeros((len(cols), 8520), dtype=float)
    dim = 0
    train2 = []
    for i in range(0, len(cols)):
        temp = []
        for j in range(0, len(cols[i])):
            x = data[i][j]
            if (x >= 5) and (x <= 20):
                train[i, j] = x
                temp.append(x)
                dim = dim + 1
        train2.append(temp)
    
    train2 = [x for x in train2 if x]
    train2 = np.array(train2, dtype=object)  
    
    return train
            
def preprocess_and_plot(GA_words):
    print("Loading files...")
    train_coo, train_words, test_coo, test_words = load_dataset(GA_words)
    
    print("Preprocessing...")
    
    train = make_train_dataset(train_coo)
    test = make_train_dataset(test_coo)
    
    # centering
    #train = train - train.mean()
    #test = test - test.mean()
    
    # standardization - Uncomment these
    #scaler = preprocessing.StandardScaler().fit(train)
    #train = scaler.transform(train)
    
    #scaler = preprocessing.StandardScaler().fit(test)
    #test = scaler.transform(test)
    
    # normalization
    #train = preprocessing.normalize(train)
    #test = preprocessing.normalize(test)
    
    c = coo_matrix(train)

    x = [i for i in range(0, len(c.col)) if c.col[i] == 0]
    y = [c.data[i] for i in range(0, len(c.data)) if c.col[i] == 0]
    plt.plot(x, y)
    plt.show()
    
    # lst contains the dictionary of the terms with associated term frequency
    # ys contains the preprocessed term frequencies
    # both are sorted in descending order
    #return lst, ys
    return train, test
    
def Preprocess_and_apply_GA():
    GA_words = Apply_GA()    
    print(GA_words)
    
    train, test = preprocess_and_plot(GA_words)
    print("len(words): ", len(GA_words))
    
    return train

#train = Preprocess_and_apply_GA()
