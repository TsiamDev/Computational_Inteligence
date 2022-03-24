# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:19:14 2022

@author: Konstantinos Tsiamitros
"""

import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

import statistics

from collections import Counter

from Bag_of_words import load_dataset

from scipy.fft import fft
from scipy.sparse import coo_matrix

def standardize(X):
    return (X - X.mean()) / X.std() 

def standardize2(X):
    return (X - sum(X)/len(X)) / statistics.stdev(X) 

# centering
#ys = coo.data - coo.data.mean()
#ys2 = coo.data - coo.data.mean()

# normalization
#ys = preprocessing.normalize(coo.data.reshape(1,-1))
#ys = preprocessing.normalize(ys.reshape(1,-1))
#ys2 = preprocessing.normalize(ys2.reshape(1,-1))

# standardization
#scaler = preprocessing.StandardScaler().fit(coo.data.reshape(1,-1))
#ys = scaler.transform(coo.data.reshape(1,-1))

#scaler = preprocessing.StandardScaler().fit(ys.reshape(1,-1))
#ys = scaler.transform(ys.reshape(1,-1))
#ys = standardize(coo.data)


# transformed data
#ys.sort(reverse=True)
#ys = sorted(ys, reverse=True)
#plt.scatter(range(0, len(ys)), ys)

# set axes range
#plt.xlim(0, 500)
#plt.xlim(-200, 5000)
#plt.ylim(-0.05, 0.05)

#plt.show()

def plot2(data):
    plt.scatter(data)
    plt.show()

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
    #plt.scatter(lst.keys(), lst.values())
    #plt.scatter(range(0, len(lst.values())), lst.values())
    plt.scatter(range(0, ys.size), ys)
    plt.show()
    
    return lst, ys

def make_train_dataset(words):
    print(words)
    doc_terms = []
    for sentences in words:
        temp = []
        for sent in sentences:
            for term in sent:
                temp.append(term)
        doc_terms.append(temp)
    print(Counter(doc_terms[0]))
    

def make_train_dataset2(coo):
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
    #print(data)
    #print(cols)
    
    train = np.zeros((len(cols), 8520), dtype=float)
    """
    for i in range(0, len(cols)):
        for j in range(0, 8520):
            if j in cols[i]:
                ind = cols[i].index(j)
                x = data[i][ind]
                train[i, j] = x
    """
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
    #train2 = np.array(train2, dtype=float)
    #print(train)
    
    train2 = [x for x in train2 if x]
    #for i in range(0, len(train2)):
    #    for j in range(0, len(train2[0])):
    #        print(1)
    train2 = np.array(train2, dtype=object)  
    
    return train
            
def preprocess_and_plot():
    print("Loading files...")
    train_coo, train_words, test_coo, test_words = load_dataset()

    print("Preprocessing...")
    
    #make_train_dataset(words)
    train = make_train_dataset2(train_coo)
    test = make_train_dataset2(test_coo)
    #cnt = 0
    #for r in coo.row:
    
    # centering
    #train = train - train.mean()
    #test = test - test.mean()
    
    # standardization
    #train = standardize2(train)
    #test = standardize2(test)
    
    # standardization
    scaler = preprocessing.StandardScaler().fit(train)
    train = scaler.transform(train)
    
    # normalization
    #train = preprocessing.normalize(train)
    #test = preprocessing.normalize(test)
    
    #scaler = preprocessing.StandardScaler().fit(test)
    #test = scaler.transform(test)

    #train = fft(train)
    #test = fft(test)
    #xs = [x for x in range(0, len(train))]
    #plt.plot(xs, train)
    #plt.show()
    #plt.plot(xs, test)
    #plt.show()
    
    c = coo_matrix(train)
    #c = fft(c.todense())
    #print(c)
    #c = coo_matrix(c)
    
    #plt.plot(c.col, test)
    #plt.show()
    #x = [i for i in c.col if i==0]
    x = [i for i in range(0, len(c.col)) if c.col[i] == 0]
    y = [c.data[i] for i in range(0, len(c.data)) if c.col[i] == 0]
    plt.plot(x, y)
    plt.show()
    
    """
    xs = []
    ys = []
    cnt = 0
    for i in range(0, len(train)):
        for j in range(0, len(train[0])):
            ys.append(train[i][j])
            xs.append(cnt)
            cnt = cnt + 1
    
    plt.scatter(xs, ys)
    plt.show()
    #"""
    """
    xs2 = set()
    ys2 = dict()
    for i in range(0, len(coo.data)):
        #print(coo.data[i], coo.row[i], coo.col[i])
        xs2.add(coo.col[i])
        if coo.col[i] in ys2:
            ys2[coo.col[i]] =  ys2[coo.col[i]] + coo.data[i]
        else:
            ys2[coo.col[i]] = coo.data[i]
          

    lst, ys = plot(coo, xs2, ys2)
    """
    
    
    # lst contains the dictionary of the terms with associated term frequency
    # ys contains the preprocessed term frequencies
    # both are sorted in descending order
    #return lst, ys
    return train, test

#preprocess_and_plot()
