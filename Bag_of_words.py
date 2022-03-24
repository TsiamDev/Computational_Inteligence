# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:12:54 2022

@author: Konstantinos Tsiamitros
"""

from scipy.sparse import coo_matrix
from collections import Counter
import numpy as np

from load_data import read_file



def count_freqs(words):
    rows = []
    
    for sentences in words:
        counts = []
        for sent in sentences:
            counts.append(Counter(sent))
        # combine the sentences into one dictionary for every document
        dictionary = {}
        for cnts in counts:
            for c in cnts:
                if c in dictionary:
                    dictionary[c] = cnts[c] + dictionary[c]
                else:
                    dictionary[c] = cnts[c]
        rows.append(dictionary)
    #print(rows)
    
    return rows

def counts_to_coo(filename):  
    
    words, sent_num, word_num = read_file(filename)     
    
    dictionaries = count_freqs(words)
    #print("----------------")
    rows = []
    data = []
    cols = []
    d_index = 0
    for dictionary in dictionaries:
        for key in dictionary.keys():
            #print(key,dictionary[key])
            cols.append(key)
            data.append(dictionary[key])
            rows.append(d_index)
        d_index = d_index + 1
    rows = np.array(rows)
    #print(rows)
    data = np.array(data)
    #print(data)
    cols = np.array(cols)
    
    #print(Counter(cols))
    
    coo = coo_matrix((data, (rows, cols)), shape=(len(dictionaries), 8520), dtype=int)
    #print(coo)
    
    return coo, words

def load_dataset():
    train_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-data.dat'
    test_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\test-data.dat'
    
    #train_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-data-small.dat'
    #test_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\test-data-small.dat'
    
    train_coo, train_words = counts_to_coo(train_file)
    test_coo, test_words = counts_to_coo(test_file)
    
    return train_coo, train_words, test_coo, test_words
#counts_to_coo()