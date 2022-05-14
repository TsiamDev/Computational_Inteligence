# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:46:42 2022
@author: Konstantinos Tsiamitros
"""
#import tensorflow as tf
#from sklearn import preprocessing

#from scipy.sparse import coo_matrix, csr_matrix

from load_data import read_file

def load_vecs():
    train_file = 'C:\\Users\\HomeTheater\\Desktop\\Dataset\\Data\\train-data.dat'
    words, sent_num, word_num = read_file(train_file) 
    
    data = []
    _sent_num = []
    for i in range(0, len(sent_num)):
        _sent_num = sent_num[i][0]
        
        w = []
        wn = []
        for j in range(0, len(word_num[i])):
            wn.append(word_num[i][j])
            w.append(words[i][j])

        for n in range(0, len(wn)):
            w[n].insert(0, int(wn[n]))
        
        t1 = [item for sublist in w for item in sublist]
        t1.insert(0, int(_sent_num))
        data.append(t1)
    
    return data

# - Loads the raw vector data from an input csv file,
# - Pads the vectors with preppended zeros so as to achieve
#   specific MxN dimensions and
# - Standardizes the data
def preprocess_vecs():
    data = load_vecs()

    # padding
    #data = tf.keras.preprocessing.sequence.pad_sequences(data)#, padding='post')
    
    # standardization
    #scaler = preprocessing.StandardScaler().fit(data)
    #data = scaler.transform(data)
    
    #data = csr_matrix(data)
    #data = tf.sparse.from_dense(data)
    return data
data = preprocess_vecs()