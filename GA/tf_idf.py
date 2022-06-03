# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:48:23 2022

@author: HomeTheater
"""
from matplotlib import pyplot as plt
import math

from load_data import read_file

def get_tf(words): 
    
    mx = 0
    
    doc_tfs = []
    total_doc_terms = []
    idf_denominator = dict()
    for doc in words:
        if doc:
            doc_dict = dict()
            cnt = 0
            for sent in doc:
                for s in sent:
                    cnt = cnt + 1
                    if s not in doc_dict:
                        doc_dict[s] = 0
                    doc_dict[s] = doc_dict[s] + 1
            
            t = max(doc_dict.values())
            if t > mx:
                mx = t
            doc_tfs.append(doc_dict)
            total_doc_terms.append(cnt)
            
            doc_set = set(doc_dict)
            for t in doc_set:
                if t not in idf_denominator:
                    idf_denominator[t] = 1
                else:
                    idf_denominator[t] = idf_denominator[t] + 1
    
    print("TF:", "min denominator: ", min(total_doc_terms), "max numerator: ", mx)
    tf_upper = mx / (math.log(8521, 10) * min(total_doc_terms))
    print("TF -> [0,", tf_upper, "]")
    print("IDF -> [0,", math.log(8521/1, 10),"]")
    print("TF-IDF -> [0",  math.log(8521, 10) * tf_upper, "]")
    # len(words) : is the number of documents
    return doc_tfs, total_doc_terms, len(words), idf_denominator

def get_tfs_idfs(doc_tfs, total_doc_terms, num_of_docs, idf_denominator):
    # key: term, value: present in # of docs
    term_dict = dict()
    idfs = dict()
    # for every document
    for i in range(0, len(doc_tfs)):
        # for every term
        for term in doc_tfs[i]:
            tf_temp = doc_tfs[i][term] / total_doc_terms[i]
            idf_temp = math.log(num_of_docs/idf_denominator[i], 10)
            if term not in term_dict:
                term_dict[term] = 0
            term_dict[term] = term_dict[term] + 1
            if term not in idfs:
                idfs[term] = 0
            idfs[term] = tf_temp * idf_temp + idfs[term]
    
    return term_dict, idfs

def PrepareFitnessData():
    #file = 'C:\\Users\\HomeTheater\\Desktop\\GA\\Dataset\\Data\\train-data-small.dat'
    file = 'C:\\Users\\HomeTheater\\Desktop\\GA\\Dataset\\Data\\train-data.dat'
    glob_words, glob_sent_num, glob_word_num = read_file(file)
    
    doc_tfs, total_doc_terms, num_of_docs, idf_denominator = get_tf(glob_words)
    
    tfs, idfs = get_tfs_idfs(doc_tfs, total_doc_terms, num_of_docs, idf_denominator)
    
    #plt.plot(range(0, len(doc_tfs[0])), doc_tfs[0].values())
    m_idfs = sorted(idfs.values(), reverse=True)
    plt.plot(range(0, len(m_idfs)), m_idfs)
    plt.xlabel('position of term in dictionary')
    plt.ylabel('tf-idf metric')
    plt.title('TF-IDF')
    plt.show()
    
    m_tfs = sorted(tfs.values(), reverse=True)
    plt.plot(range(0, len(m_tfs)), m_tfs)
    #plt.plot(range(0, len(idfs)), idfs)
    plt.xlabel('position of term in dictionary')
    plt.ylabel('term frequency')
    plt.title('TF')
    plt.show()
    
    return idfs