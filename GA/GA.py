# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:57:31 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
import bitarray.util
import random
import math

from matplotlib import pyplot as plt

#import load_raw_data, 
import load_data
from random_generator import Get_Init_Pop

# Constants
POP_SIZE = 20
BIT_NUM = 8520 # number of words
ONES_THRESH = 1000 # lower threshold for number of words
ONES_UPPER_THRESH = 4500 # upper threshold for number of words

PC = 0.6 # propability of crossover
PM = 0.01 # propability of mutation

# Termination criteria
MAX_IT = 100 # maximum number of generations
TOL = 0.01 # tolerance

"""
def GetBitArray():
    s = ''

    #random.seed(42)

    for i in range(0, BIT_NUM):
        t = random.randint(0, 1)
        s = s + str(t)

    b = bitarray.bitarray(s)
    print(b)
    
    return b

def GenerateInitPop():
    # each bitarray is a chromosome
    # each bit in every chromosome is a gene
    # sample: holds all the chromosomes for the intialized generation
    sample = []
    for i in range(0, POP_SIZE):
        flag = True
        while(flag):
            #temp = random.randint(0, 8519)
            #a = bitarray(8520)
            #a.setall(0)
            #a = a | temp
            #temp = bitarray.util.urandom(BIT_NUM)
            temp = Get_Init_Pop()
            #print(a)
            for t in temp:
                if t not in sample:
                    #diadikasia epidiorthosis ii)
                    #temp = bitarray.util.ba2int(temp, False)
                    #cnt = bin(t).count("1")
                    cnt = t.count("1")
                    #print("# of 1's is: " + str(cnt))
                    if cnt >= ONES_THRESH:
                        #print(temp)
                        flag = False
                        sample.append(t)
    return sample
"""   
def get_tf(words): 
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
            
            doc_tfs.append(doc_dict)
            total_doc_terms.append(cnt)
            
            doc_set = set(doc_dict)
            for t in doc_set:
                if t not in idf_denominator:
                    idf_denominator[t] = 1
                else:
                    idf_denominator[t] = idf_denominator[t] + 1

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
            idf_temp = math.log(num_of_docs/idf_denominator[i])
            if term not in term_dict:
                term_dict[term] = 0
            term_dict[term] = term_dict[term] + 1
            if term not in idfs:
                idfs[term] = 0
            idfs[term] = tf_temp * idf_temp + idfs[term]
            
    # get the means
    mean_idfs = dict()
    for k,v in term_dict.items():
        mean_idfs[k] = idfs[k] / term_dict[k]        
    
    # i think returning just idfs is enough (?)
    return term_dict, idfs, mean_idfs

def PrepareFitnessData():
    #data = load_raw_data.preprocess_vecs()
    file = 'C:\\Users\\HomeTheater\\Desktop\\Dataset\\Data\\train-data.dat'
    glob_words, glob_sent_num, glob_word_num = load_data.read_file(file)
    
    doc_tfs, total_doc_terms, num_of_docs, idf_denominator = get_tf(glob_words)
    
    tfs, idfs, mean_idfs = get_tfs_idfs(doc_tfs, total_doc_terms, num_of_docs, idf_denominator)
    
    #plt.plot(range(0, len(doc_tfs[0])), doc_tfs[0].values())
    m_idfs = sorted(mean_idfs, reverse=True)
    plt.plot(range(0, len(m_idfs)), m_idfs)
    #plt.plot(range(0, len(idfs)), idfs)
    plt.xlabel('position of term in dictionary')
    plt.ylabel('mean - inverse document frequency')
    plt.title('TF-IDF')
    plt.show()
    
    return mean_idfs

def Crossover(pop):
    
    new_pop = pop
    
    return new_pop

if __name__ == '__main__':
    
    print("Preprocessing fitness data...")
    mean_idfs = PrepareFitnessData()
    
    print("Initializing...")
    #pop = GenerateInitPop()
    pop = Get_Init_Pop()
    pop = [bitarray.bitarray(p) for p in pop]
    print("Generated initial population...")
    
    it_cnt = 0
    last_tol = 100
    tol = 1
    while(it_cnt < MAX_IT)and(abs(last_tol - tol) > TOL):
        scores = []
        
        #evaluate
        ones = []
        for p in pop:
            # Calculate the scores for each pop
            cnt = p.count(1)
            ones.append(cnt)
            temp = 0
            #print(cnt)
            # reject if pop has less 1's than allowed
            if cnt < ONES_THRESH:
                print("reject")
                temp = -1
            # apply penalty if has more 1's than allowed
            elif cnt > ONES_UPPER_THRESH:
                print("penalty")
                temp = mean_idfs[cnt] - min(mean_idfs)
            scores.append(temp)
                
        #select
        #calculate the total score for each pop
        #(normalization to 1.0)
        total_score = sum(scores)
        temp_scores = scores
        for i in range(0, len(ones)):
            scores[i] = scores[i] / total_score
        #print(sum(scores)) # DEBUG - should be 1.0
        
        scores = sorted(scores, reverse=True)
        #print(scores)
        
        #calculate cumulative scores
        cumulative_scores = []
        for i in range(0, len(scores)):
            x = range(0, i+1)
            cumulative_scores.append(sum(scores[0:i+1]))
        #print(cumulative_scores) # DEBUG - should amount to 1.0
        
        # for each pop check if it will cross
        to_cross = []
        new_pop = []
        for i in range(0, POP_SIZE):
            #print(i)
            cross_chance = random.random()
            if cross_chance < PC:
                to_cross.append(pop[i])
            else:
                new_pop.append(pop[i])
        #select the pairs (i, i+1) - wraps arround
        pairs = []
        for i in range(0, len(to_cross), 2):
            p1 = i
            p2 = i + 1
            if p2 == len(to_cross):
                p2 = 0
            #print(p1, p2)
            pairs.append([p1, p2])
                
        #crossover
        #new_pop = Crossover(pop)
        ch = []
        for pair in pairs:
            cross_bit = random.randint(0, BIT_NUM)
            ind0 = pair[0]
            ind1 = pair[1]
            ch1 = to_cross[ind0][:cross_bit] + to_cross[ind1][cross_bit:]
            ch2 = to_cross[ind1][:cross_bit] + to_cross[ind0][cross_bit:]
            #print("cross at bit ", cross_bit)
            
            #print(to_cross[ind0], to_cross[ind1])
            
            #print(ch1, ch2)
            #print("------------------")
            #ch.append([ch1, ch2])
            new_pop.append(ch1)
            new_pop.append(ch2)
        
        #mutate
        for i in range(0, POP_SIZE):
            #print(new_pop[i])
            for j in range(0, BIT_NUM):
                mutation_chance = random.random()
                if mutation_chance < PM:
                    new_pop[i][j] = 1 - new_pop[i][j]
                    
        
        #next generation
        pop = new_pop
        
        print("Max score: ", max(temp_scores))
        print("Total score: ", total_score)
        
        it_cnt = it_cnt + 1
        print("It #: ", it_cnt)