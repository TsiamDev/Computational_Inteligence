# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:57:31 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
import bitarray.util
import random
import math
#import numpy as np

from matplotlib import pyplot as plt

#import load_raw_data, 
import load_data
from random_generator import get_rand_bit_array, Get_Init_Pop

#from constants import *

from time import perf_counter
import multiprocessing as mp

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
    """
    # get the means
    mean_idfs = dict()
    for k,v in term_dict.items():
        mean_idfs[k] = idfs[k] / term_dict[k]        
    """
    # i think returning just idfs is enough (?)
    return term_dict, idfs#, mean_idfs

def PrepareFitnessData():
    #data = load_raw_data.preprocess_vecs()
    #file = 'C:\\Users\\HomeTheater\\Desktop\\GA\\Dataset\\Data\\train-data-small.dat'
    file = 'C:\\Users\\HomeTheater\\Desktop\\GA\\Dataset\\Data\\train-data.dat'
    glob_words, glob_sent_num, glob_word_num = load_data.read_file(file)
    
    doc_tfs, total_doc_terms, num_of_docs, idf_denominator = get_tf(glob_words)
    
    #tfs, idfs, mean_idfs = get_tfs_idfs(doc_tfs, total_doc_terms, num_of_docs, idf_denominator)
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

def Pre_Cross(chosen, PC, POP_SIZE):
    new_pop = [None] * POP_SIZE
    
    #determine which individuals will cross
    to_cross = []
    for i in range(0, len(chosen)):
        #ch = np.arange(0, 1)
        #ch = np.random.normal(0.0, 1.0)
        ch = random.random()
        if ch < PC:
            to_cross.append(i)
        else:
            # individual won't cross, so just asdd it to the new_pop
            new_pop[i] = chosen[i]
    
    #determine the pairs
    # cross pairs with similar score
    pairs = []
    for i in range(0, len(to_cross), 2):
        p1 = to_cross[i]
        if i+1 == len(to_cross):
            p2 = to_cross[0]
        else:
            p2 = to_cross[i + 1]

        #print(p1, p2)
        pairs.append([p1, p2])
        
    return pairs, new_pop

def One_Point_Crossover(PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, PC, POP_SIZE)

    for pair in pairs:
        cross_bit = random.randint(0, BIT_NUM)
        ind0 = pair[0]
        ind1 = pair[1]
        ch1 = chosen[ind0][:cross_bit] + chosen[ind1][cross_bit:]
        ch2 = chosen[ind1][:cross_bit] + chosen[ind0][cross_bit:]
        #print("cross at bit ", cross_bit)
        
        #print(to_cross[ind0], to_cross[ind1])
        
        #print(ch1, ch2)
        #print("------------------")
        #ch.append([ch1, ch2])
        
        if ch1.count(1) < ONES_THRESH:
            ch1 = bitarray.bitarray(BIT_NUM)
        
        if ch2.count(1) < ONES_THRESH:
            ch2 = bitarray.bitarray(BIT_NUM)
        
        new_pop[ind0] = ch1
        new_pop[ind1] = ch2

    return new_pop
        

def Two_Point_Crossover(PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, PC, POP_SIZE)
    
    for pair in pairs:
        cross_bit = random.randint(0, int(BIT_NUM/3))
        ind0 = pair[0]
        ind1 = pair[1]
        ch1 = chosen[ind0][:cross_bit]
        ch2 = chosen[ind1][:cross_bit]
        
        cross_bit2 = random.randint(cross_bit+1, int(2*BIT_NUM/3))
        ch1 = ch1 + chosen[ind1][cross_bit:cross_bit2] + chosen[ind0][cross_bit2:]
        ch2 = ch2 + chosen[ind0][cross_bit:cross_bit2] + chosen[ind1][cross_bit2:]
        
        if ch1.count(1) < ONES_THRESH:
            ch1 = bitarray.bitarray(BIT_NUM)
        
        if ch2.count(1) < ONES_THRESH:
            ch2 = bitarray.bitarray(BIT_NUM)
        
        new_pop[ind0] = ch1
        new_pop[ind1] = ch2

    return new_pop

def Uniform_Crossover(PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, PC, POP_SIZE)
    
    for pair in pairs:
        #cross_bit = random.randint(0, 1)
        cross_bits = bitarray.bitarray(BIT_NUM)
        ind0 = pair[0]
        ind1 = pair[1]
        children = []
        for i in range(0, 2):
            ch = bitarray.bitarray()
            for j in range(0, BIT_NUM):
                if cross_bits[j] == 1:
                    ch = ch + str(chosen[ind0][j])
                else:
                    ch = ch + str(chosen[ind1][j])
            children.append(ch)
        
        if children[0].count(1) < ONES_THRESH:
            children[0] = bitarray.bitarray(BIT_NUM)
        
        if children[1].count(1) < ONES_THRESH:
            children[1] = bitarray.bitarray(BIT_NUM)
        
        new_pop[ind0] = children[0]
        new_pop[ind1] = children[1]
    
    return new_pop

def Evaluate(mean_idfs, pop, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH):
    scores = []
    ones = []
    for i in range(0, len(pop)):
        ones.append(pop[i].count(1))
        score = 0
        digit_cnt = 0
        #mean_cnt = 0
        for k in range(0, len(pop[i])):
            # add the score for each selected word
            if pop[i][k] == 1:
                score = score + mean_idfs[k]
        scores.append(score)
    
    #print(scores)
    mx = scores.index(max(scores))
    #print(mx)
    mx_score = scores[mx]
    #print(mx)
    for i in range(0, len(scores)):
        scores[i] = scores[i] / 1000
    """
    min_idfs = min(scores)
    #print(min_idfs)
    #max_idfs = max(scores)
    #print(max_idfs)
    
    #find max score
    mx = scores.index(max(scores))
    
    
    for i in range(0, len(pop)):
        if pop[i].count(1) >= ONES_UPPER_THRESH:
            scores[i] = scores[i] - min_idfs  
            if scores[i] <= 0:
                scores[i] = 0.0000000001
        elif pop[i].count(1) < ONES_THRESH:
            print("before ", pop[i].count(1))
            pop[i] = pop[mx]
            scores[i] = scores[mx]
            print("after ", pop[i])
    """
    return scores, ones

def Choose(cumulative_scores):
    #cc = np.arange(0, 1) # cross chance
    #cc = np.random.normal(0.0, 1.0)
    cc = random.random()
    """
    if cc <= cumulative_scores[0]:
        return 0
    else:    
        for i in range(1, len(cumulative_scores)-1):
            if cc > cumulative_scores[i] and cc <= cumulative_scores[i+1]:
                return i
    """
    if cc <= cumulative_scores[0]:
        return 0
    else:
        for i in range(0, len(cumulative_scores)-1):
            if cc > cumulative_scores[i] and cc <= cumulative_scores[i+1]:
                return i+1
        
    # if you got here something went wrong
    return None

def Select(ones, scores, total_score, pop, POP_SIZE):
    for i in range(0, len(scores)):
        scores[i] = scores[i] / total_score
    #print(sum(scores)) # DEBUG - should be 1.0
    #print(scores)
    #print(len(scores))
    
    # rank based roullete wheel
    # sort scores (while keeping its' relation to pop)
    #scores, pop = zip(*sorted(zip(scores, pop)))
    #scores = list(scores)
    #pop = list(pop)
    
    #print(scores)
    
    #calculate cumulative scores
    cumulative_scores = []
    for i in range(0, len(scores)):
        #x = range(0, i+1)
        cumulative_scores.append(sum(scores[0:i+1]))
    #print(cumulative_scores) # DEBUG - should amount to 1.0
    #print("-----")
    # for each pop check if it will cross
    chosen = []
    for i in range(0, POP_SIZE):
        ch = Choose(cumulative_scores)
        chosen.append(pop[ch])

    return chosen

def Mutate(new_pop, scores, BIT_NUM, PM, POP_SIZE, ONES_THRESH):
    m = 0
    mx = scores.index(max(scores))
    #print(new_pop)
    for i in range(0, POP_SIZE):
        #print(new_pop[i])
        # Elitism
        if i == mx:
            continue
        else:
            choice = bitarray.bitarray(BIT_NUM)
            new_pop[i] = new_pop[i] ^ choice
        
            #print(choice)
            #for j in range(0, BIT_NUM):
            #    ch = choice.pop(0)
            #    if ch == 0:
            #        new_pop[i].invert(j)
            #        m = m + 1
    
    return new_pop

def Main_Loop(pop, idfs, PC, PM, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, POP_SIZE, ES):
    it_cnt = 0
    last_tol = 100
    tol = 1
    last_max_score = -1
    max_mean_score = -1
    es_cnt = 0
    max_scores = []
    last_max_scores = []
    total_mean_scores = []
    mean_scores = []
    while(it_cnt < MAX_IT):#and(es_cnt < ES):#and(abs(last_tol - tol) > TOL):
        it_cnt = it_cnt + 1
        
        #t0 = perf_counter()
        #evaluate
        scores, ones = Evaluate(idfs, pop, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH)
        #t1 = perf_counter()
        
        #t2 = perf_counter()
        #select
        #calculate the total score for each pop
        #(normalization to 1.0)
        total_score = sum(scores)
        temp_scores = scores
        chosen = Select(ones, scores, total_score, pop, POP_SIZE)
        #t3 = perf_counter()
        
        #t4 = perf_counter()
        #crossover
        # 1 point crossover
        #new_pop = One_Point_Crossover(PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH)
        # 2 point crossover  
        new_pop = Two_Point_Crossover(PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH)
        # Uniform crossover 
        #new_pop = Uniform_Crossover(PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH)
        #t5 = perf_counter()
        
        #t6 = perf_counter()
        new_pop = Mutate(new_pop, scores, BIT_NUM, PM, POP_SIZE, ONES_THRESH)      
        #t7 = perf_counter()
        
        
        
        #print("times: ", t1-t0, t3-t2, t5-t4, t7-t6)
        #next generation
        pop = new_pop
        
        #print("Max score: ", max(temp_scores))
        total_mean_score = total_score#/POP_SIZE
        #print("Total mean score: ", total_mean_score)
        total_mean_scores.append(total_mean_score)
        
        # Termination Criteria
        #print("last max:", last_max_score, " max: ", max(temp_scores))
        if last_max_score <= max(temp_scores):
            last_max_score = max(temp_scores)
            es_cnt = 0
        else:
            #print("Max score diminished")
            es_cnt = es_cnt + 1
            #break
        
        #"""
        """
        if max_mean_score <= total_mean_score:
            max_mean_score = total_mean_score
            es_cnt = 0
        else:
            es_cnt = es_cnt + 4
        #"""
             
        # Tolerance
        if it_cnt > 1:
            if last_max_score - max(temp_scores) < 0.01:
            #if abs(max(temp_scores) - max_scores[len(max_scores)-1]) < 0.01:
                #max_scores.append(max(temp_scores))
                #last_max_scores.append(last_max_score)
                es_cnt = es_cnt + 4
                #break
            

        #print(es_cnt)
        #"""
        
        #print("Ones avg: ", sum(ones)/len(ones))
        
        max_scores.append(max(temp_scores))
        last_max_scores.append(last_max_score)
        
        # Early Stopping
        if es_cnt >= ES:
            break
        
        #print("It #: ", it_cnt)
    print("Took ",  it_cnt, " iterations")
    mx = max(total_mean_scores)
    total_mean_scores = [x/mx for x in total_mean_scores]
    return max_scores, last_max_scores, total_mean_scores, pop, it_cnt

"""
if __name__ == '__main__':
    
    # Constants
    POP_SIZE = 50
    BIT_NUM = 8520 # number of words
    ONES_THRESH = 1000 # lower threshold for number of words
    ONES_UPPER_THRESH = 3500 # upper threshold for number of words

    PC = 0.6 # propability of crossover
    PM = 0.1 # propability of mutation

    # Termination criteria
    MAX_IT = 800 # maximum number of generations

    # TO-DO
    TOL = 0.01 # tolerance
    ES = 10 # early stopping
    
    print("Preprocessing fitness data...")
    mean_idfs = PrepareFitnessData()
    
    print("Initializing...")
    #pop = GenerateInitPop()
    pop = Get_Init_Pop(POP_SIZE)
    pop = [bitarray.bitarray(p) for p in pop]
    print("Generated initial population...")
    
    # multiprocessing stuff
    #pool = mp.Pool(mp.cpu_count())
    
    max_scores, last_max_scores, total_mean_scores = Main_Loop(pop, PC, PM, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, POP_SIZE, mean_idfs)
        
    #pool.close()
        
    plt.plot(range(0, len(max_scores)), max_scores)
    plt.ylabel('Max Scores')
    plt.xlabel('# of iteration')
    plt.show()
    
    plt.plot(range(0, len(last_max_scores)), last_max_scores)
    plt.ylabel('Last Max Scores')
    plt.xlabel('# of iteration')
    plt.show()
    
    plt.plot(range(0, len(total_mean_scores)), total_mean_scores)
    plt.ylabel('Total Mean Scores')
    plt.xlabel('# of iteration')
    plt.show()
"""