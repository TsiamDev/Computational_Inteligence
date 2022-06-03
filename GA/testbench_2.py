# -*- coding: utf-8 -*-
"""
Created on Sun May 15 13:54:12 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
from matplotlib import pyplot as plt
import numpy as np

#import multiprocessing as mp

from GA import Main_Loop, PrepareFitnessData
from random_generator import Get_Init_Pop

def Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH):
    #pop_size = args[0]
    #mean_idfs = args[1]
    
    # best is iteration 4
    #PC = [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
    #PM = [0.00, 0.01, 0.10, 0.01, 0.01, 0.00, 0.01, 0.10, 0.01, 0.01]
    #POP_SIZE = [20, 20, 20, 20, 20, 200, 200, 200, 200, 200]
    
    # variance of max eval value is less than before
    #PC = [x for x in np.arange(0, 1, 0.1)]
    #PM = [x for x in np.arange(0, 0.11, 0.01)]
    #POP_SIZE = [20 for i in range(0, 10)]
    
    # variance of mean eval value is less than before
    PC = [x for x in np.arange(0, 1, 0.1)]
    PM = [x for x in np.arange(0, 0.11, 0.01)]
    POP_SIZE = [40 for i in range(0, 10)]
    
    maxs = []
    max_xs = []
    pops = []
    means = []
    last_pops = []
    it = 0

    for j in range(0, 10):
        pc = PC[j] 
        pm = PM[j]
        pop_size = POP_SIZE[j]
        print(pop_size)
        print("Initializing...")
        pop = Get_Init_Pop(pop_size)
        pop = [bitarray(p) for p in pop]
        print("Generated initial population...")
        #for pm in PM: 
        max_scores = 0
        last_max_scores = 0
        total_mean_scores = 0
        for i in range(0, 10):
            m, lm, tm, lp = Main_Loop(pop, mean_idfs, pc, pm, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, pop_size, ES)
            max_scores = max_scores + sum(m)/len(m)
            last_max_scores = last_max_scores + sum(lm)/len(lm)
            total_mean_scores = total_mean_scores + sum(tm)/len(tm)
            last_pops.append(lp)
            #if not maxs:
            maxs.append(last_max_scores/10)
            max_xs.append(it)
            pops.append(pop_size)
            means.append(total_mean_scores/10)
            #else:
            #    if last_max_scores/10 >= max(maxs):
            #        maxs.append(last_max_scores/10)
            #        max_xs.append(it)
            #        pops.append(pop_size)
            #        means.append(total_mean_scores/10)
            #        last_pops.append(last_pop)
                
        it = it + 1
        print("iteration ", it)
    return maxs, max_xs, pops, means, last_pops

if __name__ == '__main__':

    BIT_NUM = 8520 # number of words
    ONES_THRESH = 1000 # lower threshold for number of words
    ONES_UPPER_THRESH = 3500 # upper threshold for number of words
    
    #PC = 0.6 # propability of crossover
    #PM = 0.1 # propability of mutation
    
    # Termination criteria
    MAX_IT = 800 # maximum number of generations
    ES = 10 # Early stopping
    
    # multiprocessing stuff
    #pool = mp.Pool(mp.cpu_count())
    
    print("Preprocessing fitness data...")
    mean_idfs = PrepareFitnessData()
    
    #maxs, max_xs, pops = [pool.apply(Loop, args=([pop_size, mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH])) for pop_size in range(40, 200, 40)]
    # upper bound is not inclusive!!

    maxs, max_xs, pops, means, last_pops = Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH)
        
    #pool.close()
    
    xs = [i for i in range(0, 10)]
    # 1
    for i in range(0, len(max_xs), 10):
        #for j in range(i, i+10):
        plt.plot(xs, maxs[i:i+10], label='it: ' + str(int(i/10)))
    plt.xlabel('Population')
    plt.ylabel('Avg of 10 - Max eval value')
    plt.legend(loc='upper left')
    plt.show()
       
    # 2
    for i in range(0, len(max_xs), 10):
        #for j in range(i, i+10):
        plt.plot(xs, means[i:i+10], label='it: ' + str(int(i/10)))
    plt.xlabel('Population')
    plt.ylabel('Avg of 10 - Mean eval value')
    plt.legend(loc='upper left')
    plt.show()
    
    # 3
    for i in range(0, len(max_xs), 10):
        #for j in range(i, i+10):
        plt.plot(xs, pops[i:i+10], label='it: ' + str(int(i/10)))
    plt.xlabel('# of iteration')
    plt.ylabel('Population')
    plt.legend(loc='upper left')
    plt.show()
    
    # use this to print which ever group of ones you need
    #print([x.count(1) for x in last_pops[4][3]])
    
    """       
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
