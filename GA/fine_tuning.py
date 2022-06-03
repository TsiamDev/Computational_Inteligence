# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:44:19 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
from matplotlib import pyplot as plt
import numpy as np

#import multiprocessing as mp

from GA import Main_Loop, PrepareFitnessData
from random_generator import Get_Init_Pop

def Loop(pop_size, mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH):
    #pop_size = args[0]
    #mean_idfs = args[1]
    
    maxs = []
    max_xs = []
    pops = []
    means = []
    last_pops = []
    it = 0
    print("Initializing...")
    pop = Get_Init_Pop(pop_size)
    pop = [bitarray(p) for p in pop]
    print("Generated initial population...")
    for pc in np.arange(0, 1, 0.1):
        for pm in np.arange(0, 0.11, 0.01): 
            max_scores = 0
            last_max_scores = 0
            total_mean_scores = 0
            last_pop = []
            for i in range(0, 10):
                m, lm, tm, lp = Main_Loop(pop, mean_idfs, pc, pm, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, pop_size, ES)
                max_scores = max_scores + sum(m)/len(m)
                last_max_scores = last_max_scores + sum(lm)/len(lm)
                total_mean_scores = total_mean_scores + sum(tm)/len(tm)
                last_pop.append(lp)
            if not maxs:
                maxs.append(last_max_scores/10)
                max_xs.append(it)
                pops.append(pop_size)
                means.append(total_mean_scores/10)
                last_pops.append(last_pop)
            else:
                if last_max_scores/10 >= max(maxs):
                    maxs.append(last_max_scores/10)
                    max_xs.append(it)
                    pops.append(pop_size)
                    means.append(total_mean_scores/10)
                    last_pops.append(last_pop)
                    
            it = it + 1
            print("iteration ", it)
    return maxs, max_xs, pops, means, last_pops

if __name__ == '__main__':
    #POP_SIZE = 80
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
    
    
    maxs = []
    max_xs = []
    pops = []
    means = []
    last_pops = []
    #maxs, max_xs, pops = [pool.apply(Loop, args=([pop_size, mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH])) for pop_size in range(40, 200, 40)]
    # upper bound is not inclusive!!
    for pop_size in range(40, 240, 40):
        t_maxs, t_max_xs, t_pops, t_means, t_last_pops = Loop(pop_size, mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH)
        maxs.append(t_maxs)
        max_xs.append(t_max_xs)
        pops.append(t_pops)
        means.append(t_means)
        last_pops.append(t_last_pops)
        
    #pool.close()
    
    # 1
    for i in range(0, len(max_xs)):
        plt.plot(pops[i], maxs[i])
    plt.xlabel('Population')
    plt.ylabel('Max eval value')
    plt.show()
       
    # 2
    for i in range(0, len(max_xs)):
        plt.plot(pops[i], means[i])
    plt.xlabel('Population')
    plt.ylabel('Mean eval value')
    plt.show()
    
    # 3
    for i in range(0, len(max_xs)): 
        plt.plot(pops[i], max_xs[i])
    plt.xlabel('Population')
    plt.ylabel('# of iterations')
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