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
    #PC = [x for x in np.arange(0, 1, 0.1)]
    #PM = [x for x in np.arange(0, 0.11, 0.01)]
    #POP_SIZE = [40 for i in range(0, 10)]
    
    maxs = []
    max_xs = []
    pops = []
    means = []
    last_pops = []
    
    pms = []
    pcs = []
    
    it = 0

    #for j in range(0, 10):
        #pc = PC[j] 
        #pm = PM[j]
    pop_size = 20#POP_SIZE[j]
    total_its = []
    for pc in np.arange(0, 1, 0.3):
        for pm in np.arange(0, 0.11, 0.03):
            #print(pop_size)
            #print("Initializing...")
            pop = Get_Init_Pop(pop_size)
            pop = [bitarray(p) for p in pop]
            print("Generated initial population...")
            #for pm in PM: 
            max_scores = 0
            last_max_scores = 0
            total_mean_scores = 0
            temp_its = []
            for i in range(0, 10):
                m, lm, tm, lp, its = Main_Loop(pop, mean_idfs, pc, pm, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, pop_size, ES)
                max_scores = max_scores + sum(m)/len(m)
                last_max_scores = last_max_scores + sum(lm)/len(lm)
                total_mean_scores = total_mean_scores + sum(tm)/len(tm)
                
                temp_its.append(its)
                
                last_pops.append(lp)
                #if not maxs:
                maxs.append(last_max_scores)
                max_xs.append(it)
                pops.append(pop_size)
                means.append(total_mean_scores)
                #else:
                #    if last_max_scores/10 >= max(maxs):
                #        maxs.append(last_max_scores/10)
                #        max_xs.append(it)
                #        pops.append(pop_size)
                #        means.append(total_mean_scores/10)
                #        last_pops.append(last_pop)
                pms.append(pm)
                pcs.append(pc)
        total_its.append(temp_its)
        it = it + 1
        print("iteration ", it)
    maxs = [ sum(maxs[i:i+10])/10 for i in range(0, len(maxs)-1, 10)]
    means = [ sum(means[i:i+10])/10 for i in range(0, len(means)-1, 10)]
    return maxs, max_xs, pops, means, last_pops, pms, pcs

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

    maxs, max_xs, pops, means, last_pops, pms, pcs = Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH)
        
    #pool.close()
    
    xs = [i for i in range(0, len(maxs))]
    # 1
    mn_ind = maxs.index(min(maxs))
    mx_ind = maxs.index(max(maxs))
    #for i in range(0, len(max_xs)-1, 10):
        #for j in range(i, i+10):
    plt.plot(xs, maxs)
    plt.xlabel('Samples')
    plt.ylabel('Avg of 10 - Max eval value')
    title = 'Min Pc/Pm: ' + str(pcs[mn_ind*10]) + '/' + str(pms[mn_ind*10]) + ' Max Pc/Pm: ' + str(pcs[mx_ind*10]) + '/' + str(pms[mx_ind*10])
    plt.title(title)
    plt.show()
       
    # 2
    mn_ind = means.index(min(means))
    mx_ind = means.index(max(means))
    #for i in range(0, len(max_xs)-1, 10):
    #for j in range(i, i+10):
    plt.plot(xs, means)
    plt.xlabel('Samples')
    plt.ylabel('Avg of 10 - Mean eval value')
    title = 'Min it: ' + str(mn_ind) + ' Max it: ' + str(mx_ind)
    plt.title(title)
    plt.show()
    
    # 3
    #for i in range(0, len(max_xs)-1, 10):
    #for j in range(i, i+10):
    plt.plot(xs, pops)
    plt.xlabel('Samples')
    plt.ylabel('Population')
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
