# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:42:40 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
from matplotlib import pyplot as plt
import numpy as np

#import multiprocessing as mp

from GA import Main_Loop, PrepareFitnessData
from random_generator import Get_Init_Pop

def Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH, PC, PM, POP_SIZE):
    #pop_size = args[0]
    #mean_idfs = args[1]
    
    # best is iteration 4
    #PC = [0.6, 0.6,]# 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
    #PM = [0.00, 0.01,]# 0.10, 0.01, 0.01, 0.00, 0.01, 0.10, 0.01, 0.01]
    #POP_SIZE = [20, 20,]# 20, 20, 20, 200, 200, 200, 200, 200]
    
    # variance of max eval value is less than before
    #PC = [x for x in np.arange(0, 1, 0.1)]
    #PM = [x for x in np.arange(0, 0.11, 0.01)]
    #POP_SIZE = [20 for i in range(0, 10)]
    
    # variance of mean eval value is less than before
    #PC = [x for x in np.arange(0, 1, 0.1)]
    #PM = [x for x in np.arange(0, 0.11, 0.01)]
    #POP_SIZE = [40 for i in range(0, 10)]

    maxs = []
    total_maxs = []
    total_means = []
    pops = []
    it_cnts = []    

    for i in range(0, len(PC)):
        pc = PC[i]
        pm = PM[i]
        pop_size = POP_SIZE[i]
        
        print("Initializing...")
        pop = Get_Init_Pop(pop_size)
        pop = [bitarray(p) for p in pop]
        print("Generated initial population...")
        
        for j in range(0, 10):
            m, lm, tm, lp, its = Main_Loop(pop, mean_idfs, pc, pm, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, pop_size, ES)
            maxs.append(m)
            total_maxs.append(lm)
            total_means.append(tm)
            pops.append(lp)
            it_cnts.append(its)
            
    return maxs, total_maxs, total_means, pops, it_cnts

if __name__ == '__main__':

    BIT_NUM = 8520 # number of words
    ONES_THRESH = 1000 # lower threshold for number of words
    ONES_UPPER_THRESH = 3500 # upper threshold for number of words
    
    PC = [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
    PM = [0.00, 0.01, 0.10, 0.01, 0.01, 0.00, 0.01, 0.10, 0.01, 0.01]
    POP_SIZE = [20, 20, 20, 20, 20, 200, 200, 200, 200, 200]
    
    # Termination criteria
    MAX_IT = 800 # maximum number of generations
    ES = 10 # Early stopping
    
    # multiprocessing stuff
    #pool = mp.Pool(mp.cpu_count())
    
    print("Preprocessing fitness data...")
    mean_idfs = PrepareFitnessData()

    maxs, total_maxs, total_means, pops, it_cnts = Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH, PC, PM, POP_SIZE)
        
    # max eval value
    vals = []
    for i in range(0, len(maxs)):
        vals.append(sum(maxs[i])/len(maxs))
    
    best_out_of_10 = []
    for i in range(0, len(maxs), 10):
        j = maxs.index(max(maxs[i:i+10]))
        best_out_of_10.append(j)
    
    index = vals.index(max(vals))
    p_index = int(index/10)
    
    for i in range(0, len(best_out_of_10)):
        xs = [i for i in range(0,  it_cnts[best_out_of_10[i]]+1)]
        
        plt.plot(xs, maxs[best_out_of_10[i]], label='it: ' + str(best_out_of_10[i]))
    title = 'Best Pc/Pm: ' + str(PC[p_index]) + '/' + str(PM[p_index])
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel('Max eval values - Best average of 10')
    plt.legend(loc='upper right')
    plt.show()
    
    # Total max eval value
    vals = []
    for i in range(0, len(total_maxs)):
        vals.append(sum(total_maxs[i])/len(total_maxs))
    
    best_out_of_10 = []
    for i in range(0, len(total_maxs), 10):
        j = total_maxs.index(max(total_maxs[i:i+10]))
        best_out_of_10.append(j)
    
    index = vals.index(max(vals))
    p_index = int(index/10)
    
    for i in range(0, len(best_out_of_10)):
        xs = [i for i in range(0,  it_cnts[best_out_of_10[i]]+1)]

        plt.plot(xs, total_maxs[best_out_of_10[i]], label='it: ' + str(best_out_of_10[i]))
    title = 'Best Pc/Pm: ' + str(PC[p_index]) + '/' + str(PM[p_index])
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel('(Total) Max eval values - Best average of 10')
    plt.legend(loc='upper right')
    plt.show()

    """
    index = vals.index(max(vals))
    p_index = int(index/10)
    
    for i in range(0, len(maxs)):
        #xs = [i for i in range(0, len(maxs[index]))]
        xs = [i for i in range(0,  it_cnts[i])]
        
        plt.plot(xs, maxs[i], label='it: ' + str(i))
    title = 'Best Pc/Pm: ' + str(PC[p_index]) + '/' + str(PM[p_index])
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel('Max eval values - Best average of 10')
    plt.legend(loc='upper left')
    plt.show()
    
    #-----------------
    
    vals = []
    for i in range(0, len(total_maxs)):
        vals.append(sum(total_maxs[i])/len(total_maxs))
    
    xs = [i for i in range(0, len(total_maxs))]
    #for i in range(0, len(maxs)):
    plt.plot(xs, vals)
    
    plt.title('Max (total) eval values - Avg of 10')
    plt.show()
    
    
    
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
