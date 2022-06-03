# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:24:22 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
from matplotlib import pyplot as plt
import numpy as np

#import multiprocessing as mp

from GA import Main_Loop, PrepareFitnessData
from random_generator import Init_Pop

def Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH):
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
    
    pop_size = POP_SIZE = 20
    cnt = 0
    #for pop_size in range(20, 100, 20):
    #for pc in np.arange(0, 1, 0.1):
    #    for pm in np.arange(0, 0.11, 0.01):
    pc = 0.9
    pm = 0.01
    print("pc - ", pc, " pm - ", pm)        

    print("Initializing...")
    pop = Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH)
    #pop = [bitarray(p) for p in pop]
    print("Generated initial population...")
    
    for j in range(0, 20):
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
    ONES_UPPER_THRESH = 4500 # upper threshold for number of words
    
    PC = 0.9#0.9
    PM = 0.10#0.01
    #PC = [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
    #PM = [0.00, 0.01, 0.10, 0.01, 0.01, 0.00, 0.01, 0.10, 0.01, 0.01]
    #POP_SIZE = [20, 20, 20, 20, 20, 200, 200, 200, 200, 200]
    
    # Termination criteria
    MAX_IT = 200 # maximum number of generations
    ES = 30 # Early stopping
    
    # multiprocessing stuff
    #pool = mp.Pool(mp.cpu_count())
    
    print("Preprocessing fitness data...")
    mean_idfs = PrepareFitnessData()

    maxs, total_maxs, total_means, pops, it_cnts = Loop(mean_idfs, MAX_IT, ES, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH)
        
    sets = []
    s = dict()
    for i in range(0, len(pops)):
        
        #for j in range(0, len(pops[i])):
        if pops[i][-1].count(1) not in s:
            s[pops[i][-1].count(1)] = 0
        s[pops[i][-1].count(1)] = s[pops[i][-1].count(1)] + 1
    sets.append(s)
    
    
    sets = {k: v for d in sets for k, v in d.items()}
    sort_by_value = dict(sorted(sets.items(), key=lambda item: item[1]))
    print(sort_by_value)
    
    ########################################## max eval value
    vals = []
    for i in range(0, len(maxs)):
        vals.append(sum(maxs[i])/len(maxs[i]))
    
    best_out_of_10 = []
    #for i in range(0, len(maxs), 10):
    best_maxs_ind = vals.index(max(vals))
    #best_out_of_10.append(j)
    
    index = vals.index(max(vals))
    p_index = int(index/10)
    
    for j in range(0, len(maxs)):
        #xs = [i for i in range(0,  it_cnts[best_maxs_ind])]
        #plt.plot(xs, maxs[best_maxs_ind], label='(every iter) max eval value')
        xs = [i for i in range(0,  it_cnts[j])]
        plt.plot(xs, maxs[j])#, label='(every iter) max eval value')
        print("it: ", j, " total max: ", total_maxs[j])
    plt.show()
    
    #for i in range(0, len(total_maxs[best_maxs_ind])):
    xs = [i for i in range(0,  it_cnts[best_maxs_ind])]
    plt.plot(xs, total_maxs[best_maxs_ind], label='(total) max eval value')
    
    #for i in range(0, len(maxs[best_maxs_ind])):
    xs = [i for i in range(0,  it_cnts[best_maxs_ind])]
    plt.plot(xs, total_means[best_maxs_ind], label='mean value')
    
    title = 'Best output for max eval value on iter: ' + str(best_maxs_ind)
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel('Best average of 10')
    plt.legend(loc='upper left')
    
    plt.show()
    
    ##########################################
    """
    # Total max eval value
    vals = []
    for i in range(0, len(total_maxs)):
        vals.append(sum(total_maxs[i])/len(total_maxs[i]))
    
    best_out_of_10 = []
    #for i in range(0, len(vals), 10):
    best_ind = vals.index(max(vals))
    #best_out_of_10.append(j)
    
    index = vals.index(max(vals))
    p_index = int(index/10)
    
    #for i in range(0, len(total_maxs[best_ind])):
    xs = [i for i in range(0,  it_cnts[best_ind])]

    plt.plot(xs, total_maxs[best_ind], label='it: ' + str(best_ind))
    title = 'Best Pc/Pm: ' + str(PC) + '/' + str(PM)
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel('(Total) Max eval values - Best average of 10')
    plt.legend(loc='upper right')
    plt.show()
    """
    # mean eval value
    vals = []
    for i in range(0, len(total_means)):
        vals.append(sum(total_means[i])/len(total_means[i]))
    
    best_ind = vals.index(max(vals))
    
    #index = vals.index(max(vals))
    #p_index = int(index/10)
    
    #for i in range(0, len(maxs[best_maxs_ind])):
    xs = [i for i in range(0,  it_cnts[best_ind])]
    plt.plot(xs, maxs[best_ind], label='(every iter) max eval value')
    
    #for i in range(0, len(total_maxs[best_maxs_ind])):
    xs = [i for i in range(0,  it_cnts[best_ind])]
    plt.plot(xs, total_maxs[best_ind], label='(total) max eval value')
    
    #for i in range(0, len(maxs[best_maxs_ind])):
    xs = [i for i in range(0,  it_cnts[best_ind])]
    plt.plot(xs, total_means[best_ind], label='mean value')
    
    title = 'Best Pc/Pm: ' + str(PC) + '/' + str(PM) + ' Best output for max eval value on iter: ' + str(best_ind)
    plt.title(title)
    plt.xlabel('Number of iterations')
    plt.ylabel('Best average of 10')
    plt.legend(loc='upper left')
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
