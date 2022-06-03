# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:37:27 2022

@author: Konstantinos Tsiamitros
"""

from matplotlib import pyplot as plt
import math

from tf_idf import PrepareFitnessData
from GA2 import Select, Evaluate,Two_Point_Crossover, Uniform_Crossover, Mutate, Repair, Tournament

from random_generator import Get_Exp_Pop
from normal_init_pop import Get_Normal_Pop

#POP_SIZE = 20
BIT_NUM = 8520
ONES_THRESH = 1000
ONES_UPPER_THRESH = 4500
#PC = 0.9
#PM = 0.01
PC = [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
PM = [0.00, 0.01, 0.10, 0.01, 0.01, 0.00, 0.01, 0.10, 0.01, 0.01]
POP_SIZE = [20, 20, 20, 20, 20, 200, 200, 200, 200, 200]
MAX_IT = 50

#POP_SIZE = [ 20, 200, 200, 200, 200, 200]

idfs = PrepareFitnessData()


for j in range(0, 10):
    


    for i in range(0, 10):
        pop, ones = Get_Normal_Pop(POP_SIZE[j], BIT_NUM, ONES_THRESH, 2500, 500)
        #pop, ones = Get_Exp_Pop(POP_SIZE, BIT_NUM, ONES_THRESH)
            
        #plt.hist(ones, range=(1500, 7500), bins=POP_SIZE[j])
        #plt.show()
        
        #total_maxs = []
        total_maxs_counts = [None] * MAX_IT
        
        mean_ones = []
        it_maxs = [None] * MAX_IT
        
        it = 0  # iterations without improvement
        it_act = 0  #actual iterations
        last_max_score = -1
        while it < 20 and it_act < MAX_IT:
        #while it_act < MAX_IT:
            scores, ones = Evaluate(idfs, pop, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH)
            
            #fix? nope
            #scores = [abs(s) for s in scores]
            
            
            #print(scores)
            #chosen, scores = Select(ones, scores, pop, POP_SIZE[j])
            chosen = Tournament(scores, pop, POP_SIZE[j])
            
            mx = max(scores)
            
            #new_pop = Uniform_Crossover(scores, PC[j], BIT_NUM, chosen, POP_SIZE[j], ONES_THRESH)
            new_pop = Two_Point_Crossover(scores, PC[j], BIT_NUM, chosen, POP_SIZE[j], ONES_THRESH)
            
            # Repair
            new_pop = Repair(pop, new_pop, scores, ONES_THRESH)
            #new_pop = Repair(new_pop, scores, ONES_THRESH)
            
            new_pop = Mutate(new_pop, PM[j])
            #new_pop = Mutate(new_pop, scores, BIT_NUM, PM[j], POP_SIZE[j], ONES_THRESH)
            
            # Repair
            new_pop = Repair(pop, new_pop, scores, ONES_THRESH)
            #new_pop = Repair(new_pop, scores, ONES_THRESH)
            
            cnts = 0
            ones = []
            #if len(new_pop) == 0:
            #    raise Exception("EXCEPTION len(pop): ", len(pop))
            
            for x in new_pop:
                #t = bitarray(int(x))
                ones.append(x.count(1))
                #print(ones[-1])
                cnts = cnts + x.count(1)
            #print(cnts/POP_SIZE)
            #print(len(pop))
            #plt.hist(ones, range=(0, 9000), bins=POP_SIZE[j])
            #plt.show()
            
            
            
            
            # Termination Criteria
            #print("last max:", last_max_score, " max: ", max(temp_scores))
            #"""
            if last_max_score < mx:
                last_max_score = mx
                it = 0
                #if max(ones) == min(ones):
                #    break
            else:
                print("Max score diminished")
                it = it + 3
                #break
            #"""
            """
            if last_max_score >= 0:
                if (mx - last_max_score) < 0.001:
                    print("max score stopped improving.")
                    #break
                    it = it + 3
            #"""
            #if (mx - last_max_score) > 3.0:
               # break
            
            #metrics
            if it_maxs[it_act] == None:
                it_maxs[it_act] = mx
                total_maxs_counts[it_act] = 1
            else:
                it_maxs[it_act] = it_maxs[it_act] + mx
                total_maxs_counts[it_act] = total_maxs_counts[it_act] + 1
            """
            #for i in range(it, len(it_maxs)):
            if it_act < len(total_maxs):
                total_maxs[it_act] = total_maxs[it_act] + it_maxs[it_act]
                total_maxs_counts[it_act] = total_maxs_counts[it_act] + 1
            else:
                total_maxs.append(it_maxs[it_act])
                total_maxs_counts.append(1)
            """    
            pop = new_pop
            
            mean = sum(ones)/len(ones)
            print("mean ones: ", mean)
            mean_ones.append(mean)
            
            print("min score: ", min(scores))
            print("max score: ", mx)
            
            
            # check if pop has converged
            #if max(ones) == min(ones):
            #   break
            
            last_max_score = mx
            #print("it: ", it)
            #print("it_act: ", it_act)
            it = it + 1
            it_act = it_act + 1
                    
    for i in range(0, len(it_maxs)):
        if it_maxs[i] is not None:
            #total_maxs[i] = total_maxs[i] / total_maxs_counts[i]
            it_maxs[i] = it_maxs[i] / total_maxs_counts[i]
        
    plt.plot(it_maxs)
    #plt.show()
    
    plt.plot(mean_ones)
    plt.show()
