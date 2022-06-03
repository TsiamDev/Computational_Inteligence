# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:47:56 2022

@author: Konstantinos Tsiamitros
"""

from bitarray import bitarray
import bitarray.util
import random
from numpy.random import randint
import math


#from time import perf_counter


def Pre_Cross(chosen, scores, PC, POP_SIZE):
    new_pop = [None] * POP_SIZE
    
    mx = max(scores)
    #mx_ind = scores.index(mx)
    mx_indices = [i for i, e in enumerate(scores) if e == mx]
    
    #determine which individuals will cross
    to_cross = []
    for i in range(0, len(chosen)):
        #ch = np.arange(0, 1)
        #ch = np.random.normal(0.0, 1.0)
        # elitism - the best individual crosses over unchanged
        if i in mx_indices:
            new_pop[i] = chosen[i]
        else:
            ch = random.random()
            if ch < PC:
                to_cross.append(i)
            else:
                # individual won't cross, so just add it to the new_pop
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

def One_Point_Crossover(scores, PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, scores, PC, POP_SIZE)

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
        

def Two_Point_Crossover(scores, PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, scores, PC, POP_SIZE)
    
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

def Uniform_Crossover(scores, PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, scores, PC, POP_SIZE)
    
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
        
        new_pop[ind0] = children[0]
        new_pop[ind1] = children[1]
    
    return new_pop
"""
def Uniform_Crossover(scores, PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH):
    pairs, new_pop = Pre_Cross(chosen, scores, PC, POP_SIZE)
    
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
"""
def Evaluate(idfs, pop, BIT_NUM, ONES_THRESH, ONES_UPPER_THRESH):
    # decode the representation of each chromosome
    scores = []
    ones = []
    for i in range(0, len(pop)):
        ones.append(pop[i].count(1))
        score = 0
        for k in range(0, len(pop[i])):
            # add the score for each selected word
            if pop[i][k] == 1:
                score = score + idfs[k]
        #scores.append((score/ones[i] - 1/ones[i])**2)
        #scores.append((score/ones[i])**2)
        #scores.append( 1/(1 + (score/ones[i])) )
        
        #scores.append( (1/(1 + score) - ones[i]/BIT_NUM) )
        scores.append( (score - score/(score + ones[i]/BIT_NUM) ) )
        
        #scores.append()
        
        #scores.append(score#)/BIT_NUM)
    
    mx = max(scores)
    mn = min(scores)
    #scale scores to [0,1]
    #for i in range(0, len(scores)):
    #    scores[i] = (scores[i] - mn + 0.001)/ (mx + 0.001)
    
    # Penalty for solutions with higher than the limit
    # count of ones
    min_idfs = min(scores)
    #print("min score: ", min_idfs)
    #print("max score: ", max(scores))
    
    #penalty = (max(scores) + 0.001 - min(scores))/( max(scores) + 0.001)
    penalty = math.e**(1 - max(scores)/BIT_NUM) * (max(scores))
    #penalty = max(scores)#/BIT_NUM)
    
    #print(scores)
    #mx = scores.index(max(scores))
    #print(mx)
    #mx_score = scores[mx]
    #print(mx)
    for i in range(0, len(pop)):
    #    scores[i] = scores[i] / 1000
        if pop[i].count(1) >= ONES_UPPER_THRESH:
            #print(i)
            #print("bef", scores[i])
            scores[i] = scores[i] - penalty
            #scores[i] = scores[i] - max(scores)
            #print("aft", scores[i])
       
    return scores, ones

def Choose(cumulative_scores):
    #cc = np.arange(0, 1) # cross chance
    #cc = np.random.normal(0.0, 1.0)
    cc = random.random()

    if cc <= cumulative_scores[0]:
        return 0
    else:
        for i in range(0, len(cumulative_scores)-1):
            if cc > cumulative_scores[i] and cc <= cumulative_scores[i+1]:
                return i+1
        
    # if you got here something went wrong
    return None

# forced roullette wheel
def Select(ones, scores, pop, POP_SIZE):
    total_score = sum(scores)
    for i in range(0, len(scores)):
        scores[i] = scores[i] / total_score
    #print(sum(scores)) # DEBUG - should be 1.0
    #print(scores)
    #print(len(scores))
    
    # uncomment to use
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

    return chosen, scores

# tournament selection
# https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
def Tournament(scores, pop, POP_SIZE, k=5):
    chosen = []
    
    for i in range(0, POP_SIZE):
        selection = randint(0, len(pop))
        for index in randint(0, len(pop), k-1):
            if scores[index] < scores[selection]:
                selection = index
        chosen.append(pop[selection])
        
    return chosen

def Mutate(new_pop, PM):
    for i in range(0, len(new_pop)):
        for j in range(0, 8520):
            choice = random.random()
            if choice < PM:
                #mutate this bit
                if new_pop[i][j] == 1:
                    new_pop[i][j] = 0
                else:
                    new_pop[i][j] = 1
            
    return new_pop

"""
def Mutate(new_pop, scores, BIT_NUM, PM, POP_SIZE, ONES_THRESH):
    #m = 0
    mx = scores.index(max(scores))
    #print(new_pop)
    for i in range(0, POP_SIZE):
        #print(new_pop[i])
        # Elitism
        #if i == mx:
        #    continue
        #else:
        choice = bitarray.bitarray(BIT_NUM)
        new_pop[i] = new_pop[i] ^ choice
    
            #print(choice)
            #for j in range(0, BIT_NUM):
            #    ch = choice.pop(0)
            #    if ch == 0:
            #        new_pop[i].invert(j)
            #        m = m + 1
    
    return new_pop
"""

def Repair(pop, new_pop, scores, ONES_THRESH):
    ind = scores.index(max(scores))
    for i in range(0, len(new_pop)):
        if new_pop[i].count(1) < ONES_THRESH:
            #replace with new
            new_pop[i] = pop[ind]
            
    return new_pop
    
"""
def Repair(pop, scores, ONES_THRESH):
    #mx = max(score for score in scores)
    mx = max(scores)
    ind = scores.index(mx)
    print("mx: ", mx)
    #print("ind: ", ind)
    print(pop[ind])
    
    for i in range(0, len(pop)):
        if pop[i].count(1) < ONES_THRESH:
            pop[i] = pop[ind]
            scores[i] = mx
            
    return pop
"""            

def Main_Loop(pop, idfs, PC, PM, MAX_IT, ONES_THRESH, ONES_UPPER_THRESH, BIT_NUM, POP_SIZE, ES):
    it_cnt = 0
    #last_tol = 100
    #tol = 1
    last_max_score = -1
    #max_mean_score = -1
    es_cnt = 0
    max_scores = []
    last_max_scores = []
    total_mean_scores = []
    #mean_scores = []
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
        new_pop = Two_Point_Crossover(scores, PC, BIT_NUM, chosen, POP_SIZE, ONES_THRESH)
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
