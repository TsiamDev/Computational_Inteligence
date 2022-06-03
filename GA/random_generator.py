# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:04:07 2022

@author: Konstantinos Tsiamitros
"""
#import matplotlib.pyplot as plt
import numpy as np
import random

from bitarray import bitarray

def My_Rand(BIT_NUM):
    """
    pop = ""
    for i in range(0, BIT_NUM):
        t = random.randint(0, 1)
        pop = pop + str(t)
    return(pop)
    """
    return get_rand_bit_array(BIT_NUM, mu = 0, sigma = 0.8)        

def Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH):
    temp_strs = []
    pops = []
    for i in range(0, POP_SIZE):
        flag = True
        while(flag):
            t = My_Rand(BIT_NUM)
            if t not in temp_strs:
                # here <t> is the bit string
                temp_strs.append(t)
                t = bitarray(t)
                cnt = t.count(1)
                if cnt > ONES_THRESH:
                    flag = False
                    # here <t> is the bitarray
                    pops.append(t)

    
    return pops

def get_rand_bit_array(BIT_NUM, mu = 0, sigma = 0.6): # mean and standard deviation

    #BIT_NUM = 8520

    s = np.random.normal(mu, sigma, BIT_NUM)
    s = [int(x) for x in s]
    s = [1 if x != 0 else 0 for x in s]
    #print(s.count(1))
    
    """
    count, bins, ignored = plt.hist(s, 30, density=True)
    
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    
                   np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    
             linewidth=2, color='r')
    
    plt.show()
    #"""
    return s


def Get_Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH):
    temp = []
    for i in range(0, 500, int(500/POP_SIZE)):
        flag = True
        while(flag):
            t = get_rand_bit_array(BIT_NUM, mu = 0, sigma = 0.65 + i/100)
            t = bitarray(t)
            if t not in temp:
                cnt = t.count(1)
                #print("# of 1's is: " + str(cnt))
                #if cnt >= ONES_THRESH:
                if cnt > ONES_THRESH:
                    #print(temp)
                    flag = False
                    temp.append(t)

    #print(t)
    
    return temp

def Get_Exp_Pop(POP_SIZE, BIT_NUM, ONES_THRESH):
    pop = Get_Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH)
    cnts = 0
    ones = []
    for x in pop:
        ones.append(x.count(1))
        print(ones[-1])
        cnts = cnts + x.count(1)
    print(cnts/POP_SIZE)
    
    return pop, ones
    #plt.plot(ones, range(0, len(ones)))
    #plt.show()

#Get_Init_Pop()
#get_rand_bit_array(10, 1, 0.6)
#Get_Rand_Float_In_0_1()