# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:21:41 2022

@author: HomeTheater
"""
import numpy as np
from matplotlib import pyplot as plt
import random

from bitarray import bitarray

class MyBitArray:
    def __init__(self, number_of_ones, BIT_NUM, ONES_THRESH):
        self.val = [None] * BIT_NUM
        self.BIT_NUM = BIT_NUM
        self.ONES_THRESH = ONES_THRESH
        
        ind = 0
        while number_of_ones > 0:
            ch = random.random()
            if ch < 0.5:
                self.val[ind] = 1
                number_of_ones = number_of_ones - 1
            else:
                self.val[ind] = 0
            ind = (ind + 1) % BIT_NUM
            
        
                    
                    
    def count(self, bit):
        cnt = 0
        
        for i in range(0, self.BIT_NUM):
            if self.val[i] == bit:
                cnt = cnt + 1
        
        return cnt

def get_rand_bit_array(POP_SIZE, mu = 0, sigma = 0.6): # mean and standard deviation

    #BIT_NUM = 8520

    s = np.random.normal(mu, sigma, POP_SIZE)
    #s = [int(x) for x in s]
    #s = [1 if x != 0 else 0 for x in s]
    #print(s.count(1))
    
    """
    count, bins, ignored = plt.hist(s, 30, density=True)
    
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    
                   np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    
             linewidth=2, color='r')
    
    plt.show()
    #"""
    return s

def Get_Normal_Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH, mu, sigma):
    #nums = []
    #for i in range(0, 100):
    #flag = True
    
    pop = get_rand_bit_array(POP_SIZE, mu, sigma)
    t = [p for p in pop if p > ONES_THRESH and p < BIT_NUM]
    #t = pop
    """
    while(flag):
        
        for t0 in t:
            if len(nums) < POP_SIZE:
                nums.append(t0)
            else:
                flag = False
                break
    """
    #plt.hist(t, range=(1500, 7500), bins=POP_SIZE)
    
    #plt.show()
    
    return t

def Get_X_Ones(x, BIT_NUM):
    c = bitarray(BIT_NUM)
    cnt = c.count(1)
    if cnt < x:
        #print("cnt < x")
        while cnt < x:
            ch = random.randint(0, BIT_NUM-1)
            #print("ch: ", ch, " cnt: ", cnt, " x: ", x)
            if c[ch] == 0:
                c.invert(ch)
            cnt = c.count(1)
    elif cnt > x:
        #print("cnt > x")
        while cnt > x:
            ch = random.randint(0, BIT_NUM-1)            
            if c[ch] == 1:
                c.invert(ch)
            cnt = c.count(1)
    #print("cnt == x")
    
    #print(c.count(1))
    
    return c

def Get_Normal_Pop(POP_SIZE, BIT_NUM, ONES_THRESH, mu, sigma):
    t = Get_Normal_Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH, mu, sigma)
    print("t[0]: ", int(t[0]))
    t1 = [int(t0) for t0 in t]
    pop = [Get_X_Ones(x, BIT_NUM) for x in t1]
    if len(pop) < POP_SIZE:
        t = Get_Normal_Init_Pop(POP_SIZE, BIT_NUM, ONES_THRESH, mu, sigma)
        print("t[0]: ", int(t[0]))
        t1 = [int(t0) for t0 in t]
        t2 = [Get_X_Ones(x, BIT_NUM) for x in t1]
        
        seen = pop
        for i in range(0, POP_SIZE - len(pop)):
            ch = random.randint(0, len(t2)-1)
            if t2[ch] not in pop:
                pop.append(t2[ch])
            else:
                seen.append(t2[ch])
    #"""
    cnts = 0
    ones = []
    for x in pop:
        #t = bitarray(int(x))
        ones.append(x.count(1))
        #print(ones[-1])
        cnts = cnts + x.count(1)
    print(cnts/POP_SIZE)
    print(len(pop))
    
    #plt.hist(ones, range=(1500, 7500), bins=POP_SIZE)
    #plt.show()
    #"""
    
    return pop, ones

"""
POP_SIZE = 20
BIT_NUM = 8520
ONES_THRESH = 1000
Get_Normal_Pop(POP_SIZE, BIT_NUM, ONES_THRESH)
"""