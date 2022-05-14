# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:04:07 2022

@author: HomeTheater
"""
import matplotlib.pyplot as plt
import numpy as np

from bitarray import bitarray

BIT_NUM = 8520
ONES_THRESH = 1000

def get_rand_bit_array(mu = 0, sigma = 0.6): # mean and standard deviation

    s = np.random.normal(mu, sigma, BIT_NUM)
    s = [int(x) for x in s]
    s = [1 if x != 0 else 0 for x in s]
    print(s.count(1))
    
    
    count, bins, ignored = plt.hist(s, 30, density=True)
    
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    
                   np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    
             linewidth=2, color='r')
    
    plt.show()
    
    return s


def Get_Init_Pop():
    temp = []
    for i in range(0, 100, 20):
        flag = True
        while(flag):
            t = get_rand_bit_array(sigma = 0.65 + i/100)
            if t not in temp:
                cnt = t.count(1)
                #print("# of 1's is: " + str(cnt))
                if cnt >= ONES_THRESH:
                    #print(temp)
                    flag = False
                    temp.append(t)
                    
    for i in range(100, 200, 20):
        flag = True
        while(flag):
            t = get_rand_bit_array(sigma = 0.65 + i/100)
            if t not in temp:
                cnt = t.count(1)
                #print("# of 1's is: " + str(cnt))
                if cnt >= ONES_THRESH:
                    #print(temp)
                    flag = False
                    temp.append(t)
    
    for i in range(200, 500, 30):
        flag = True
        while(flag):
            t = get_rand_bit_array(sigma = 0.65 + i/100)
            if t not in temp:
                cnt = t.count(1)
                #print("# of 1's is: " + str(cnt))
                if cnt >= ONES_THRESH:
                    #print(temp)
                    flag = False
                    temp.append(t)

    #print(t)
    
    return temp

#Get_Init_Pop()