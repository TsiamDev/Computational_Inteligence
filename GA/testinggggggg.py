# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:28:28 2022

@author: HomeTheater
"""
from matplotlib import pyplot as plt
import numpy as np
from bitarray import bitarray

POP_SIZE = 20
BIT_NUM = 8

pop = []
for i in range(0, POP_SIZE):
    x = np.random.randint(2, size=BIT_NUM)
    #xs = np.array2string(x, formatter={'int':lambda x: str(x)})
    y = x.tolist()
    xs = bitarray(y)
    if xs.count(1) > 1000:
        pop.append(xs)  
        print(xs.count(1))

#plt.hist(x, range=(1000, 8520), bins=POP_SIZE)
#plt.show()