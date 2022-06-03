# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:54:36 2022

@author: Konstantinos Tsiamitros
"""
from matplotlib import pyplot as plt

from random_generator import Get_Exp_Pop
from normal_init_pop import Get_Normal_Pop

POP_SIZE = 20
BIT_NUM = 8520
ONES_THRESH = 1000

for i in range(0, 20):
    pop, ones = Get_Normal_Pop(POP_SIZE, BIT_NUM, ONES_THRESH)
    
    plt.hist(ones, range=(1500, 7500), bins=POP_SIZE)
plt.show()


for i in range(0, 20):
    ones = Get_Exp_Pop(POP_SIZE, BIT_NUM, ONES_THRESH)
    
    plt.hist(ones, range=(1500, 7500), bins=POP_SIZE)
plt.show()