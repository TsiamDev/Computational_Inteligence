# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:59:15 2022

@author: Konstantinos Tsiamitros
"""


# Constants
POP_SIZE = 50
BIT_NUM = 8520 # number of words
ONES_THRESH = 1000 # lower threshold for number of words
ONES_UPPER_THRESH = 3500 # upper threshold for number of words

PC = 0.6 # propability of crossover
PM = 0.1 # propability of mutation

# Termination criteria
MAX_IT = 800 # maximum number of generations

# TO-DO
TOL = 0.01 # tolerance
ES = 10 # early stopping