# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:12:12 2022

@author: HomeTheater
"""
import tf_idf as tf
from matplotlib import pyplot as plt

idfs = tf.PrepareFitnessData()

idfs = sorted(idfs.values(), reverse=True)
#print(idfs[:1000])

plt.plot(idfs)
plt.xlabel("# of term")
plt.show("tf-idf value of term")