# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:42:58 2022

@author: Konstantinos Tsiamitros
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, KFold
import matplotlib.pyplot as plt

import numpy as np

from preprocessing import preprocess_and_plot
from load_labels import load_labels

import keras_tuner as kt

from numpy.random import seed
seed(42)
tf.random.set_seed(42)

def model(hp):
    # Build model
    model = keras.Sequential()
    hp_a = hp.Float('alpha', min_value=0.01, max_value=0.3, step=0.01)
    
    model.add(layers.LeakyReLU(alpha=hp_a))
    model.add(layers.Dense(20, activation='sigmoid'))
    hp_lr = hp.Float('learning_rate', min_value=0.01, max_value=0.1, step=0.01)
    hp_m = hp.Float('momentum', min_value=0.01, max_value=0.9, step=0.05)
    hp_d = hp.Float('decay', min_value=0.01, max_value=0.1, step=0.01)
    opt = keras.optimizers.SGD(learning_rate=hp_lr, momentum=hp_m, decay=hp_d, nesterov=False)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=opt, metrics=['mse', 'accuracy'])
    
    return model

# load dataset
train, test = preprocess_and_plot()

# load labels
#train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label-small.dat'
train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label.dat'
train_labels = load_labels(train_label_file)

tuner = kt.Hyperband(model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='sigmoid',
                     project_name='intro_to_kt')

tuner.search(train, train_labels, epochs=50, validation_split=0.2)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#print(f"The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('alpha')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}, best momentum is {best_hps.get('momentum')}, best decay is {best_hps.get('decay')}.")
# best accuracy
#1 h layer softmax: 0.28285887837409973 - 0.21, 0.08, 0.86, 0.05
#1 h layer tanh: 0.2719563841819763 - 0.28, 0.09, 0.86, 0.08
#1 h layer sigmoid:  0.2816474735736847 - 0.03, 0.03, 0.76, 0.01