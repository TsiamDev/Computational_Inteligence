# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:22:32 2022

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

from numpy.random import seed
seed(42)
tf.random.set_seed(42)

train, test = preprocess_and_plot()

# load labels
#train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label-small.dat'
train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label.dat'
train_labels = load_labels(train_label_file)

#test_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\test-label-small.dat'
test_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\test-label.dat'
test_labels = load_labels(test_label_file)

# define the dimensions
n_inputs = len(train[0])#8520
n_outputs = 20
# first hidden layer
n_hidden1 = n_outputs # best
#n_hidden1 = (n_inputs + n_outputs)/2
#n_hidden1 = n_outputs + n_inputs
# second hidden layer
n_hidden2 = n_outputs # best
#n_hidden2 = (n_inputs + n_outputs)/4
#n_hidden2 = (n_inputs + n_outputs)/2

scores = []
#cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (_train, _test) in enumerate(kfold.split(train)):
#for i, (_train, _test) in enumerate(cv.split(train)):

    #print(i, _train, _test)    

    # Hyperparameter tuning
    # n_outputs = O
    #Best val_loss So Far: 8.289506912231445
    #Total elapsed time: 00h 05m 19s

    #Search: Running Trial #30

    #Hyperparameter    |Value             |Best Value So Far 
    #alpha             |0.18              |0.29              
    #learning_rate     |0.1               |0.07              
    #momentum          |0.61              |0.51              
    #decay             |0.04              |0.08              
    #tuner/epochs      |10                |10                
    #tuner/initial_e...|0                 |4                 
    #tuner/bracket     |0                 |2                 
    #tuner/round       |0                 |2   
    
    # n_outputs = (I+O)/2
    #Best loss So Far: 8.245855331420898
    #Total elapsed time: 00h 29m 54s

    #Search: Running Trial #21

    #Hyperparameter    |Value             |Best Value So Far 
    #alpha             |0.01              |0.04              
    #learning_rate     |0.02              |0.08              
    #momentum          |0.51              |0.51              
    #decay             |0.05              |0.07              
    #tuner/epochs      |4                 |10                
    #tuner/initial_e...|0                 |4                 
    #tuner/bracket     |1                 |2                 
    #tuner/round       |0                 |2                 
    
    # n_outputs = I + O
    #Best val_loss So Far: 8.375269889831543
    #Total elapsed time: 00h 14m 21s

    #Search: Running Trial #7

    #Hyperparameter    |Value             |Best Value So Far 
    #alpha             |0.2               |0.1               
    #learning_rate     |0.04              |0.09              
    #momentum          |0.61              |0.06              
    #decay             |0.07              |0.1               
    #tuner/epochs      |2                 |2                 
    #tuner/initial_e...|0                 |0                 
    #tuner/bracket     |2                 |2                 
    #tuner/round       |0                 |0  
    
    # Build model
    inputs = layers.Input(shape=(n_inputs,))
    lReLU = layers.LeakyReLU(alpha=0.04)
    hidden1 = layers.Dense(n_hidden1, activation=lReLU)(inputs)
    hidden2 = layers.Dense(n_hidden2, activation=lReLU)(hidden1)
    output = layers.Dense(n_outputs, activation='sigmoid')(hidden2)
    
    outputs = []
    for j in range(0, 20):
        outputs.append( layers.Lambda(lambda x: x[...,j:j+1])(output))
    
    # uncomment for each experiement
    #opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.2, nesterov=False)
    #opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.6, nesterov=False)
    #opt = keras.optimizers.SGD(learning_rate=0.05, momentum=0.6, nesterov=False)
    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, nesterov=False)
    
    model = tf.keras.Model(inputs, outputs)
    los = tf.keras.losses.BinaryCrossentropy(from_logits=False), 
    metr =['mse', 'accuracy']
    model.compile(optimizer=opt, loss=[20*los], metrics=metr)
    
    print("Training Model...")
    # For training the model, max itreations is the termination criteria (in the form of epochs)
    model.fit(x=train[_train], y=train_labels[_train], verbose=1, batch_size=7000, epochs=50, shuffle=True)
    print("Evaluating...")
    score = model.evaluate(train[_test], train_labels[_test], verbose=1)    

    scores.append(score)
    
    print("Fold :", i, score)
    """
    # a minimun tolerance was specified, for loop termination
    # but since I did not know the optimal hyperparameter values
    # I decided to leave it up to max iterations of the 5-fold CV
    if i > 0:
        tol = float(abs(scores[i][2] - scores[i-1][2])/(scores[i][2] + scores[i-1][2]))
        print(tol)
    
        if (tol < 0.05) or (tol > 0.7):
            break
    
    #"""
print("Scores: ", scores)
xs = [i for i in range(0, len(scores))]

mse = []
loss = []
acc = []
for i in range(0, len(scores)):
    loss.append(scores[i][0])
    # all independent sigmoids in the output layer return the same accurqacy
    # so avg_* = (_val * 20 )/20 - instead just read the accuracy and mse
    # of the first sigmoid output
    mse.append(scores[i][21])
    acc.append(scores[i][22])

# plot the figures
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(xs, mse)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_title('MSE')

ax[0, 1].plot(xs, loss)
ax[0, 1].set_title('CE Loss')

ax[1, 0].plot(xs, acc)
ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_title('Accuracy')
