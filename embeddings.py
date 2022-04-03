# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:38:38 2022

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

from load_raw_data import preprocess_vecs
from load_labels import load_labels

from tensorflow.keras.preprocessing.sequence import pad_sequences

from numpy.random import seed
seed(42)
tf.random.set_seed(42)

# load train dataset
train = preprocess_vecs()

# load labels
#train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label-small.dat'
train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label.dat'
train_labels = load_labels(train_label_file)

# define the dimensions
n_inputs = len(train[0])#8520
n_outputs = 20

# first hidden layer
n_hidden1 = n_outputs # best
#n_hidden1 = (n_inputs + n_outputs)/2
#n_hidden1 = n_outputs + n_inputs
#second hidden layer
n_hidden2 = n_outputs # best
#n_hidden2 = (n_inputs + n_outputs)/4
#n_hidden2 = (n_inputs + n_outputs)/2

scores = []
#cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
#for i, (_train, _test) in enumerate(cv.split(train)):
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (_train, _test) in enumerate(kfold.split(train)):
    #print(i, _train, _test)    

    # Build model
    inputs = layers.Input(shape=(n_inputs,))
    # An embedding layer alongside a flatten layer have been added to the model
    #embeddings = layers.Embedding(input_dim=8520, output_dim=20, input_length=302)(inputs)
    embeddings = layers.Embedding(input_dim=8520, output_dim=40, input_length=302)(inputs)
    flat = layers.Flatten()(embeddings)
    lReLU = layers.LeakyReLU(alpha=0.04)
    hidden1 = layers.Dense(n_hidden1, activation=lReLU)(flat)
    hidden2 = layers.Dense(n_hidden2, activation=lReLU)(hidden1)
    output = layers.Dense(n_outputs, activation='sigmoid')(hidden2)
    
    outputs = []
    for j in range(0, 20):
        outputs.append( layers.Lambda(lambda x: x[...,j:j+1])(output))
    
    opt = keras.optimizers.SGD(learning_rate=0.08, momentum=0.51, decay=0.07, nesterov=False)
    
    model = tf.keras.Model(inputs, outputs)
    los = tf.keras.losses.BinaryCrossentropy(from_logits=False), 
    metr =['mse', 'accuracy']
    model.compile(optimizer=opt, loss=[20*los], metrics=metr)
    
    model.summary()

    print("Training Model...")
    model.fit(x=train[_train], y=train_labels[_train], verbose=1, batch_size=7000, epochs=50, shuffle=True)
    print("Evaluating...")
    
    # Evaluate model
    score = model.evaluate(train[_test], train_labels[_test], verbose=1)    

    scores.append(score)
    
    print("Fold :", i, score)

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

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(xs, mse)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_title('MSE')

ax[0, 1].plot(xs, loss)
#ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_title('CE Loss')

ax[1, 0].plot(xs, acc)
ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_title('Accuracy')

