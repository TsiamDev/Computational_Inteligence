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

from numpy.random import seed
seed(42)
tf.random.set_seed(42)

def evaluate_model(X, y, model):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        print("Training Model...")
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        print("Predicting...")
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results

def train_predict(_train, _test, n_hidden1, n_hidden2, lr=0.001, m=0.2, d=0.0):
    model = keras.Sequential()
    model.add(layers.Dense(n_hidden1, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.Dense(n_hidden2, input_dim=n_hidden1, kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.Dense(n_outputs, input_dim=n_hidden2, activation='sigmoid'))
    #es = EarlyStopping(monitor='accuracy', mode='max', min_delta=0.0001)#, baseline=0.2)
    #es = EarlyStopping(monitor='loss', mode='min')#, baseline=0.001)
    #opt = keras.optimizers.Adam(learning_rate=lr)
    opt = keras.optimizers.SGD(learning_rate=lr, momentum=m, decay=d, nesterov=False)
    #opt = keras.optimizers.SGD(learning_rate=lr, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse', 'accuracy'])
    #print(model.metrics_names)
    
    #print(evaluate_model(train, train_labels, model))
    
    #"""
    print("Training Model...")
    #model.fit(train[_train], train_labels[_train], verbose=1, batch_size=7000, epochs=50, callbacks=[es])
    model.fit(train[_train], train_labels[_train], verbose=1, batch_size=7000, epochs=50)
    print("Evaluating...")
    #yhat = model.predict(test)
    #yhat = yhat.round()
    
    # Evaluate model
    score = model.evaluate(train[_test], train_labels[_test], verbose=1)
    
    
    #acc = accuracy_score(yhat, test_labels)
    #print(acc)
    #plt.plot(model.history['mean_squared_error'])

    return score

def elbow_grease():
    accs = []
    for n_hidden in range(0, 8600, 200):
        acc = train_predict(n_hidden)
        
        accs.append(acc)
        #"""
        
    plt.plot(range(0, len(accs)), accs)
    plt.show()

#lst, ys = preprocess_and_plot()
train, test = preprocess_and_plot()

# load labels
#train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label-small.dat'
train_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\train-label.dat'
train_labels = load_labels(train_label_file)

#test_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\test-label-small.dat'
test_label_file = 'C:\\Users\\HomeTheater\\Desktop\\ΥΝ\\Εργασία2022\\Data\\test-label.dat'
test_labels = load_labels(test_label_file)

# define the model
n_inputs = len(train[0])#8520
n_outputs = 20

#n_hidden = n_outputs
#n_hidden = (n_inputs + n_outputs)/2
#n_hidden = n_inputs + n_outputs
#n_hidden = 8520

#n_hidden1 = (n_inputs + n_outputs)/2
#n_hidden2 = n_outputs

#n_hidden1 = n_outputs
#n_hidden1 = (n_inputs + n_outputs)/2
#n_hidden2 = (n_inputs + n_outputs)/2
#n_hidden2 = n_outputs # ~ 0.1 acc
#n_hidden2 = (n_inputs + n_outputs)/4

n_hidden1 = 400
n_hidden2 = 400



#train_predict(n_hidden)

# elbow grease method
#elbow_grease()

"""
# Build model
model = keras.Sequential()
model.add(layers.Dense(n_hidden1, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
model.add(layers.Dense(n_hidden2, kernel_initializer='he_uniform', activation='relu'))
model.add(layers.Dense(n_outputs, activation='sigmoid'))
#es = EarlyStopping(monitor='accuracy', mode='max', min_delta=0.0001)#, baseline=0.2)
#es = EarlyStopping(monitor='loss', mode='min')#, baseline=0.001)
#opt = keras.optimizers.Adam(learning_rate=lr)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, decay=0.9, nesterov=False)
#opt = keras.optimizers.SGD(learning_rate=lr, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse', 'accuracy'])
#print(model.metrics_names)

#print(evaluate_model(train, train_labels, model))
#"""
scores = []
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
#kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#for i, (_train, _test) in enumerate(kfold.split(train)):
for i, (_train, _test) in enumerate(cv.split(train)):

    #print(i, _train, _test)    
    """
    inputs = keras.Input(shape=(n_inputs,), name="digits")
    x = layers.Dense(n_hidden1, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(n_hidden2, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(20, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.SGD(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.BinaryCrossentropy(),
        # List of metrics to monitor
        metrics=["mse", "accuracy"]
    )
    
    history = model.fit(
        train[_train],
        train_labels[_train],
        batch_size=7000,
        epochs=10,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(train[_test], train_labels[_test])
    )
    """
    # Build model
    model = keras.Sequential()
    #leaky_RELU = tf.keras.layers.LeakyReLU(alpha=0.3)
    #model.add(layers.Dense(n_hidden1, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    #model.add(layers.Dense(n_hidden1, input_dim=n_inputs, activation=leaky_RELU))
    model.add(layers.LeakyReLU(alpha=0.21))
    #model.add(layers.Dense(n_hidden1, activation='sigmoid'))
    #model.add(layers.Dense(n_outputs, activation=leaky_RELU))
    model.add(layers.Dense(n_outputs, activation='tanh'))
    #model.add(layers.Dense(n_outputs, activation='softmax'))
    #es = EarlyStopping(monitor='accuracy', mode='max')#, min_delta=0.0001)#, baseline=0.2)
    #es = EarlyStopping(monitor='loss', mode='min')#, baseline=0.001)
    #opt = keras.optimizers.Adam(learning_rate=lr)
    
    #opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.1, decay=0.02, nesterov=False)
    opt = keras.optimizers.SGD(learning_rate=0.08, momentum=0.86, decay=0.05, nesterov=False)
    #opt = keras.optimizers.Adam(learning_rate=0.1, decay=0.9)
    #opt = keras.optimizers.SGD(learning_rate=lr, nesterov=False)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse', 'accuracy'])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=opt, metrics=['mse', 'accuracy'])
    #print(model.metrics_names)
    
    #print(evaluate_model(train, train_labels, model))

    print("Training Model...")
    #model.fit(train[_train], train_labels[_train], verbose=1, batch_size=7000, epochs=50, callbacks=[es])
    #model.fit(x=train[_train], y=train_labels[_train], validation_split=0.3, verbose=1, batch_size=7000, epochs=50, shuffle=True)
    model.fit(x=train[_train], y=train_labels[_train], verbose=0, batch_size=7000, epochs=80, shuffle=True)
    #model.fit(x=train[_train], y=train_labels[_train], verbose=1, batch_size=7000, epochs=10, shuffle=True, callbacks=[es])
    print("Evaluating...")
    #yhat = model.predict(test)
    #yhat = yhat.round()
    
    # Evaluate model
    score = model.evaluate(train[_test], train_labels[_test], verbose=1)    


    #score = train_predict(_train, _test, n_hidden1, n_hidden2, 0.1, 0.6, 0.9)
    scores.append(score)
    
    print("Fold :", i, score)
    """
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
    mse.append(scores[i][1])
    acc.append(scores[i][2])

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(xs, mse)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_title('MSE')
#plt.show()
ax[0, 1].plot(xs, loss)
ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_title('CE Loss')
#plt.show()
ax[1, 0].plot(xs, acc)
ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_title('Accuracy')
#plt.show()
#ax[1, 1].plot(loss, acc)
#ax[1, 1].set_title('Learning curve')
"""
for a in ax.flat:
    a.set(xlabel='x-label', ylabel='y-label')

for a in ax.flat:
    a.label_outer()
    
"""
"""
for i in range(0, len(scores[0])):
    plt.plot(xs, scores[i])
    plt.show()
"""