from keras.datasets import mnist
from matplotlib import pyplot
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import inflect #convert number to string (1 to one)
from sklearn import svm as hi
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import tensorflow as tf
import tensorflow.keras.layers as KL
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
print(trainX.shape)
print(trainy.shape)
print(testX.shape)
print(testy.shape)

#CNN
inputs_CNN = KL.Input(shape=(28,28, 1))
c = KL.Conv2D(512, (3,3), padding = "valid", activation = tf.nn.relu)(inputs_CNN)
m = KL.MaxPool2D((2,2), (2,2))(c)
f = KL.Flatten()(m)
outputs_CNN = KL.Dense(68,activation=tf.nn.softmax)(f)
model_CNN = tf.keras.models.Model(inputs_CNN,outputs_CNN)
model_CNN.summary
model_CNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model_CNN.fit(trainX,trainy, epochs=3)#problem
test_loss, test_acc = model_CNN.evaluate(testX, testy)
print("Test Loss for CNN: {0} - Test Acc for CNN: {1}".format(test_loss, test_acc))


#model Feed-Forward
inputs = KL.Input(shape=(28,28))
l = KL.Flatten()(inputs)
#print(l)
l = KL.Dense(512, activation=tf.nn.relu)(l)
outputs = KL.Dense(256,activation=tf.nn.softmax)(l)
model = tf.keras.models.Model(inputs,outputs)
model.summary
model.compile(optimizer = "Adamax", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model.fit(trainX,trainy, epochs =50)#problem
test_loss, test_acc = model.evaluate(testX, testy)
print("Test Loss for Feed-Forward: {0} - Test Acc for Feed-Forward: {1}".format(test_loss, test_acc))


#RNN
inputs_RNN = KL.Input(shape=(28,28))
x = KL.SimpleRNN(512, activation="sigmoid")(inputs_RNN)
outputs_RNN = KL.Dense(512, activation="softmax")(x)
model_RNN = tf.keras.models.Model(inputs_RNN, outputs_RNN)
model_RNN.summary()
model_RNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["acc"])
model_RNN.fit(trainX,trainy, epochs=50)#problem
test_loss, test_acc = model_RNN.evaluate(testX, testy)
print("Test Loss for RNN: {0} - Test Acc for RNN: {1}".format(test_loss, test_acc))
