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
mat = scipy.io.loadmat('/Users/mahesh1/Desktop/Data Science/603/pose.mat')
#print(type(mat))
#print(mat)



data = mat['pose']
print(data.shape)
# plt.figure()
# plt.imshow(data[:,:,1,0])
# plt.show()

img = []
arr_train = []
arr_test = []
for j in range(0,68):
	for i in range(0,13):
		if i < 10:
			new = data[:,:,i,j]
			arr_train.append(new)

		else:
			new = data[:,:,i,j]
			arr_test.append(new)




arr_train = np.asarray(arr_train)
arr_test = np.asarray(arr_test)

print(arr_train.shape)
print(arr_test.shape)
#data = np.reshape(data, (884, 48,40))
#print(data.shape)
#plt.figure()
#plt.imshow(arr_train[10,:,:])
#plt.show()
 
#arr_train = []
#print(arr_big)


# p = inflect.engine()
# #training
# for i in range(0,68):
# 	num = p.number_to_words(i+1)
# 	#print(num)
# 	t = "_train"
# 	num_update = num+t
# 	#print(num_update)
# 	for j in range(0,10):
# 		num_update = img[i*13:i*13+10,:,:]
# 		num_update = np.
# 	arr_train.append(num_update)
# 	#print(arr_big)

# arr_train = np.asarray(arr_train)
# #arr_train = np.reshape(arr_train, (680, 48,40))
arr_train_CNN = np.expand_dims(arr_train, axis = -1)

# plt.figure()
# plt.imshow(arr_test[3,:,:])
# plt.show()

#print(arr_train[1])
print(arr_train.shape)
#print(arr_train[1].shape)

# arr_test = []
# #testing
# for i in range(0,68):
# 	num = p.number_to_words(i+1) #1->one
# 	#print(num)
# 	t = "_test"
# 	num_update = num+t
# 	#print(num_update)
# 	num_update = img[i*13+10:i*13+10+3,:,:]
# 	arr_test.append(num_update)
# arr_test  = np.asarray(arr_test)
# #print(arr_train[1])
# arr_test = np.reshape(arr_test, (68*3, 48,40))
arr_test_CNN = np.expand_dims(arr_test, axis = -1)
#print(arr_test[0])
#print(arr_test.shape)
#print(arr_test[1].shape)


#for i in range(0, )
labels_train = np.zeros((68*10))
labels_test = np.zeros((68*3))

#train
for i in range(0,68):
	#labels_train[i*13:i*13+10]= i
	labels_train[i*10:i*10+10] = i
	#labels_train[i*10:i*10+10] = i+1
#test
for i in range(0,68):
		#labels_test[i*13+10:i*13+10+3] = i
		labels_test[i*3:i*3+3] = i
		#labels_test[i*3:i*3+3] = i+1
print(labels_test)




# plt.figure()
# plt.imshow(arr_train[0,:,:])
# plt.show()

#model Feed-Forward
inputs = KL.Input(shape=(48,40))
l = KL.Flatten()(inputs)
#print(l)
l = KL.Dense(512, activation=tf.nn.relu)(l)
outputs = KL.Dense(256,activation=tf.nn.softmax)(l)
model = tf.keras.models.Model(inputs,outputs)
model.summary
model.compile(optimizer = "Adamax", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model.fit(arr_train,labels_train, epochs =50)#problem
test_loss, test_acc = model.evaluate(arr_test, labels_test)
print("Test Loss for Feed-Forward: {0} - Test Acc for Feed-Forward: {1}".format(test_loss, test_acc))

#CNN
inputs_CNN = KL.Input(shape=(48,40, 1))
c = KL.Conv2D(512, (3,3), padding = "valid", activation = tf.nn.relu)(inputs_CNN)
m = KL.MaxPool2D((3,3), (3,3))(c)
f = KL.Flatten()(m)
outputs_CNN = KL.Dense(68,activation=tf.nn.softmax)(f)
model_CNN = tf.keras.models.Model(inputs_CNN,outputs_CNN)
model_CNN.summary
model_CNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model_CNN.fit(arr_train_CNN,labels_train, epochs=50)#problem
test_loss, test_acc = model_CNN.evaluate(arr_test_CNN, labels_test)
print("Test Loss for CNN: {0} - Test Acc for CNN: {1}".format(test_loss, test_acc))



#RNN
inputs_RNN = KL.Input(shape=(48,40))
x = KL.SimpleRNN(512, activation="sigmoid")(inputs_RNN)
outputs_RNN = KL.Dense(512, activation="softmax")(x)
model_RNN = tf.keras.models.Model(inputs_RNN, outputs_RNN)
model_RNN.summary()
model_RNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["acc"])
model_RNN.fit(arr_train,labels_train, epochs=50)#problem
test_loss, test_acc = model_RNN.evaluate(arr_test, labels_test)
print("Test Loss for RNN: {0} - Test Acc for RNN: {1}".format(test_loss, test_acc))
























