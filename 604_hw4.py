import scipy.io
import numpy as np
from sklearn import svm as hi
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
mat = scipy.io.loadmat('/Users/mahesh1/Downloads/usps_all_final.mat')
#print(type(mat))
#print(mat)


data = mat['data']
#print(data.shape)

#training
one = data[:,0:800:,0]
two = data[:,0:800,1]
three = data[:,0:800,2]
four = data[:,0:800,3]
five = data[:,0:800,4]
six= data[:,0:800,5]
seven = data[:,0:800,6]
eight= data[:,0:800,7]
nine = data[:,0:800,8]
zero = data[:,0:800,9]

# #validation
one_v = data[:,800:1000:,0]
two_v = data[:,800:1000,1]
three_v = data[:,800:1000,2]
four_v = data[:,800:1000,3]
five_v = data[:,800:1000,4]
six_v = data[:,800:1000,5]
seven_v = data[:,800:1000,6]
eight_v = data[:,800:1000,7]
nine_v = data[:,800:1000,8]
zero_v = data[:,800:1000,9]

#testing
one_t = data[:,1000:1100:,0]
two_t = data[:,1000:1100,1]
three_t = data[:,1000:1100,2]
four_t = data[:,1000:1100,3]
five_t = data[:,1000:1100,4]
six_t= data[:,1000:1100,5]
seven_t = data[:,1000:1100,6]
eight_t= data[:,1000:1100,7]
nine_t = data[:,1000:1100,8]
zero_t = data[:,1000:1100,9]

print(two.shape)

#a = np.array([[1, 2], [3, 4]])
#b = np.array([[5, 6]])

#print(np.concatenate((a, b), axis=0))





train =np.concatenate((one,two,three,four,five,six,seven,eight,nine,zero), axis=1)
train = train.T
print(train.shape)



labels = np.zeros((8000))





for i in range(0,10):
	if i ==9:
		labels[i*800:i*800+300] = 0
	else:
		labels[i*800:i*800+300] = i+1



valid =np.concatenate((one_v,two_v,three_v,four_v,five_v,six_v,seven_v,eight_v,nine_v,zero_v), axis=1)
valid = valid.T
print(valid.shape)

labels_valid = np.zeros((2000))


for i in range(0,10):
	if i ==9:
		labels_valid[i*800:i*800+200] = 0
	else:
		labels_valid[i*800:i*800+200] = i+1



test =np.concatenate((one_t,two_t,three_t,four_t,five_t,six_t,seven_t,eight_t,nine_t,zero_t), axis=1)
test = test.T
print(test.shape)

labels_test = np.zeros((1000))


for i in range(0,10):
	if i ==9:
		labels_test[i*1000:i*1000+100] = 0
	else:
		labels_test[i*1000:i*1000+100] = i+1

# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(train, labels)
# print(neigh.predict(test))
# acc = neigh.score(test, labels_test)
#knn for 1-20 neighbors

# for i in range(1,21):
# 	neigh = KNeighborsClassifier(n_neighbors=i)
# 	neigh.fit(train, labels)
# 	#print(neigh.predict(test))
# 	acc = neigh.score(valid, labels_valid)
# 	error_rate = 1-acc
# 	print(f'accuracy for knn for valid =: {i} {acc}')
# 	print(f'error_rate for knn for valid =: {i} {error_rate}')


# for i in range(1,21):
# 	neigh = KNeighborsClassifier(n_neighbors=i)
# 	neigh.fit(train, labels)
# 	#print(neigh.predict(test))
# 	acc = neigh.score(test, labels_test)
# 	error_rate = 1-acc
# 	print(f'accuracy for knn for test =: {i} {acc}')
# 	print(f'error_rate for knn for test =: {i} {error_rate}')





# #print(f'accuracy for knn: {acc}')


# svm = LinearSVC()
# svm.fit(train, labels)
# print(svm.predict(test))
# acc_svc_test = svm.score(test, labels_test)
# error_rate_test_svm = 1- acc_svc_test
# print(f'accuracy for svm for test: {acc_svc_test}')
# print(f'error_rate for svm for test =: {error_rate_test_svm}')

# svm = LinearSVC()
# svm.fit(train, labels)
# print(svm.predict(valid))
# acc_v = svm.score(valid, labels_valid)
# error_rate_valid_svm = 1 - acc_v
# print(f'accuracy for svm for valid: {acc_v}')
#print(f'error_rate for svm for valid =: {error_rate_valid_svm}')



# for i in range(1,21):
# 	neigh = KNeighborsClassifier(n_neighbors=i)
# 	neigh.fit(train, labels)
# 	#print(neigh.predict(test))
# 	acc = neigh.score(test, labels_test)
# 	print(f'accuracy for knn =: {i} {acc}')


# Nonlin = hi.NuSVC()
# Nonlin.fit(train, labels)
# print(Nonlin.predict(test))
# acc = Nonlin.score(test, labels_test)
# print(f'accuracy for Nonlinear SVM: {acc}')









