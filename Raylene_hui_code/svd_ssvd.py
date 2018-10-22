#!/usr/bin/env python

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler

all_data = open('all_data_pickled.pkl', 'rb')
data_holder = pickle.load(all_data)
all_data.close()
np.random.shuffle(data_holder)
min_max_scaler = MinMaxScaler()  # to the [0, 1] range
data_holder = min_max_scaler.fit_transform(data_holder)


testing = data_holder[27000:29000]
x_test, y_test = [], []
for r in testing:
    x_test.append(r[:-1])
    y_test.append(r[-1])

y_1, y_0 = [], []
for r in data_holder[:25000]:
    label = r[-1]
    if label == 1:
        y_1.append(r)
    else:
        y_0.append(r)

new_data = y_0[:5000] + y_1[:5000]
np.random.shuffle(new_data)

# new_data = data_holder
x_train, y_train = [], []
for r in new_data:
    x_train.append(r[:-1])
    y_train.append(r[-1])
# x_train = x_data
# y_train = y_data
# print x_data[0]
# print y_data[0]
x_train = np.asarray(x_train)  # [:3000])
y_train = np.asarray(y_train)  # [:3000])
orig_data = x_train

U, s, V = np.linalg.svd(x_train, full_matrices=False)
# s[5:] = 0
k = 20
s = s[:k]  # first k columns of U, first k rows of V
U = U[:, :k]
V = V[:k, :]
# eps = 0.01
# U[np.abs(U) < eps] = 0
# V[np.abs(V) < eps] = 0
#
# # print U.shape
# print U[:10]
# # print V.shape
# print V[:10]
# print s.shape
# print s
x_train = np.dot(U, np.dot(np.diag(s), V))
# print x_data.shape
# np.random.shuffle(x_train)
accuracy = 0
j = 5
length = int(len(x_train) / j)  # this is 600 entries for the test set
# print length
# for i in range(j):
#     start = i*length
#     end = (i+1)*length
#     x_test = x_data[start:end]
#     y_test = y_data[start:end]
#     x_train = np.concatenate([x_data[:start], x_data[end:]])
#     y_train = np.concatenate([y_data[:start], y_data[end:]])

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
# accuracy += rf.score(x_test, y_test)

y_predicted = rf.predict(x_test)
print (metrics.classification_report(y_test, y_predicted))
print (metrics.confusion_matrix(y_test, y_predicted))
print rf.score(x_test, y_test)
# avg_accuracy = accuracy / float(j)
# print avg_accuracy

vd = np.linalg.norm(orig_data - x_train) / np.linalg.norm(orig_data)
print "vd: ", vd
# print len([x for x in y_train if x == 0]) / float(len(y_train))  # 77.4% zeros!
# nb = GaussianNB()  # use default params for now
# nb.fit(x_train, y_train)
# y_predicted = nb.predict(x_test)
# print nb.score(x_test, y_test)
# print nb.get_params()

#print(metrics.classification_report(y_test, y_predicted))
#print(metrics.confusion_matrix(y_test, y_predicted))
