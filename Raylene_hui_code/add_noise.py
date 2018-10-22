#!/usr/bin/env python

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize, MinMaxScaler
import pickle
import numpy as np

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


m_uniform = np.random.uniform(0, 0.5, (10000, 23))
x_train = x_train + m_uniform

# mu, sigma = 0, 0.5  # mean and standard deviation
# m_normal = np.random.normal(mu, sigma, (10000, 23))
# x_train = x_train + m_normal

# x_data_perturbed = x_data

# accuracy = 0
# k = 5
# length = int(len(x_data) / k)  # this is 600 entries for the test set

# for i in range(k):
#     start = i*length
#     end = (i+1)*length
#     x_test = x_data_perturbed[start:end]
#     y_test = y_data[start:end]
#     x_train_perturbed = np.concatenate([x_data_perturbed[:start], x_data_perturbed[end:]])
#     y_train = np.concatenate([y_data[:start], y_data[end:]])

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print rf.score(x_test, y_test)

# avg_accuracy = accuracy / float(k)
# print avg_accuracy

vd = np.linalg.norm(orig_data - x_train) / np.linalg.norm(orig_data)
print "vd: ", vd
