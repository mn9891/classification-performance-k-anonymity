#!/usr/bin/env python

import csv
import pickle

input_data = open("all_data.csv")
data_input = csv.reader(input_data, delimiter=',')
next(data_input)  # skip header row
next(data_input)

data_holder = []

for row in data_input:
    r = [int(float(n)) for n in row[1:]]  # get rid of id
    for i in range(5, 11):
        r[i] += 2
    for i in range(11, 23):
        if r[i] < 0:
            r[i] = 0
    data_holder.append(r)

print data_holder[0]
all_data_pickled = open('all_data_pickled.pkl', 'wb')

pickle.dump(data_holder, all_data_pickled)
all_data_pickled.close()
