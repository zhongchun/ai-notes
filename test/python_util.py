#! /usr/bin/env python
# -*- coding: utf-8 -*ﬁ-
"""
 @FileName: python_util
 @Desc:  
 @Author: yuzhongchun
 @Date: 2019-02-20 17:21
 @Last Modified by: yuzhongchun
 @Last Modified time: 2019-02-20 17:21
"""

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import sys

# for i in range(1, 1001):
#     num = np.ceil(np.random.uniform(0, 10000))
#     print(i, num)

sum = 1000  # 3.69%
sum = 10000  # 36.33%
# sum = 20000     # 56.68%
# sum = 30000     # 68.35%
# sum = 100000    # 90%
# sum = 1000000   # 99.0%
# sum = 10000000  # 99.9%
# sum = 100000000 # 99.99%
rand_data = np.random.randint(0, 10000, sum)
# print(rand_data)
# print(type(rand_data))

# plot a hist figure
# n, bins, patches = plt.hist(rand_data, bins=50)
# plt.grid(True)
# plt.show()

rand_data_frame = pd.DataFrame(rand_data)
# print(rand_data_frame)

repeat_data = rand_data_frame[0].value_counts()

# print(repeat_data)
# print(type(repeat_data))

count = 0
for index, value in enumerate(repeat_data):
    if value > 1:
        count = count + value - 1
        # print(index, value)

print("count =", count)
print("conflict rate = ", count / sum * 100, "%")

size = 28
instances = np.zeros((100, 784))
images = [instance.reshape(size, size) for instance in instances]

print(type(instances))
print(instances.shape)
print(len(instances))

print(type(images))
print(len(images))
print(type(images[0]))
print(images[0].shape)

tmp = np.zeros((28, 28 * 9))
print(tmp.shape)

images.append(np.zeros((size, size * 9)))
print(len(images))
print(images[100].shape)

rimages = images[0: 1 * 10]
print(type(rimages))
print(rimages[0].shape)
it = np.concatenate(rimages, axis=1)
print(it.shape)

tmp = np.random.randint(20, size=(3, 4))
print(tmp)
print(type(tmp[0][0]))

tmp_scaled = tmp.astype(np.float64)
print(tmp_scaled)
print(type(tmp_scaled[0][0]))
sys.exit()
