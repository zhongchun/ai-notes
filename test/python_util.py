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

# for i in range(1, 1001):
#     num = np.ceil(np.random.uniform(0, 10000))
#     print(i, num)

rand_data = np.random.randint(0, 10000, 100000)
# print(rand_data)
# print(type(rand_data))

n, bins, patches = plt.hist(rand_data, bins=50)
plt.grid(True)
plt.show()

rand_data_frame = pd.DataFrame(rand_data)
# print(rand_data_frame)

repeat_data = rand_data_frame[0].value_counts()

# print(repeat_data)
# print(type(repeat_data))

count = 0
for index, value in enumerate(repeat_data):
    if value > 1:
        count += 1
        # print(index, value)

print("count =", count)
