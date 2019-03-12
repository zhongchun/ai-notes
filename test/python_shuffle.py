#! /usr/bin/env python
# -*- coding: utf-8 -*ﬁ-
"""
 @FileName: python_shuffle
 @Desc:  
 @Author: yuzhongchun
 @Date: 2019-03-12 11:55
 @Last Modified by: yuzhongchun
 @Last Modified time: 2019-03-12 11:55
"""

import random

max = 10000
index = [i for i in range(0, max)]
random.shuffle(index)

# print(index)
print(len(index))
# print(index[0])

f1 = open('count_code.txt', 'w')
for i in range(0, max):
    s = "%04d" % index[i]
    line = str(i + 1) + ',' + str(i) + ',' + s + '\n'
    print(line)
    # print("%d,%s" % (i, s))
    f1.write(line)
f1.close()
