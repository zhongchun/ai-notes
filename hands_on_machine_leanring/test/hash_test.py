#! /usr/bin/env python
# -*- coding: utf-8 -*ﬁ-
"""
 @FileName: hash_test
 @Desc:  
 @Author: yuzhongchun
 @Date: 2019-02-18 17:51
 @Last Modified by: yuzhongchun
 @Last Modified time: 2019-02-18 17:51
"""

import numpy as np
import hashlib
import pandas as pd


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


if __name__ == '__main__':
    ret = test_set_check(4, 0.2, hashlib.md5)
    print("==================================================")
    print(ret)
    print("==================================================")
    print(256 * 0.2)
    hash = hashlib.md5
    h = hash(np.int64(4))
    print(h.digest()[-1])

    # test where
    s = pd.Series(range(5))
    print(s.where(s > 0))
    print('===========================================')
    print(s.where(s > 1, 20))
    print('===========================================')
    print(s.where(s > 1, 20, inplace=True))
    print('===========================================')
    print(s)




