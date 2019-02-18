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


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


if __name__ == '__main__':
    print("hi")
    ret = test_set_check(1, 0.2, hashlib.md5)
    print(ret)
    print("==================================================")
    hash = hashlib.md5
    h = hash(np.int64(1))
    print(h.digest()[-1])
