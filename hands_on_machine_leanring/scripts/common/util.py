#! /usr/bin/env python
# -*- coding: utf-8 -*ﬁ-
"""
 @FileName: util
 @Desc:  
 @Author: yuzhongchun
 @Date: 2019-02-21 18:09
 @Last Modified by: yuzhongchun
 @Last Modified time: 2019-02-21 18:09
"""
import os
import matplotlib.pyplot as plt


def print_line(v, name=None, length=120):
    print('=' * length)
    if name is not None:
        print(name)
    print(v)
    print()


def save_fig(fig_id, project_root_dir, chapter_id, tight_layout=True, length=120):
    path = os.path.join(project_root_dir, "images", chapter_id, fig_id + ".png")
    print('=' * length)
    print("Saving figure: ", fig_id)
    print()
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
