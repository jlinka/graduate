#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 下午3:10
# @Author  : jlinka
# @File    : __bigfile.py

# 用于处理大文件

import csv


# 读取所有文件内容 通过yield每次返回一行内容
def get_lines(filepath: str):
    with open(filepath, 'r', errors='ignore') as read_file:  # 以读的方式打开文件
        for line in read_file:
            yield line  # 返回每行内容

