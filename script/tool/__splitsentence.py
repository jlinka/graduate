#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 下午3:09
# @Author  : jlinka
# @File    : __splitsentence.py

import re


# 将单条新闻分句
def resume2sentences(srcresume: str):
    # 停用掉某些符号 《》<>（）()「」{}|
    # pattern = r'[|]+'
    # pat = re.compile (pattern)
    # srcresume = re.sub (pat, '，', srcresume)  # 将'|'转成'，'

    pattern1 = r'[《》<>「」{}【】()（）""“” \[\]]'
    pat1 = re.compile(pattern1)
    srcresume = re.sub(pat1, '', srcresume)  # 将'《》<>「」{}'去掉

    # 分句 ，。？！；
    # pattern2 = r'[||:：,，.。?？!！;；\n\t\r]'
    pattern2 = r'[||:：。?？!！;；\n\t\r]'
    pat2 = re.compile(pattern2)
    sentences = re.split(pat2, srcresume.strip())  # 以'，。？！；'为句子分隔符分割句子
    return [sentence for sentence in sentences if sentence]



# 将单条新闻句子去标点
def resume2sentences1(srcresume: str):
    # 停用掉某些符号 《》<>（）()「」{}|
    # pattern = r'[|]+'
    # pat = re.compile (pattern)
    # srcresume = re.sub (pat, '，', srcresume)  # 将'|'转成'，'

    pattern1 = r'[《》<>「」{}【】()（）""“” \[\]]'
    pat1 = re.compile(pattern1)
    srcresume = re.sub(pat1, '', srcresume)  # 将'《》<>「」{}'去掉

    # 分句 ，。？！；
    # pattern2 = r'[||:：,，.。?？!！;；\n\t\r]'
    pattern2 = r'[||:：，,。.?？!！;；\n\t\r]'
    pat2 = re.compile(pattern2)
    sentences = re.split(pat2, srcresume.strip())  # 以'，。？！；'为句子分隔符分割句子
    return [sentence for sentence in sentences if sentence]
