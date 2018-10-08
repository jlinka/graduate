#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 下午5:36
# @Author  : jlinka
# @File    : doc2vec_model.py

# 句向量模型

import gensim
import config

def train(x_train, size=500, epoch_num=1):
    model_dm = gensim.models.Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5,
                                     workers=4)  # 模型的初始化，设置参数
    #  提供x_train可初始化, min_cout 忽略总频率低于这个的所有单词, window 预测的词与上下文词之间最大的距离, 用于预测  size 特征向量的维数 negative 接受杂质的个数 worker 工作模块数
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)  # corpus_count是文件个数  epochs 训练次数
    model_dm.save(config.MODEL_DIC + 'doc2vec.model')  # 保存模型训练结果，释放内存空间，后续可用load加载
    return model_dm
