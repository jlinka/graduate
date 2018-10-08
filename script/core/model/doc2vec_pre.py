#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 下午5:37
# @Author  : jlinka
# @File    : doc2vec_pre.py

# 句向量模型数据预处理

import config
import os
import csv
import jieba
import gensim
import config

TaggededDocument = gensim.models.doc2vec.TaggedDocument  # 输入输出内容都为 词袋 + tag列表， 作用是记录每一篇博客的大致内容，并给该博客编号

list_name = os.listdir(config.SR_DIC + "/doc")  # 用于训练模型的语料先进行预处理


def getText():
    sentence = open('sentence100.csv', encoding='utf-8-sig')
    df_train = list(csv.reader(sentence))
    return df_train


def cut_sentence(text):
    stop_list = [line[:-1] for line in open(config.ST_DIC + '/停用词整合.txt', encoding='utf-8-sig')]
    result = []
    for each in text:
        each_cut = jieba.cut(each[0])
        each_split = ' '.join(each_cut).split()
        # each_result = []
        # for word in each_cut:
        #     if word not in stop_list:
        #         each_result.append(word)
        each_result = [word for word in each_split if word not in stop_list]
        result.append(' '.join(each_result))
    return result


def get_trainset(cut_sentence):
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = []
    for i, text in enumerate(cut_sentence):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


def train(x_train, size=500, epoch_num=1):
    model_dm = gensim.models.Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5,
                                     workers=4)  # 模型的初始化，设置参数
    #  提供x_train可初始化, min_cout 忽略总频率低于这个的所有单词, window 预测的词与上下文词之间最大的距离, 用于预测  size 特征向量的维数 negative 接受杂质的个数 worker 工作模块数
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)  # corpus_count是文件个数  epochs 训练次数
    model_dm.save(config.MODEL_DIC + 'doc2vec.model')  # 保存模型训练结果，释放内存空间，后续可用load加载
    return model_dm


if __name__ == '__main__':
    get_trainset(cut_sentence(getText()))
