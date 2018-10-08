#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 上午10:30
# @Author  : jlinka
# @File    : test_file.py

import config
import os
import mongoColletcion
from script.tool import __splitsentence, __bigfile
import csv
import jieba
import pandas as pd

# num = 1
# client, db, collection = mongoColletcion.connection('graduate', 'news')
# # news = collection.find({}).limit(2000)
# news = collection.find({})
# flag = 0
# splitsentencefile = 'file' + str(flag) + '.csv'
# for data in news:
#     print(str(num) + str(data))
#     if (num-1) % 400 == 0:
#         flag += 1
#         splitsentencefile = 'file' + str(flag) + '.csv'
#     with open(splitsentencefile, 'a', encoding='UTF-8-sig') as wf1:
#         data_list = __splitsentence.resume2sentences(data['content'])
#         for i in data_list:
#             wf1.write(i + '\n')
#         wf1.write(";;;;;;;;;;" + '\n')
#
#     num += 1

# # 分句code，训练句向量presentence
# client, db, collection = mongoColletcion.connection('graduate', 'news')
# # news = collection.find({}).limit(2000)
# news = collection.find({}).limit(200)
# flag = 0
# splitsentencefile = 'sentence100' + '.csv'
# with open(splitsentencefile, 'a', encoding='UTF-8-sig') as wf1:
#     for data in news:
#         data_list = __splitsentence.resume2sentences(data['content'])
#         for i in data_list:
#             print(i)
#             wf1.write(i + '\n')


# from gensim.models import Doc2Vec
#
# # doc2vec parameters
# vector_size = 300  # 300维
# window_size = 15
# min_count = 1
# sampling_threshold = 1e-5
# negative_size = 5
# train_epoch = 100
# dm = 0  # 0 = dbow; 1 = dmpv
# worker_count = 8  # number of parallel processes
#
# # input corpus
# train_corpus = "../train_data/train_docs.txt"
#
# def train(run_dir):
#     # 训练Doc2Vec，并保存模型
#     docs = gensim.models.doc2vec.TaggedLineDocument(train_corpus)
#     '''
#     dm: 训练算法：默认为1，指DM；dm=0，则使用DBOW
#     dm_mean：当使用DM训练算法时，对上下文向量相加（默认为0）；若设为1，则求均值
#     dm_concat：默认为0，当设为1时，在使用DM训练算法时，直接将上下文向量和Doc向量拼接
#     dbow_words：当设为1时，则在训练doc_vector（DBOW) 的同时训练word_vector; 默认为0，只训练doc_vector，速度更快
#     '''
#     model = Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
#                     workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
#                     iter=train_epoch)
#     model.save(os.path.join(run_dir, saved_model))  # save dov2vec
#     model.wv.save_word2vec_format(os.path.join(run_dir, word_vector_path), binary=False)  # save word2vec


# client, db, collection = mongoColletcion.connection('graduate', 'news')
# news = collection.find({})
# flag = 0
# splitsentencefile = 'file' + str(flag) + '.txt'
#
# with open(splitsentencefile, 'a', encoding='UTF-8-sig') as wf1:
#     for data in news:
#         data_list = __splitsentence.resume2sentences(data['content'])
#         for i in data_list:
#             wf1.write(str(jieba.lcut(i)).rstrip("]").lstrip("[").replace(",", "").replace("'", "") + '\n')
#             print(str(jieba.lcut(i)).rstrip("]").lstrip("[").replace(",", "").replace("'", "") + '\n')

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

