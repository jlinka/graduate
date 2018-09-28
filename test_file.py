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

# filenames = os.listdir(config.SR_DIC + '/data')
# tagdata_filepaths = [config.SR_DIC + '/data/' + filename for filename in filenames]
# with open('split_text.txt', 'a', encoding='UTF-8-sig') as wf1:
#     for tagdata_filepath in tagdata_filepaths:
#         if os.path.exists(tagdata_filepath):
#             for line in __bigfile.get_lines(tagdata_filepath):
#                 a = jieba.cut(line)
#                 for i in a:
#                     wf1.write(i + ' ')


# array = [1, 2, 3, 6, 5, 4]
# for i in range(len(array)):
#     for j in range(i):
#         if array[j] > array[j + 1]:
#             array[j], array[j + 1] = array[j + 1], array[j]
# print(array)
#
#
# #coding=utf-8
# # 本题为考试多行输入输出规范示例，无需提交，不计分。
# import sys
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans += v
#     print(ans)


# a = 7
# b = 5
# c = a ^ b
# print(c)

#
# import sys
#
# if __name__ == '__main__':
#     # n = int(sys.stdin.readline().strip())
#     line = sys.stdin.readline().strip()
#     value = list(map(int, line.split()))
#     list = [x+1 for x in range(value[0])]
#
#     for j in range(value[1]):
#         qizhi = sys.stdin.readline().strip()
#         test = []
#         test.append(int(qizhi))
#         list.remove(int(qizhi))
#         test.extend(list)
#         list = test
#
#     print(str(list).replace(',', '').lstrip("[").rstrip("]"))


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


def get_trainset():
    x_train = []  # 用来存放语料
    index = 0  # 每一篇博客需要一个对应的编号
    doc_dict = {}  # 由编号映射博客ID的字典
    for name in list_name:
        user_file = '/home/wayne/2017SMP/fenci2/testingcorpus/' + name
        # 语料预处理
        data = open(user_file).read()
        data = data.replace('\n', '').replace('  ', ' ')
        data = data.lower()
        words = data.split(" ")
        x_train.append(TaggededDocument(words, tags=[index]))
        doc_dict[index] = name.strip(".txt")
        print('append ok!')

        index += 1
    return x_train, doc_dict  # doc_dict的key和value 分别为编号和对应博客ID


def train(x_train, size=500, epoch_num=1):
    model_dm = gensim.models.Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5,
                                     workers=4)  # 模型的初始化，设置参数
    #  提供x_train可初始化, min_cout 忽略总频率低于这个的所有单词, window 预测的词与上下文词之间最大的距离, 用于预测  size 特征向量的维数 negative 接受杂质的个数 worker 工作模块数
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)  # corpus_count是文件个数  epochs 训练次数
    model_dm.save(config.MODEL_DIC + 'doc2vec.model')  # 保存模型训练结果，释放内存空间，后续可用load加载
    return model_dm
