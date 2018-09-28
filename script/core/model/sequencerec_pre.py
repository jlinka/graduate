#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/18 上午11:53
# @Author  : jlinka
# @File    : sequencerec_pre.py

# 命名实体识别模型预处理

import config
import random
import numpy as np
import jieba
# from script.tool.pynlpir_holder import pynlpir
import os
from script.tool import __bigfile

from script.tool.__wordvecmodel_holder import wordvec_model

#################################################################################
# total_labels = ['B-name', 'I-name', 'E-name',
#                 'B-sex', 'I-sex', 'E-sex',
#                 'B-nationality', 'I-nationality', 'E-nationality',
#                 'B-nation', 'I-nation', 'E-nation',
#                 'B-time', 'I-time', 'E-time',
#                 'B-school', 'I-school', 'E-school',
#                 'B-college', 'I-college', 'E-college',
#                 'B-pro', 'I-pro', 'E-pro',
#                 'B-degree', 'I-degree', 'E-degree',
#                 'B-edu', 'I-edu', 'E-edu',
#                 'B-company', 'I-company', 'E-company',
#                 'B-department', 'I-department', 'E-department'
#                                                 'B-job', 'I-job', 'E-job',
#                 'O']

total_labels = [0, 1]


# 获取标签总数
def get_labels_count():
    return len(total_labels)


# 根据标签获取index
def get_label_index(label: str):
    return total_labels.index(label)


# 根据index获取标签
def get_label(index: int):
    return total_labels[index]


# 用于模型预测 将预测的输出值转成标签列表
def indexs2strs(indexs_list: list):
    labels_list = []

    for indexs in indexs_list:
        labels = []

        for index in indexs:
            labels.append(get_label(index))

        labels_list.append(labels)

    return labels_list


#################################################################################
# 批量获取训练数据
def get_batch_traindata(batch_size: int):
    strain_x = np.load(config.PREDATA_DIC + '/wtrain_x.npy')
    strain_y = np.load(config.PREDATA_DIC + '/wtrain_y.npy')

    strain_x, strain_y = __shuffle_both(strain_x, strain_y)  # 打乱数据

    total_size = len(strain_x)
    start = 0
    while start + batch_size < total_size:
        yield strain_x[start:start + batch_size], strain_y[start:start + batch_size]
        start += batch_size
    if len(strain_x[start:]) > 0:
        yield strain_x[start:], strain_y[start:]


# 批量获取测试数据
def get_batch_testdata(batch_size: int):
    stest_x = np.load(config.PREDATA_DIC + '/wtest_x.npy')
    stest_y = np.load(config.PREDATA_DIC + '/wtest_y.npy')

    stest_x, stest_y = __shuffle_both(stest_x, stest_y)  # 打乱数据

    total_size = len(stest_x)
    start = 0
    while start + batch_size < total_size:
        yield stest_x[start:start + batch_size], stest_y[start:start + batch_size]
        start += batch_size
    if len(stest_x[start:]) > 0:
        yield stest_x[start:], stest_y[start:]


# 获取所有的训练数据
def get_traindata():
    strain_x = np.load(config.PREDATA_DIC + '/wtrain_x.npy')
    strain_y = np.load(config.PREDATA_DIC + '/wtrain_y.npy')

    strain_x, strain_y = __shuffle_both(strain_x, strain_y)  # 打乱数据

    return strain_x, strain_y


# 获取所有的测试数据
def get_testdata():
    stest_x = np.load(config.PREDATA_DIC + '/wtest_x.npy')
    stest_y = np.load(config.PREDATA_DIC + '/wtest_y.npy')

    stest_x, stest_y = __shuffle_both(stest_x, stest_y)  # 打乱数据

    return stest_x, stest_y


# 删除处理好的训练数据
def remove_traindata():
    try:
        os.remove(config.PREDATA_DIC + '/wtrain_x.npy')
        os.remove(config.PREDATA_DIC + '/wtrain_y.npy')
    except:
        pass


# 删除处理好的测试数据
def remove_testdata():
    try:
        os.remove(config.PREDATA_DIC + '/wtest_x.npy')
        os.remove(config.PREDATA_DIC + '/wtest_y.npy')
    except:
        pass


# 删除处理好的数据
def remove_data():
    remove_traindata()
    remove_testdata()


#################################################################################
# 句子分词 返回处理好的词列表
def sentence2regwords(sentences: list):
    words_list = []

    for sentence in sentences:
        new_words = []

        sent_len = 0

        for word in jieba.lcut(sentence, HMM=True):
            # for word in pynlpir.segment(sentence, pos_tagging=False):
            if sent_len < config.SENTENCE_LEN:
                new_words.append(word)
            else:
                break
            sent_len += 1

        while sent_len < config.SENTENCE_LEN:
            new_words.append('。')
            sent_len += 1

        words_list.append(new_words)

    return words_list


# 句子分词
def sentence2words(sentences: list):
    words_list = []

    for sentence in sentences:
        words_list.append(jieba.lcut(sentence, HMM=True))
        # words_list.append(pynlpir.segment(sentence, pos_tagging=False))

    return words_list


#################################################################################
# 将标注数据分训练集和测试集，并将它们转成向量保存
def deal_tagdata(tagdata_filepaths: list, rate: float = config.WR_RATE):
    datas = []
    for tagdata_filepath in tagdata_filepaths:
        if os.path.exists(tagdata_filepath):
            for line in __bigfile.get_lines(tagdata_filepath):
                datas.append(line)
        else:
            raise FileNotFoundError('{} 标注数据文件不存在'.format(tagdata_filepath))

    words_list, labels_list = __split_data(datas)

    datas.clear()

    words_list, labels_list = __shuffle_both(words_list, labels_list)

    regwords_list, reglabels_list = __deal_data(words_list, labels_list)

    words_list.clear()
    labels_list.clear()

    wordvecs_list, labelvecs_list = __data2vec(regwords_list, reglabels_list)

    regwords_list.clear()
    reglabels_list.clear()

    # 将数据保存下来
    total_size = len(labelvecs_list)

    train_x = wordvecs_list[:int(total_size * rate)]
    train_y = labelvecs_list[:int(total_size * rate)]
    test_x = wordvecs_list[int(total_size * rate):]
    test_y = labelvecs_list[int(total_size * rate):]

    wordvecs_list.clear()
    labelvecs_list.clear()

    if rate == 1.0:
        # 特殊要求
        if len(train_x) > 0:
            np.save(config.PREDATA_DIC + '/wtrain_x.npy', np.array(train_x))
            np.save(config.PREDATA_DIC + '/wtrain_y.npy', np.array(train_y))
        else:
            raise ValueError('rate为1.0，但数据长度为0')
    elif rate == 0.0:
        # 特殊要求
        if len(test_x) > 0:
            np.save(config.PREDATA_DIC + '/wtest_x.npy', np.array(test_x))
            np.save(config.PREDATA_DIC + '/wtest_y.npy', np.array(test_y))
        else:
            raise ValueError('rate为0.0，但数据长度为0')
    elif rate > 0.0 and rate < 1.0:
        if len(train_x) <= 0 or len(test_x) <= 0:
            raise ValueError('数据长度为0')

        # 正常要求
        np.save(config.PREDATA_DIC + '/wtrain_x.npy', np.array(train_x))
        np.save(config.PREDATA_DIC + '/wtrain_y.npy', np.array(train_y))
        np.save(config.PREDATA_DIC + '/wtest_x.npy', np.array(test_x))
        np.save(config.PREDATA_DIC + '/wtest_y.npy', np.array(test_y))


# 切分数据
def __split_data(datas: list):
    words_list = []  # 保存分词后的句子 每一项是字符串： 词 词
    labels_list = []  # 保存标签 每一项是字符串： label label

    # 奇数行是分好词的句子，偶数行是对应的词标签
    temp = True  # 判断是奇数行还是偶数行 True 为奇数行
    words = []
    labels = []

    for line in datas:
        if line:
            if temp:
                words = line.strip('\n').split(' ')
                # 要判断是否乱行
                temp = False
            else:
                labels = line.strip('\n').split(' ')
                temp = True

                if len(words) == len(labels) and len(words) != 0 and words[0] not in total_labels \
                        and labels[0] in total_labels:
                    words_list.append(words)
                    labels_list.append(labels)
                else:
                    print('错误数据：{} {}'.format(words, labels))

    return words_list, labels_list  # 返回 分词后的句子列表，表示该句句子的标签列表


# 处理数据 将数据规则化
def __deal_data(words_list: list, labels_list: list):
    new_words_list = __deal_words(words_list)
    new_labels_list = __deal_labels(labels_list)

    return new_words_list, new_labels_list


# 处理词序列 返回规则化的词序列列表
def __deal_words(words_list: list):
    new_words_list = []

    for words in words_list:
        new_words = []

        sent_len = 0
        for word in words:
            if sent_len < config.SENTENCE_LEN:
                new_words.append(word)
            else:
                break
            sent_len += 1

        while sent_len < config.SENTENCE_LEN:
            new_words.append('。')
            sent_len += 1

        new_words_list.append(new_words)

    return new_words_list


# 处理标签 返回规则化的标签列表
def __deal_labels(labels_list: list):
    new_labels_list = []

    for labels in labels_list:
        new_labels = []

        sent_len = 0
        for label in labels:
            if sent_len < config.SENTENCE_LEN:
                new_labels.append(label)
            else:
                break
            sent_len += 1

        while sent_len < config.SENTENCE_LEN:
            new_labels.append(total_labels[len(total_labels) - 1])
            sent_len += 1

        new_labels_list.append(new_labels)

    return new_labels_list


# 数据转成词向量序列
def __data2vec(words_list: list, labels_list: list):
    print('开始将数据转成向量')

    wordvecs_list = __words2vecs(words_list)  # 句子向量列表
    labelvecs_list = __labels2vecs(labels_list)  # 标签向量列表

    print('数据转成向量结束')

    return wordvecs_list, labelvecs_list


# 句子转成词向量序列
def __words2vecs(words_list: list):
    wordvecs_list = []  # 词向量列表的列表

    for words in words_list:
        wordvecs = []

        for word in words:
            try:
                wordvecs.append(wordvec_model[word])
            except:
                wordvecs.append(wordvec_model['。'])

        wordvecs_list.append(wordvecs)

    return wordvecs_list


# 标签转成one-hot向量
def __labels2vecs(labels_list: list):
    labelvecs_list = []  # 标签向量列表

    for labels in labels_list:
        labelvecs = []

        for label in labels:
            labelvecs.append(get_label_index(label))

        labelvecs_list.append(labelvecs)

    return labelvecs_list


#################################################################################

# 打乱数据
def __shuffle_both(x: list, y: list):
    both = list(zip(x, y))
    random.shuffle(both)
    shuffle_x, shuffle_y = zip(*both)

    return list(shuffle_x), list(shuffle_y)


#################################################################################

if __name__ == '__main__':
    deal_tagdata([config.TAG_DIC + '/wr/tag_wr.txt'])
