#!/usr/bin/python 
# -*- coding: utf-8 -*-

# 预处理句子数据

# 将标注数据全部处理成词向量序列，将词向量序列分成训练集和测试集，最后将训练集和测试集保存到本地
# 批量获取训练集、批量获取测试集、一次获取所有训练集、一次获取所有测试集

import config
import random
import jieba
import json
import numpy as np
import os
import socket
import time
from script.tool import __bigfile, __splitsentence
from script.tool.__wordvecmodel_holder import wordvec_model

#################################################################################
# ctime：现在时间；ptime：过去时间；basic：基本信息；wexp：不含时间的工作信息句子；sexp：不含时间的学习信息句子；noinfo：不含任何有用信息句子
# total_labels = ['ctime', 'ptime', 'basic', 'wexp', 'sexp', 'noinfo']  # 标签列表
total_labels = ['0', '1']  # 标签列表


# 获取标签总数
def get_labels_count():
    return len(total_labels)


# 获取one-hot向量
def get_onehot_vec(label: str):
    vector = []

    if label in total_labels:
        for l in total_labels:
            if l == label:
                vector.append(1.0)
            else:
                vector.append(0.0)
        return vector
    else:
        raise ValueError('没有该标签：', label)


# 将标签向量转成标签字符串 labelvec_list 是（batch_size，classic_size）返回大小为batch_size的标签列表
# TODO bug
def labelvec2str(labelvec_list: list):
    label_list = []

    for index in np.argmax(labelvec_list, axis=1):
        label_list.append(total_labels[index])

    return label_list


#################################################################################
# 句子分词 输入句子列表 返回词序列列表
def sentence2words(sentences: list):
    words_list = []

    for sentence in sentences:
        words_list.append(jieba.lcut(sentence))

    return words_list


# 句子分词 返回处理好的词序列列表
def sentence2regwords(sentences: list):
    words_list = []

    for sentence in sentences:
        new_words = []

        sent_len = 0

        for word in jieba.lcut(sentence):
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


# 将句子转成词向量序列 输入词序列列表 返回词向量序列列表
def sentence2vec(words_list: list):
    return __sentence2vec(words_list)


#################################################################################
# 将标注数据分训练集和测试集，并将它们转成向量保存
def deal_tagdata(tagdata_filepaths: list, rate: float = config.SR_RATE):
    datas = []
    for tagdata_filepath in tagdata_filepaths:
        if os.path.exists(tagdata_filepath):
            for line in __bigfile.get_lines(tagdata_filepath):
                datas.append(line)
        else:
            raise FileNotFoundError('{} 标注数据文件不存在'.format(tagdata_filepath))

    random.shuffle(datas)  # 打乱数据

    sentences, labels = __split_tagdata(datas)

    datas.clear()

    words_list = __tagsentence2regwords(sentences)

    sentences.clear()

    sentencevec_list, labelvec_list = __data2vec(words_list, labels)

    words_list.clear()
    labels.clear()

    # 将数据保存下来
    total_size = len(labelvec_list)

    train_x = sentencevec_list[:int(total_size * rate)]
    train_y = labelvec_list[:int(total_size * rate)]
    test_x = sentencevec_list[int(total_size * rate):]
    test_y = labelvec_list[int(total_size * rate):]

    sentencevec_list.clear()
    labelvec_list.clear()

    if rate == 1.0:
        # 特殊要求
        if len(train_x) > 0:
            np.save(config.PREDATA_DIC + '/strain_x.npy', np.array(train_x))
            np.save(config.PREDATA_DIC + '/strain_y.npy', np.array(train_y))
        else:
            raise ValueError('rate为1.0，但数据长度为0')

    elif rate == 0.0:
        # 特殊要求
        if len(test_x) > 0:
            np.save(config.PREDATA_DIC + '/stest_x.npy', np.array(test_x))
            np.save(config.PREDATA_DIC + '/stest_y.npy', np.array(test_y))
        else:
            raise ValueError('rate为0.0，但数据长度为0')

    elif rate > 0.0 and rate < 1.0:
        train_size = len(train_x)
        test_size = len(test_x)

        if train_size <= 0 or test_size <= 0:
            raise ValueError('数据长度为0')

        # 正常要求
        np.save(config.PREDATA_DIC + '/strain_x.npy', np.array(train_x))
        np.save(config.PREDATA_DIC + '/strain_y.npy', np.array(train_y))
        np.save(config.PREDATA_DIC + '/stest_x.npy', np.array(test_x))
        np.save(config.PREDATA_DIC + '/stest_y.npy', np.array(test_y))

    else:
        raise ValueError('rate 超出范围，rate应该在0.0和1.0之间 rate:{}'.format(rate))


# 处理标注数据
def __split_tagdata(datas: list):
    sentences = []  # 保存分词后的句子
    label_list = []  # 保存标签

    for line in datas:
        if line:
            # pair = line.strip('\n').split(';;;')  # 将句子和标签分开
            line = line.encode('utf-8').decode('utf-8-sig')
            pair = [line[0:1], line.strip('\n').lstrip(' ').lstrip('0,').lstrip('1,')]
            # pair = line.strip('\n').split(',')
            if len(pair) == 2 and pair[0] and pair[1]:
                if pair[0] in total_labels:
                    sentence_words = jieba.lcut(pair[1])  # 将句子分词
                    if '，' in sentence_words:
                        sentence_words.remove('，')
                    sentences.append(sentence_words)
                    label_list.append(pair[0])
                else:
                    print('error line {}'.format(line))
                    if len(line) > 10:
                        print(line)
            else:
                print('error line {}'.format(line))
                if len(line) > 10:
                    print(line)
        else:
            print('error line {}'.format(line))
            if len(line) > 10:
                print(line)
    return sentences, label_list  # 返回 分词后的句子列表，表示该句句子的标签列表


# 用于处理标注数据
def __tagsentence2regwords(sentences: list):
    words_list = []

    for sentence in sentences:
        new_words = []

        sent_len = 0

        for word in sentence:
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


# 数据转成向量
def __data2vec(sentences: list, labels: list):
    print('开始将数据转成向量')

    sentencevec_list = __sentence2vec(sentences)  # 句子向量列表
    labelvec_list = __label2vec(labels)  # 标签向量列表

    print('数据转成向量结束')

    return sentencevec_list, labelvec_list


# 句子转成词向量序列
def __sentence2vec(words_list: list):
    sentencevec_list = []  # 句子向量列表

    for sentence in words_list:
        sentencevec = []
        for word in sentence:
            try:
                if word in wordvec_model.keys():
                    sentencevec.append(wordvec_model[word])
                # else:
                #     sentencevec.append(wordvec_model['。'])
            except:
                sentencevec.append(wordvec_model['。'])

        sentencevec_list.append(sentencevec)

    return sentencevec_list


# 标签转成one-hot向量
def __label2vec(labels: list):
    labelvec_list = []  # 标签向量列表

    for label in labels:
        labelvec = get_onehot_vec(label)
        labelvec_list.append(labelvec)

    return labelvec_list


#################################################################################
# 批量获取训练数据
def get_batch_traindata(batch_size: int):
    strain_x = np.load(config.PREDATA_DIC + '/strain_x.npy')
    strain_y = np.load(config.PREDATA_DIC + '/strain_y.npy')

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
    stest_x = np.load(config.PREDATA_DIC + '/stest_x.npy')
    stest_y = np.load(config.PREDATA_DIC + '/stest_y.npy')

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
    strain_x = np.load(config.PREDATA_DIC + '/strain_x.npy')
    strain_y = np.load(config.PREDATA_DIC + '/strain_y.npy')

    strain_x, strain_y = __shuffle_both(strain_x, strain_y)  # 打乱数据

    return strain_x, strain_y


# 获取所有的测试数据
def get_testdata():
    stest_x = np.load(config.PREDATA_DIC + '/stest_x.npy')
    stest_y = np.load(config.PREDATA_DIC + '/stest_y.npy')

    stest_x, stest_y = __shuffle_both(stest_x, stest_y)  # 打乱数据

    return stest_x, stest_y


# 删除处理好的训练数据
def remove_traindata():
    try:
        os.remove(config.PREDATA_DIC + '/strain_x.npy')
        os.remove(config.PREDATA_DIC + '/strain_y.npy')
    except:
        pass


# 删除处理好的测试数据
def remove_testdata():
    try:
        os.remove(config.PREDATA_DIC + '/stest_x.npy')
        os.remove(config.PREDATA_DIC + '/stest_y.npy')
    except:
        pass


# 删除处理好的数据
def remove_data():
    remove_traindata()
    remove_testdata()


# 打乱数据
def __shuffle_both(x: list, y: list):
    both = list(zip(x, y))
    random.shuffle(both)
    shuffle_x, shuffle_y = zip(*both)

    return shuffle_x, shuffle_y

#################################################################################
