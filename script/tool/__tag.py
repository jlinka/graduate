#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/22 下午3:57
# @Author  : jlinka
# @File    : __tag.py

# 用已有的模型来标注新的数据
import config
import json
import numpy as np
import os
import datetime
from script.tool import __bigfile, __splitsentence
from script.core.model import sentencerec_pre as srpre
from script.core.model.sentencerec_model import SentenceRecModel
# from script.core.model import wordrec_pre as wrpre
# from script.core.model.wordrec_model import WordRecModel


# 获取输入生成器
def __get_inputs_generator(filepath: str):
    for resume in __bigfile.get_lines(filepath):
        sentences = __splitsentence.resume2sentences(resume)
        words_list = srpre.sentence2regwords(sentences)
        yield srpre.sentence2vec(words_list)


# 获取句子生成器
def __get_sentences_generator(filepath: str):
    for resume in __bigfile.get_lines(filepath):
        yield __splitsentence.resume2sentences(resume)


# # 获取词生成器
# def __get_words_generator(filepath: str):
#     for resume in __bigfile.get_lines(filepath):
#         sentences = __splitsentence.resume2sentences(resume)
#         words_list = wrpre.sentence2words(sentences)
#         yield words_list


# 找到各个标签的阀值
def __load_thresholds():
    read_file = open(config.PREDATA_DIC + '/thresholds.json', 'r')
    json_str = read_file.read()
    thresholds = json.loads(json_str, encoding='utf-8')
    return thresholds


# 处理预测数据 用于自学习
def __deal_data(sentences, pred, thresholds):
    new_sentences = []
    new_pred = []

    add_label = {'ctime': 0, 'ptime': 0, 'basic': 0, 'wexp': 0, 'sexp': 0, 'noinfo': 0}
    remove_label = {'ctime': 0, 'ptime': 0, 'basic': 0, 'wexp': 0, 'sexp': 0, 'noinfo': 0}

    for sentence, p in zip(sentences, pred):
        max_index = np.argmax(p)
        label = srpre.total_labels[max_index]

        if label != 'ctime' or label != 'ptime':
            # 大于设定的阀值就认为是预测正确的
            if p[max_index] >= thresholds[label]:
                new_sentences.append(sentence)
                new_pred.append(p)
                add_label[label] += 1
            else:
                remove_label[label] += 1

    # __save_log (add_label, remove_label, filename)

    return new_sentences, new_pred


# 保存日志
def __save_log(add, remove, filename):
    with open(config.LOG_DIC + '/self_train.txt', 'a', encoding='utf-8') as write_file:
        json_str = json.dumps(
            {'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'add': add, 'remove': remove,
             'filename': filename},
            ensure_ascii=False)
        write_file.write(json_str + '\n')


# 标注特征句分类模型数据
def tag_sentence(filepaths: list):
    sr_model = SentenceRecModel()
    # 加载阀值

    for filepath in filepaths:
        if os.path.exists(filepath):
            filename = filepath.split('/')[-1]
            with open(config.TMP_SR_DIC + '/' + filename + '.tmp', 'w', encoding='utf-8') as write_file:
                for sentences, pred_y in zip(__get_sentences_generator(filepath),
                                             sr_model.predict_generator(__get_inputs_generator(filepath))):
                    # 设置是否使用自学习
                    if config.SR_USE_SELFTRAIN:
                        # 根据阀值来限制数据
                        thresholds = __load_thresholds()
                        sentences, pred_y = __deal_data(sentences, pred_y, thresholds)

                    labels = srpre.labelvec2str(pred_y)
                    for sentence, label in zip(sentences, labels):
                        print(sentence, label)
                        write_file.write(sentence + ';;;' + label + '\n')
                    write_file.flush()
        else:
            print('该文件不存在 {}'.format(filepath))


# # 用来标注命名实体词识别模型数据
# def tag_word(filepaths: list):
#     sr_model = WordRecModel()
#
#     for filepath in filepaths:
#         if os.path.exists(filepath):
#             filename = filepath.split('/')[-1]
#             with open(config.TMP_WR_DIC + '/' + filename + '.tmp', 'w', encoding='utf-8') as write_file:
#                 for words_list, pred_y in zip(__get_words_generator(filepath),
#                                               sr_model.predict_generator(__get_inputs_generator(filepath))):
#
#                     labels_list = wrpre.indexs2strs(pred_y)
#
#                     for words, labels in zip(words_list, labels_list):
#                         print(words, labels[:len(words)])
#                         write_file.write(' '.join(words) + '\n')
#                         write_file.write(' '.join(labels[:len(words)]) + '\n')
#                     write_file.flush()
#         else:
#             print('该文件不存在 {}'.format(filepath))


# 移除特征句标注数据文件
def remove_sentence_tagfile():
    # TODO
    filenames = os.listdir(config.TMP_SR_DIC)
    for filename in filenames:
        os.remove(config.TMP_SR_DIC + '/' + filename)


def remove_word_tagfile():
    # TODO
    filenames = os.listdir(config.TMP_WR_DIC)
    for filename in filenames:
        os.remove(config.TMP_SR_DIC + '/' + filename)


if __name__ == '__main__':
    # tag_word ([config.SRCDATA_DIC + '/resume_data1.txt'])
    tag_sentence([config.SRCDATA_DIC + '/resume_data101.txt'])
