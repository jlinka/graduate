#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/20 上午9:58
# @Author  : jlinka
# @File    : splitword.py

# 将语料库分词

import config
import os
import jieba
from script.tool import __bigfile


# 将中文wiki语料库分词，保存到predata目录下
def wikisplit2word():
    if os.path.exists(config.CORPUS_DIC + '/wiki_chs'):
        with open(config.PREDATA_DIC + '/totalpart.txt', 'a', encoding='utf-8') as write_file:
            print('开始分词')
            for line in __bigfile.get_lines(config.CORPUS_DIC + '/wiki_chs'):
                if line:
                    write_file.write(' '.join(jieba.lcut(line)))
            print('分词结束')
    else:
        raise FileNotFoundError('{} 不存在'.format(config.CORPUS_DIC + '/wiki_chs'))


def othersplit2word(filepath: str):
    if os.path.exists(filepath):

        with open(config.PREDATA_DIC + '/' + filepath.split('/')[-1], 'a', encoding='utf-8') as write_file:
            print('开始分词')
            for line in __bigfile.get_lines(filepath):
                if line:
                    write_file.write(' '.join(jieba.lcut(line)))
            print('分词结束')
    else:
        raise FileNotFoundError('{} 不存在'.format(filepath))

