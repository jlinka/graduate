#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 下午3:15
# @Author  : jlinka
# @File    : wordvec_model.py

# 训练、加载词向量模型

import config
import gensim


class WordVecModel:

    def __init__(self,
                 corpus_filepath: str = config.PREDATA_DIC + '/totalpart.txt',
                 wordvec_filepath: str = config.MODEL_DIC + '/wordvec.model',
                 wordvec_size: str = config.WORDVEC_SIZE):
        self.corpus_filepath = corpus_filepath
        self.wordvec_filepath = wordvec_filepath
        self.wordvec_size = wordvec_size

    def train_and_save_wordvec_model(self):
        print('开始训练词向量模型')
        sentences = gensim.models.word2vec.Text8Corpus(self.corpus_filepath)  # 加载分词后的文件
        model = gensim.models.Word2Vec(sentences, size=self.wordvec_size, window=5, min_count=1, workers=4)  # 训练词向量模型
        print('词向量模型训练结束')
        print('开始保存词向量模型')
        model.save(self.wordvec_filepath)  # 保存词向量模型
        print('保存词向量模型结束')

    def load_trained_wordvec_model(self):
        print('开始加载词向量模型')
        try:
            model = gensim.models.Word2Vec.load(self.wordvec_filepath)
            print('加载词向量模型结束')
            return model
        except:
            raise FileNotFoundError('{} 路径下没有词向量模型'.format(self.wordvec_filepath))

    # 训练更多的词向量
    def train_more(self, more_filepaths: list):
        model = self.load_trained_wordvec_model()
        print('开始训练词向量模型')
        for more_filepath in more_filepaths:
            sentences = gensim.models.word2vec.Text8Corpus(more_filepath)  # 加载分词后的文件
            model.train(sentences, epochs=model.iter, total_examples=model.corpus_count)
        print('词向量模型训练结束')
        print('开始保存词向量模型')
        model.save(self.wordvec_filepath)  # 保存词向量模型
        print('保存词向量模型结束')


if __name__ == '__main__':
    word2vec = WordVecModel()
    word2vec.train_more(['111.txt'])
