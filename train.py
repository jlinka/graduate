#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/22 下午4:49
# @Author  : jlinka
# @File    : train.py

# 用来训练整一个项目所需要的所有文件

import config
import os
from script.core import splitword
from script.core.model.wordvec_model import WordVecModel

from script.core.model.sentencerec_model import SentenceRecModel
from script.core.model import sentencerec_pre as srpre

# from script.core.model.wordrec_model import WordRecModel
# from script.core.model import wordrec_pre as wrpre

# # # 把中文语料库和原始简历给分词
# splitword.wikisplit2word()
#
# filenames = os.listdir(config.SRCDATA_DIC)
# save_filepaths = []
# if len(filenames) > 0:
#     save_filepaths = [config.PREDATA_DIC + '/' + filename for filename in filenames]
#
#     for filename in filenames:
#         splitword.othersplit2word(config.SRCDATA_DIC + '/' + filename)
#
# # 训练词向量模型
# wordvec_model = WordVecModel()
# wordvec_model.train_and_save_wordvec_model()
# wordvec_model.train_more(save_filepaths)

# 训练特征句分类模型

# 数据预处理
filenames = os.listdir(config.SR_DIC+'/data')
tag_filepaths = [config.SR_DIC + '/data/' + filename for filename in filenames]

srpre.deal_tagdata(tag_filepaths)

# 训练并测试模型
sr_model = SentenceRecModel()
sr_model.train()
sr_model.test()


# # 数据预处理
# filenames = os.listdir (config.WR_DIC)
# tag_filepaths = [config.WR_DIC + '/' + filename for filename in filenames]
#
# wrpre.deal_tagdata (tag_filepaths)
# # 训练命名实体识别模型
#
# wr_model = WordRecModel ()
# wr_model.train ()
# wr_model.test ()


