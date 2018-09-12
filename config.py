#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/13 下午4:33
# @Author  : jlinka
# @File    : config.py
from aip import AipNlp
# 配置文件

# 该文件一定要放在项目的根目录下
import os

# 目录路径
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录

SRCDATA_DIC = PROJECT_ROOT + '/file/srcdata'  # 简历源数据目录

PREDATA_DIC = PROJECT_ROOT + '/file/predata'  # 中间文件目录

CORPUS_DIC = PROJECT_ROOT + '/file/corpus'  # 文集文件目录

MODEL_DIC = PROJECT_ROOT + '/file/model'  # 模型文件目录

TAG_DIC = PROJECT_ROOT + '/file/tag'  # 标注数据文件目录

TMP_SR_DIC = TAG_DIC + '/tmp/sr'  # 未处理的特征句分类模型标注数据
TMP_WR_DIC = TAG_DIC + '/tmp/wr'  # 未处理的命名实体词识别模型标注数据

SR_DIC = TAG_DIC + '/sr'  # 处理的特征句分类模型标注数据
WR_DIC = TAG_DIC + '/wr'  # 处理的命名实体词识别模型标注数据

###############################################

# 模型配置

SENTENCE_LEN = 50  # 句子长度为50

WORDVEC_SIZE = 200  # 词向量维度 200

# 特征句分类模型参数

SR_RATE = 0.80  # 训练集占标注集的百分比

SR_STUDY_RATE = 0.001  # 学习率

SR_EPOCHS = 20  # 训练模型迭代次数

SR_BATCH_SIZE = 250  # 训练模型每次输入数据条数

SR_USE_SELFTRAIN = False  # 是否使用自学习

# 命名实体识别模型参数

WR_RATE = 0.80  # 训练集占标注集的百分比

WR_STUDY_RATE = 0.001  # 学习率

WR_EPOCHS = 10  # 训练模型迭代次数

WR_BATCH_SIZE = 250  # 训练模型每次输入数据条数

###############################################
WORDVEC_SIZE = 200

##################################################################
# MONGODB数据库配置
MONGODB_IP = '127.0.0.1'
MONGODB_PORT = 27017
MONGODB_USER = 'root'
MONGODB_PWD = 'root'
# 数据库名
DB_NAME = 'graduate'
# 表名
TABLE_NEWS = 'news'
##################################################################
# baidu-aip配置
""" 你的 APPID AK SK """
APP_ID = '11508916'
API_KEY = 'FegRIyeMcFxmrbp0435XjPGW'
SECRET_KEY = 'm9hO7Nu9qgf3SvrAsfvZrv9ETZMlHkGO'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

##################################################################
