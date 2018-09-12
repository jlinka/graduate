#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/13 下午4:29
# @Author  : jlinka
# @File    : split.py


import config
import mongoColletcion
import word2vec
import types
import jieba
from pymongo import MongoClient
import datetime

class mongDbManger_web_showClass:
    __client = None
    __db = None
    __coll = None

    def __init__(self):
        print("__init__mongDbManger_webshow")

    def connect(self, url, port, user, password, database, table, auth):
        self.__client = MongoClient(url, port)
        self.__db = self.__client[auth]
        self.__db.authenticate(user, password)
        self.__db = self.__client[database]
        self.__coll = self.__db[table]

    def closeMongo(self):
        self.__client.close()

    def findOneWeek(self, dt):
        qureyCondition = {'createTime': {'$gt': dt}}
        cursor = self.__coll.find(qureyCondition)
        return cursor
if __name__ == '__main__':
    #word2vec.word2vec('split.txt', 'newsWord2Vec.bin', size=300,verbose=True)
    # model = word2vec.load('newsWord2Vec.bin')
    # indexes = model.cosine(u'百度')
    # for index in indexes[0]:
    #     print(model.vocab[index])
    # print(len(indexes))
    # a = 1.0
    # print(isinstance(a, (float, int)))


    jieba.load_userdict("company_word2.txt")

    splitfilename = 'split_comp5.txt'
    input = open(splitfilename, 'a', encoding='utf-8', errors='ignore')
    num = 1
    db = mongDbManger_web_showClass()
    db.connect('server21.raisound.com', 24000, "webuser", "webuser1957", "web_show", "xinChuang_topic", "web_show")
    # 获取300天前的时间戳
    now_time = datetime.datetime.now()
    yes_time = now_time + datetime.timedelta(days=-300)
    t = float(str(yes_time.strftime("%Y-%m-%d %H:%M:%S")).replace("-", "").replace(" ", "").replace(":", ""))
    cursor = db.findOneWeek(t)
    for item in cursor:
        if item['content']:
            data = jieba.lcut(item['content'])
            for i in data:
                input.write(i + ' ')
        print(str(num))
        num += 1
    # input = open('company_word2.txt', 'a', encoding='utf-8', errors='ignore')
    # stopwords = [line.strip() for line in open('company2.txt', 'r', encoding='utf-8').readlines()]
    # for i in stopwords:
    #     i = i.lstrip(" ").rstrip(" ").replace(" ", "")
    # alist = list(set(stopwords))
    # for i in alist:
    #     input.write(i + "\n")
    #
    #
    # print('a')






