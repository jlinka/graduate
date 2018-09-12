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


# filenames = os.listdir(config.SR_DIC + '/data')
# tagdata_filepaths = [config.SR_DIC + '/data/' + filename for filename in filenames]
# with open('split_text.txt', 'a', encoding='UTF-8-sig') as wf1:
#     for tagdata_filepath in tagdata_filepaths:
#         if os.path.exists(tagdata_filepath):
#             for line in __bigfile.get_lines(tagdata_filepath):
#                 a = jieba.cut(line)
#                 for i in a:
#                     wf1.write(i + ' ')


a = '﻿1,7月1日，SaaS创业公司纷享销客获得1亿美元D轮融资，此轮融资由新投资机构联合前三轮投资机构IDG资本、北极光创投、DCM创投共同完成，这是该公司一年内的第三次融资'.encode('utf-8').decode(
    'utf-8-sig')
b = a.strip('\n').lstrip(' ').lstrip('0,').lstrip('1,')
c = [b[0:1], b.strip('\n').lstrip(' ').lstrip('0,').lstrip('1,')]
print(b)
