#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/13 下午4:52
# @Author  : jlinka
# @File    : mongoColletcion.py

import config
from pymongo import MongoClient

def connection(database, table):
    db_name = database
    db_table = table
    client = MongoClient(config.MONGODB_IP, config.MONGODB_PORT)
    db = client[db_name]
    collection = db[db_table]


    return client, db, collection







