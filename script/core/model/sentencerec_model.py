#!/usr/bin/python 
# -*- coding: utf-8 -*-

# 句子模型训练、测试

import config
import json
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from tensorflow import keras
from script.core.model import sentencerec_pre as srpre


# 训练、测试、获取模型
class SentenceRecModel:

    def __init__(self,
                 sentence_len: int = config.SENTENCE_LEN,
                 wordvec_size: int = config.WORDVEC_SIZE,
                 classes: int = srpre.get_labels_count(),
                 study_rate: float = config.SR_STUDY_RATE,
                 epochs: int = config.SR_EPOCHS,
                 batch_size: int = config.SR_BATCH_SIZE
                 ):
        self.sentence_len = sentence_len
        self.wordvec_size = wordvec_size
        self.classes = classes
        self.study_rate = study_rate
        self.epochs = epochs
        self.batch_size = batch_size

    # 获取模型框架（未加载数据）
    def _get_model(self):
        graph = tf.Graph()
        with graph.as_default():
            ph_x = tf.placeholder(dtype=tf.float32, shape=[None, self.sentence_len,
                                                           self.wordvec_size])  # shape(bactch_size,sentence_len,wordvec_size)
            ph_y = tf.placeholder(dtype=tf.float32,
                                  shape=[None, self.classes])  # shape(bactch_size,classifi_size)

            bigru = keras.layers.Bidirectional(
                keras.layers.GRU(400, return_sequences=False, dropout=0.5))(ph_x)
            outputs = keras.layers.Dense(self.classes, activation=tf.nn.softmax)(bigru)

            cross = keras.losses.categorical_crossentropy(y_true=ph_y, y_pred=outputs)
            loss = tf.reduce_mean(cross)

            train_opt = tf.train.AdamOptimizer(self.study_rate).minimize(loss)

            correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(ph_y, 1))  # 计算每一行的最大值是否相等，返回一个和ph_y形状一样的值
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 正确率

            return graph, ph_x, ph_y, loss, train_opt, accuracy, outputs

    # 训练模型
    def train(self, continue_train: bool = False):

        graph, ph_x, ph_y, loss, train_opt, accuracy, pred = self._get_model()

        with tf.Session(graph=graph) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(max_to_keep=1)
            if continue_train:
                saver.restore(sess, config.MODEL_DIC + '/sentencerec.ckpt')

            for epoch in range(self.epochs):
                step = 0
                for train_x, train_y in srpre.get_batch_traindata(self.batch_size):
                    _, train_loss, train_acc = sess.run([train_opt, loss, accuracy],
                                                        feed_dict={ph_x: train_x, ph_y: train_y})
                    print('epoch:{} step:{} loss:{} acc:{}'.format(epoch + 1, step + 1, train_loss, train_acc))
                    step += 1

                for test_x, test_y in srpre.get_batch_testdata(self.batch_size):
                    test_loss, test_acc = sess.run([loss, accuracy],
                                                   feed_dict={ph_x: test_x, ph_y: test_y})
                    print('epoch:{} step:{} val_loss:{} val_acc:{}'.format(epoch + 1, step + 1, test_loss, test_acc))
                    step += 1

            saver.save(sess, config.MODEL_DIC + '/sentencerec.ckpt')

    # 测试模型
    def test(self):

        graph, ph_x, ph_y, loss, train_opt, accuracy, pred = self._get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, config.MODEL_DIC + '/sentencerec.ckpt')

            labels = [key for key in range(self.classes)]
            target_names = ['0', '1']  # 标签列表

            total_test_pred_y = []
            total_test_y = []

            # 防止测试数据过大
            for test_x, test_y in srpre.get_batch_testdata(self.batch_size):

                test_pred_y = sess.run(pred, feed_dict={ph_x: test_x, ph_y: test_y})  # 得到测试数据的所有预测值

                for value in test_pred_y:
                    total_test_pred_y.append(value)

                for value in test_y:
                    total_test_y.append(value)

            # test_pred_y shape(batch size , classes )
            # test_y shape(batch size ,classes )

            # 用于自学习
            if config.SR_USE_SELFTRAIN:
                # find threshold 找到阀值
                thresholds = self.__find_thresholds(total_test_pred_y, total_test_y)
                # set threshold 设置阀值
                self.__set_thresholds(thresholds)

            # 计算每个标签的准确率、召回率、f值
            print(classification_report(np.argmax(total_test_y, 1), np.argmax(total_test_pred_y, 1), labels=labels,
                                        target_names=target_names))

    # 预测单个句子 输入一个句子处理好的序列 输出one-hot向量
    def predict(self, inputs):
        graph, ph_x, _, _, _, _, pred = self._get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, config.MODEL_DIC + '/sentencerec.ckpt')

            pred_y = sess.run(pred, feed_dict={ph_x: inputs})

            return pred_y

    # 预测多个句子 输入一个迭代器 输出one-hot向量
    def predict_generator(self, generator):
        graph, ph_x, _, _, _, _, pred = self._get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, config.MODEL_DIC + '/sentencerec.ckpt')

            for inputs in generator:
                pred_y = sess.run(pred, feed_dict={ph_x: inputs})

                yield pred_y

    #  找到各类标签的阀值 用于自学习
    def __find_thresholds(self, pred, y):
        # 计算预测是否正确
        true_false = np.equal(np.argmax(y, 1), np.argmax(pred, 1))

        ############################################
        # 创建字典 保存所有标签的正确值和错误值
        interval_total = {'ctime': {}, 'ptime': {}, 'basic': {}, 'wexp': {}, 'sexp': {}, 'noinfo': {}}
        for key in interval_total.keys():
            interval_total[key]['true'] = {}
            interval_total[key]['false'] = {}

            for i in range(20):
                interval_total[key]['true'][i] = []
                interval_total[key]['false'][i] = []
            # for end
        ############################################
        for index in range(len(true_false)):
            # 找到一个标签中的最大值的索引值
            maxindex = np.argmax(pred[index])
            # 找到一个标签中的最大值
            maxvalue = pred[index][maxindex]

            label = None
            equal = None

            # 获取标签
            label = srpre.total_labels[maxindex]

            # 判断是正确的还是错误的
            if true_false[index]:
                equal = 'true'
            else:
                equal = 'false'

            # 把数据添加到字典中
            if maxvalue >= 0.0 and maxvalue < 0.05:
                interval_total[label][equal][0].append(maxvalue)
            elif maxvalue >= 0.05 and maxvalue < 0.1:
                interval_total[label][equal][1].append(maxvalue)
            elif maxvalue >= 0.1 and maxvalue < 0.15:
                interval_total[label][equal][2].append(maxvalue)
            elif maxvalue >= 0.15 and maxvalue < 0.2:
                interval_total[label][equal][3].append(maxvalue)
            elif maxvalue >= 0.2 and maxvalue < 0.25:
                interval_total[label][equal][4].append(maxvalue)
            elif maxvalue >= 0.25 and maxvalue < 0.3:
                interval_total[label][equal][5].append(maxvalue)
            elif maxvalue >= 0.3 and maxvalue < 0.35:
                interval_total[label][equal][6].append(maxvalue)
            elif maxvalue >= 0.35 and maxvalue < 0.4:
                interval_total[label][equal][7].append(maxvalue)
            elif maxvalue >= 0.4 and maxvalue < 0.45:
                interval_total[label][equal][8].append(maxvalue)
            elif maxvalue >= 0.45 and maxvalue < 0.5:
                interval_total[label][equal][9].append(maxvalue)
            elif maxvalue >= 0.5 and maxvalue < 0.55:
                interval_total[label][equal][10].append(maxvalue)
            elif maxvalue >= 0.55 and maxvalue < 0.6:
                interval_total[label][equal][11].append(maxvalue)
            elif maxvalue >= 0.6 and maxvalue < 0.65:
                interval_total[label][equal][12].append(maxvalue)
            elif maxvalue >= 0.65 and maxvalue < 0.7:
                interval_total[label][equal][13].append(maxvalue)
            elif maxvalue >= 0.7 and maxvalue < 0.75:
                interval_total[label][equal][14].append(maxvalue)
            elif maxvalue >= 0.75 and maxvalue < 0.8:
                interval_total[label][equal][15].append(maxvalue)
            elif maxvalue >= 0.8 and maxvalue < 0.85:
                interval_total[label][equal][16].append(maxvalue)
            elif maxvalue >= 0.85 and maxvalue < 0.9:
                interval_total[label][equal][17].append(maxvalue)
            elif maxvalue >= 0.9 and maxvalue < 0.95:
                interval_total[label][equal][18].append(maxvalue)
            elif maxvalue >= 0.95 and maxvalue < 1.0:
                interval_total[label][equal][19].append(maxvalue)
            # for end

        # 存储阀值
        thresholds = {'ctime': 0, 'ptime': 0, 'basic': 0, 'wexp': 0, 'sexp': 0, 'noinfo': 0}

        # 用来保存每个标签的所有区间的预判正确数量和预判错误数量
        true_false_counts = {'ctime': {'true': [], 'false': []}, 'ptime': {'true': [], 'false': []},
                             'basic': {'true': [], 'false': []}, 'wexp': {'true': [], 'false': []},
                             'sexp': {'true': [], 'false': []}, 'noinfo': {'true': [], 'false': []}}

        tmp_list = {}

        # 把选用不同阀值的错误集、正确集给添加到字典中
        for label in true_false_counts.keys():
            for i in range(20):
                value = 0
                for j in range(i, 20):
                    value += len(interval_total[label]['true'][j])
                true_false_counts[label]['true'].append(value)

            for i in range(20):
                value = 0
                for j in range(i, 20):
                    value += len(interval_total[label]['false'][j])
                true_false_counts[label]['false'].append(value)
            # for end

        # 计算不同阀值的 错误集/总错误集(越低) 1-正确集/总正确集(越高) 比例
        for label in true_false_counts.keys():
            print('label {}'.format(label))
            tmp_list[label] = []
            # print ('true')
            temp = 0
            for i in range(20):
                true_percent = 0
                if true_false_counts[label]['true'][0] > 0:
                    true_percent = true_false_counts[label]['true'][i] / true_false_counts[label]['true'][0]

                false_percent = 0
                if true_false_counts[label]['false'][0] > 0:
                    false_percent = true_false_counts[label]['false'][i] / true_false_counts[label]['false'][0]

                avg = ((1 - true_percent) * 20 + false_percent) / 2

                if 1 - true_percent > 0.0 and 1 - true_percent <= 1.0:
                    tmp_list[label].append(
                        {'threshold': round(temp, 2), 'avg': avg})

                print('阀值:{} 1-正确集/总正确集比例:{} 引入错误集/总错误集比例:{} 加权平均值:{}'.format(round(temp, 2),
                                                                              round(1 - true_percent, 3),
                                                                              round(false_percent, 3),
                                                                              round(avg, 3)))
                temp += 0.05

            # 找到每个阀值的加权平均值的最小值
            min_index = -1
            tmp_avg = -1
            for index in range(len(tmp_list[label])):
                if tmp_avg == -1:
                    tmp_avg = tmp_list[label][index]['avg']
                    min_index = index
                else:
                    if tmp_avg > tmp_list[label][index]['avg']:
                        tmp_avg = tmp_list[label][index]['avg']
                        min_index = index
            # TODO BUG
            print('test', true_false_counts.keys())

            thresholds[label] = tmp_list[label][min_index]['threshold']
            # for end

        print()
        # 打印不同标签的各个区域的数量和平均值
        for label in interval_total.keys():
            print('label {}'.format(label))

            print('true')
            j = 0
            for i in range(20):
                count = len(interval_total[label]['true'][i])
                avg = 0
                if count > 0:
                    avg = sum(interval_total[label]['true'][i]) / count

                print('{}-{}: count:{} avg:{}'.format(round(j, 2), round(j + 0.05, 2),
                                                      count, avg))
                j += 0.05

            print('false')
            j = 0
            for i in range(20):
                count = len(interval_total[label]['false'][i])
                avg = 0
                if count > 0:
                    avg = sum(interval_total[label]['false'][i]) / count

                print('{}-{}: count:{} avg:{}'.format(round(j, 2), round(j + 0.05, 2),
                                                      count, avg))
                j += 0.05

            print()
            # for end

        # #返回阀值字典
        return thresholds

    # 设置阀值 用于自学习
    def __set_thresholds(self, thresholds):
        with open(config.PREDATA_DIC + '/thresholds.json', 'w', encoding='utf-8') as write_file:
            write_file.write(json.dumps(thresholds, ensure_ascii=False))


if __name__ == '__main__':
    # tag_filenames = os.listdir (config.SR_DIC)
    # tag_filepaths = [config.SR_DIC + '/' + tag_filename for tag_filename in tag_filenames if
    #                  tag_filename != '.DS_Store']
    # tag_filepaths=[config.SRCDATA_DIC+'/resume_data2.txt']
    # print (tag_filepaths)
    #
    # srpre.deal_tagdata (tag_filepaths, rate=0.05)

    SentenceRecModel().test()
