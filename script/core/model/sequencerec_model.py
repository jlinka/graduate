#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/18 上午10:33
# @Author  : jlinka
# @File    : sequencerec_model.py

import config
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow import keras
from sklearn.metrics import classification_report
from script.core.model import sequencerec_pre as wrpre


class WordRecModel:
    def __init__(self,
                 sentence_len: int = config.SENTENCE_LEN,
                 wordvec_size: int = config.WORDVEC_SIZE,
                 classes: int = wrpre.get_labels_count(),
                 study_rate: float = config.WR_STUDY_RATE,
                 epochs: int = config.WR_EPOCHS,
                 batch_size: int = config.WR_BATCH_SIZE
                 ):
        self.sentence_len = sentence_len
        self.wordvec_size = wordvec_size
        self.classes = classes
        self.study_rate = study_rate
        self.epochs = epochs
        self.batch_size = batch_size

    # 获取模型框架（未加载数据）
    def get_model(self):
        graph = tf.Graph()
        with graph.as_default():
            ph_x = tf.placeholder(dtype=tf.float32, shape=[None, self.sentence_len,
                                                           self.wordvec_size])  # shape(bactch_size,sentence_len,wordvec_size)

            ph_y = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len])  # shape(bactch_size,sentence_len)

            ph_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None, ])

            bigru = keras.layers.Bidirectional(
                keras.layers.GRU(256, return_sequences=True, dropout=0.5))(ph_x)
            bigru2 = keras.layers.Bidirectional(
                keras.layers.GRU(512, return_sequences=True, dropout=0.5))(bigru)

            w = tf.Variable(tf.random_normal(shape=[1024, self.classes]))

            bigru2 = tf.reshape(bigru2, shape=[-1, 1024])
            unary_scores = tf.matmul(bigru2, w)
            unary_scores = tf.reshape(unary_scores, shape=[-1, self.sentence_len, self.classes])

            log_likelihood, transition_params = crf.crf_log_likelihood(unary_scores, ph_y, ph_sequence_lengths)
            loss = tf.reduce_mean(-log_likelihood)

            viterbi_sequence, viterbi_score = crf.crf_decode(unary_scores, transition_params, ph_sequence_lengths)

            train_opt = tf.train.AdamOptimizer(self.study_rate).minimize(loss)

            correct_pred = tf.equal(viterbi_sequence, ph_y)

            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 正确率

            return graph, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, accuracy, viterbi_sequence

    def train(self, continue_train: bool = False):
        graph, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, accuracy, pred = self.get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(max_to_keep=1)
            if continue_train:
                saver.restore(sess, config.MODEL_DIC + '/wordrec.ckpt')

            for epoch in range(self.epochs):
                step = 0
                for train_x, train_y in wrpre.get_batch_traindata(self.batch_size):
                    lengths = [self.sentence_len for _ in range(len(train_y))]
                    sequence_lengths = np.array(lengths, dtype=np.int32)
                    _, train_loss, train_acc = sess.run([train_opt, loss, accuracy],
                                                        feed_dict={ph_sequence_lengths: sequence_lengths,
                                                                   ph_x: train_x, ph_y: train_y,
                                                                   })
                    print('batch size:{} epoch:{} step:{} loss:{} acc:{}'.format(self.batch_size, epoch + 1, step + 1,
                                                                                 train_loss, train_acc))
                    step += 1

                test_data_size = 0  # 测试集总批量
                total_loss = 0
                total_acc = 0  # 准确率总数
                test_step = 0  # 测试步数
                for test_x, test_y in wrpre.get_batch_testdata(self.batch_size):
                    current_batch_testdata_size = len(test_y)
                    lengths = [self.sentence_len for _ in range(current_batch_testdata_size)]
                    sequence_lengths = np.array(lengths, dtype=np.int32)
                    test_loss, test_acc = sess.run([loss, accuracy],
                                                   feed_dict={ph_sequence_lengths: sequence_lengths,
                                                              ph_x: test_x, ph_y: test_y})
                    test_data_size += current_batch_testdata_size
                    total_loss += test_loss
                    total_acc += test_acc
                    test_step += 1

                print(
                    'size:{} epoch:{} step:{} val_loss:{} val_acc:{}'.format(test_data_size, epoch + 1, step + 1,
                                                                             total_loss / test_step,
                                                                             total_acc / test_step))

            saver.save(sess, config.MODEL_DIC + '/wordrec.ckpt')

    def test(self):
        graph, ph_sequence_lengths, ph_x, ph_y, _, _, _, pred = self.get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, config.MODEL_DIC + '/wordrec.ckpt')

            # 标签列表
            labels = [key for key in range(self.classes)]
            target_names = wrpre.total_labels
            # print (labels, target_names)

            total_test_y = []
            total_test_pred_y = []

            for test_x, test_y in wrpre.get_batch_testdata(self.batch_size):
                lengths = [self.sentence_len for _ in range(len(test_y))]
                sequence_lengths = np.array(lengths, dtype=np.int32)

                test_pred_y = sess.run(pred, feed_dict={ph_sequence_lengths: sequence_lengths, ph_x: test_x,
                                                        ph_y: test_y})  # 得到测试数据的所有预测值

                total_test_y.append(np.reshape(test_y, (-1,)).tolist())
                total_test_pred_y.append(np.reshape(test_pred_y, (-1,)).tolist())

            print(classification_report(
                np.reshape(np.array(total_test_y), (-1,)),
                np.reshape(np.array(total_test_pred_y), (-1,)), target_names=target_names
            ))

            # print (classification_report (
            #     total_test_y, total_test_pred_y, target_names=target_names
            # ))

    def predict(self, inputs):
        graph, ph_sequence_lengths, ph_x, _, _, _, _, pred = self.get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, config.MODEL_DIC + '/wordrec.ckpt')

            lengths = [self.sentence_len for _ in range(len(inputs))]
            sequence_lengths = np.array(lengths, dtype=np.int32)

            pred_y = sess.run(pred, feed_dict={ph_sequence_lengths: sequence_lengths, ph_x: inputs})

            return pred_y

    def predict_generator(self, generator):
        graph, ph_sequence_lengths, ph_x, _, _, _, _, pred = self.get_model()

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, config.MODEL_DIC + '/wordrec.ckpt')

            for inputs in generator:
                lengths = [self.sentence_len for _ in range(len(inputs))]
                sequence_lengths = np.array(lengths, dtype=np.int32)

                pred_y = sess.run(pred, feed_dict={ph_sequence_lengths: sequence_lengths, ph_x: inputs})

                yield pred_y

    def __list2one(self, some: list):
        new_list = []
        for value in some:
            if value is list:
                new_list += self.__list2one(value)
            if value is np.array:
                new_list += self.__list2one(value.tolist())
            else:
                new_list.append(value)
        return new_list

# if __name__ == '__main__':
# wr_model = WordRecModel ()
# wr_model.train ()
# wr_model.get_model ()
# sequence_lengths = np.full (10, 20 - 1, dtype=np.int32)
# sequence_lengths_t = tf.constant (sequence_lengths)
# print(sequence_lengths)
# print(sequence_lengths_t)
