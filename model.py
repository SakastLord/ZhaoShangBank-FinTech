import tensorflow as tf
from params import *
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import time
from sklearn import tree
from sklearn.svm import SVC

BETA_REG = 0.0001
KEEP_PROB = 0.5
L_reg = 1
THRESHOLD = 0.5
LR = 0.0001


def use_thserod(item):
    if item[1] > THRESHOLD:
        return 1
    else:
        return 0


def generate_fc_weight(shape, name, trainable=True):
    threshold = 1.0 / np.sqrt(shape[0])
    weight_matrix = tf.random_uniform(shape, minval=-threshold, maxval=threshold)
    weight = tf.Variable(weight_matrix, name=name, trainable=trainable)
    return weight


def generate_fc_bias(shape, name, trainable=True):
    bias_distribution = tf.constant(0.01, shape=shape)
    bias = tf.Variable(bias_distribution, name=name, trainable=trainable)
    return bias


class Network(object):

    def __init__(self, sess, n_input):
        if MODEL == 'net':
            self.sess = sess
            self.dim_s = n_input
            # build_net
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, n_input], name='input_layer')

            self.truth = tf.placeholder(dtype=tf.int32, shape=[None, ], name='output_truth')
            self.truth_one_hot = tf.one_hot(self.truth, 2, dtype=tf.float32)

            self.keep_probiliaty = tf.placeholder(tf.float32, name='keep_prob')

            self.OPT = tf.train.AdamOptimizer(learning_rate=LR)

            # layer one
            self.weight_one = generate_fc_weight(shape=[n_input, 128], name='weight_1')
            self.bias_one = generate_fc_bias(shape=[128], name='bias_1')
            self.layer_1 = tf.nn.sigmoid(tf.matmul(self.input, self.weight_one) + self.bias_one)

            # drop out layer
            self.drop_layer = tf.nn.dropout(self.layer_1, keep_prob=self.keep_probiliaty)  # keep_prob  神经元被激活的概率

            # layer two_a
            self.weight_two = generate_fc_weight(shape=[128, 64], name='weight_2')
            self.bias_two = generate_fc_bias(shape=[64], name='bias_2')
            self.layer_2 = tf.nn.sigmoid(tf.matmul(self.drop_layer, self.weight_two) + self.bias_two)

            # layer output
            self.weight_three = generate_fc_weight(shape=[64, 2], name='weight_3')
            self.bias_three = generate_fc_bias(shape=[2], name='bias_3')
            logits = tf.matmul(self.layer_2, self.weight_three) + self.bias_three
            self.output = tf.nn.softmax(logits)

            # params
            self.params = [
                self.weight_one, self.bias_one,
                self.weight_two, self.bias_two,
                self.weight_three, self.bias_three
                          ]

            # prepare loss
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.truth_one_hot))

            l1_reg = tf.contrib.layers.l1_regularizer(0.5)
            l2_reg = tf.contrib.layers.l2_regularizer(0.5)

            self.reg_loss_one = tf.contrib.layers.apply_regularization(regularizer=l1_reg, weights_list=self.params)

            self.reg_loss_two = tf.contrib.layers.apply_regularization(regularizer=l2_reg, weights_list=self.params)

            self.reg_loss = BETA_REG * (self.reg_loss_two)

            self.loss = tf.reduce_mean(self.cross_entropy_loss + self.reg_loss)

            self.grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.loss, self.params)]

            self.train_op = self.OPT.apply_gradients(list(zip(self.grads, self.params)))

            # prepare saver
            self.saver = tf.train.Saver()

        elif MODEL == 'tree':
            self.model = XGBClassifier(
                booster='gbtree',
                scale_pos_weight=10,
                n_estimatores=100,
                max_depth=8,
                learning_rate=0.05,
                objective='binary:logitraw',
                gamma=0.2,
                reg_lambda=1,
                silent=0,
                colsample_bytree=0.6
                # colsample_btree = 0.8,
                # nthread = -1,
            )


    def train(self, data_x, data_y):
        if MODEL == 'net':
            dict = {self.input: data_x,
                    self.truth: data_y,
                    self.keep_probiliaty: KEEP_PROB
                    }
            self.sess.run(self.train_op, feed_dict=dict)
        else:
            self.model.fit(data_x, data_y)


    def inference(self, data_x):
        if MODEL == 'net':
            dict = {
                self.input: data_x,
                self.keep_probiliaty: 1.0
            }
            result_net = self.sess.run(self.output, feed_dict=dict).tolist()
            result = [use_thserod(item) for item in result_net]
        else:
            result = self.model.predict(data_x)
        return result

    def save_current_model(self, f_score_train, f_score_test):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        model_xgboost_name = current_time + "_Train:" + str(f_score_train) + "_Test:" + str(f_score_test) + '_xgboost.m'
        model_tree_name = current_time + "_Train:" + str(f_score_train) + "_Test:" + str(f_score_test) + '_tree.m'
        joblib.dump(self.model_one, MODEL_SAVE_PATH + model_xgboost_name)
        joblib.dump(self.model_two, MODEL_SAVE_PATH + model_tree_name)
        return





