import tensorflow as tf


class Simple_NN(object):
    def __init__(self,sess,n_input,learning_rate):
        self.sess = sess
        self.dim_s = n_input
        # build_net
        w_init = tf.random_normal_initializer(0., .1)
        self.keep_probiliaty = tf.placeholder(tf.float32)
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, n_input], name='input_layer')

        self.output_truth = tf.placeholder(dtype=tf.int32, shape=[None, ], name='output_truth')
        self.ground_truth = tf.one_hot(self.output_truth, 2, dtype=tf.float32)

        self.layer_1 = tf.layers.dense(self.input, units=128, activation=tf.nn.relu,
                                  kernel_initializer=w_init, name='l_1',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
        self.drop_layer = tf.nn.dropout(self.layer_1,keep_prob=self.keep_probiliaty)  # keep_prob  神经元被激活的概率
        self.output = tf.layers.dense(drop_layer, units=2, activation=tf.nn.softmax, kernel_initializer=w_init, name='l_output')

        # prepare loss
        self.loss_simple = tf.nn.softmax_cross_entropy_with_logits(logits=self.output , labels=self.ground_truth)
        self.loss = tf.reduce_mean(self.loss_simple)
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)




    def train(self,data_x,data_y,pro):
        dict = {self.input:data_x,self.output_truth:data_y,self.keep_probiliaty:pro}
        self.sess.run(self.train_op,feed_dict = dict)


    def inference(self,sess,data_x,pro):
        dict = {self.input:data_x,self.keep_probiliaty:pro}
        result = sess.run(self.output,feed_dict = dict)
        return result
