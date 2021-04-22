"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import robustify_network
from tensorflow.python.training import py_checkpoint_reader


class robustCNN2(robustify_network.RobustifyNetwork):

    def __init__(self, config, num_features):
        super().__init__(config['num_classes'], config['epsilon'])

        initial_learning_rate = float(config['initial_learning_rate'])
        training_batch_size = config['training_batch_size']
        batch_decrease_learning_rate = float(config['batch_decrease_learning_rate'])

        self.num_features = num_features
        self.batch_size = training_batch_size

        self.mode = 'train'

        self.train_variables = []
        if num_features == 32*32*3: #CIFAR
            self.num_channels = 3
            self.feat_size = 8
            self.im_size = 32
        else: #MNIST
            self.num_channels = 1
            self.feat_size = 7
            self.im_size = 28

        # first convolutional layer
        self.W_conv1 = self._weight_variable([5, 5, self.num_channels, 32], name="Variable")
        self.train_variables += [self.W_conv1]
        self.b_conv1 = self._bias_variable([32], name="Variable_1")
        self.train_variables += [self.b_conv1]

        # second convolutional layer
        self.W_conv2 = self._weight_variable([5, 5, 32, 64], name="Variable_2")
        self.train_variables += [self.W_conv2]
        self.b_conv2 = self._bias_variable([64], name="Variable_3")
        self.train_variables += [self.b_conv2]

        # first fully connected layer
        self.W_fc1 = self._weight_variable([self.feat_size * self.feat_size * 64, 1024], name="Variable_4")
        self.train_variables += [self.W_fc1]
        self.b_fc1 = self._bias_variable([1024], name="Variable_5")
        self.train_variables += [self.b_fc1]

        # output layer
        self.W_fc2 = self._weight_variable([1024, 10], name="Variable_6")
        self.train_variables += [self.W_fc2]
        self.b_fc2 = self._bias_variable([10], name="Variable_7")
        self.train_variables += [self.b_fc2]

        # Setting up the optimizer
        self.learning_rate = \
          tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                         training_batch_size * batch_decrease_learning_rate, 0.85, staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def feedforward_pass(self, input):

        self.x_input_image = tf.reshape(input, [-1, self.im_size, self.im_size, self.num_channels])

        h1 = self._conv2d(self.x_input_image, self.W_conv1) + self.b_conv1
        self.h_conv1 = tf.nn.relu(h1)

        self.h_conv2 = tf.nn.relu(self._conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)
        h_pool2 = self._avg_pool_4x4(self.h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, self.feat_size * self.feat_size * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.pre_softmax = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        return self.pre_softmax

    def certificate_loss(self, epsilon, labels):
        self.eps_input1 =  tf.reshape(epsilon*tf.eye(self.num_features)[None] + self.x_input[:,None],
                                      [self.num_features* self.batch_size , self.num_features])
        self.eps_input2 =  tf.reshape(-epsilon*tf.eye(self.num_features)[None] + self.x_input[:,None],
                                      [self.num_features* self.batch_size , self.num_features])
        self.y_input1 = tf.cast(labels, tf.int32)
        self.eps_image1 = tf.reshape(self.eps_input1, [-1, self.im_size, self.im_size, self.num_channels])
        self.eps_image2 = tf.reshape(self.eps_input2, [-1, self.im_size, self.im_size, self.num_channels])
        labels_r = tf.repeat(labels , repeats = self.num_features, axis = 0)


        self.g_1_pos = self._conv2d(self.eps_image1, self.W_conv1) + self.b_conv1
        self.g_1_neg = self._conv2d(self.eps_image2, self.W_conv1) + self.b_conv1
        filt1 = tf.repeat(tf.sign(self.h_conv1),
                          repeats = self.num_features*tf.ones(self.batch_size, tf.int32), axis = 0)

        # second convolutional layer
        self.g_2_pos1 = self._conv2d(tf.nn.relu(self.g_1_pos), tf.nn.relu(self.W_conv2)) - \
                        self._conv2d(tf.multiply(self.g_1_pos, filt1), tf.nn.relu(-self.W_conv2))
        self.g_2_pos2 = self._conv2d(tf.nn.relu(self.g_1_pos), tf.nn.relu(-self.W_conv2)) - \
                        self._conv2d(tf.multiply(self.g_1_pos, filt1), tf.nn.relu(self.W_conv2))
        self.g_2_neg1 = self._conv2d(tf.nn.relu(self.g_1_neg), tf.nn.relu(self.W_conv2)) - \
                        self._conv2d(tf.multiply(self.g_1_neg, filt1), tf.nn.relu(-self.W_conv2))
        self.g_2_neg2 = self._conv2d(tf.nn.relu(self.g_1_neg), tf.nn.relu(-self.W_conv2)) - \
                        self._conv2d(tf.multiply(self.g_1_neg, filt1), tf.nn.relu(self.W_conv2))
        filt2 = tf.repeat(tf.sign(self.h_conv2),
                          repeats = self.num_features*tf.ones(self.batch_size, tf.int32), axis = 0)


        # first fully connected layer
        g_2_pos1_flat = tf.reshape(self._avg_pool_4x4(tf.nn.relu(self.g_2_pos1 + self.b_conv2)),
                                   [-1, self.feat_size * self.feat_size * 64])
        g_2_neg1_flat = tf.reshape(self._avg_pool_4x4(tf.nn.relu(self.g_2_neg1 + self.b_conv2)),
                                   [-1, self.feat_size * self.feat_size * 64])
        g_2_pos2_flat = tf.reshape(self._avg_pool_4x4(tf.multiply(-self.g_2_pos2 + self.b_conv2, filt2)),
                                   [-1, self.feat_size * self.feat_size * 64])
        g_2_neg2_flat = tf.reshape(self._avg_pool_4x4(tf.multiply(-self.g_2_neg2 + self.b_conv2, filt2)),
                                   [-1, self.feat_size * self.feat_size * 64])


        self.g_3_pos1 = tf.matmul(g_2_pos1_flat, tf.nn.relu(self.W_fc1)) - \
                        tf.matmul(g_2_pos2_flat, tf.nn.relu(-self.W_fc1))
        self.g_3_pos2 = tf.matmul(g_2_pos1_flat, tf.nn.relu(-self.W_fc1)) - \
                        tf.matmul(g_2_pos2_flat, tf.nn.relu(self.W_fc1))
        self.g_3_neg1 = tf.matmul(g_2_neg1_flat, tf.nn.relu(self.W_fc1)) - \
                        tf.matmul(g_2_neg2_flat, tf.nn.relu(-self.W_fc1))
        self.g_3_neg2 = tf.matmul(g_2_neg1_flat, tf.nn.relu(-self.W_fc1)) - \
                        tf.matmul(g_2_neg2_flat, tf.nn.relu(self.W_fc1))
        filt3 = tf.repeat(tf.sign(self.h_fc1),
                          repeats = self.num_features*tf.ones(self.batch_size, tf.int32), axis = 0)

        robust_objective = 0
        robust_acc = 0

        for k in range(self.num_classes):
            mask = tf.equal(labels_r, k)
            g_3k_pos1 = tf.boolean_mask(self.g_3_pos1, mask)
            g_3k_pos2 = tf.boolean_mask(self.g_3_pos2, mask)
            g_3k_neg1 = tf.boolean_mask(self.g_3_neg1, mask)
            g_3k_neg2 = tf.boolean_mask(self.g_3_neg2, mask)
            filt3_k = tf.boolean_mask(filt3, mask)
            W_fc2_k = self.W_fc2 - tf.gather(self.W_fc2, [k], axis=1)

            g_4k_pos = tf.matmul(tf.nn.relu(g_3k_pos1 + self.b_fc1), tf.nn.relu(W_fc2_k)) - tf.matmul(tf.multiply(-g_3k_pos2 + self.b_fc1, filt3_k), tf.nn.relu(-W_fc2_k))
            g_4k_neg = tf.matmul(tf.nn.relu(g_3k_neg1 + self.b_fc1), tf.nn.relu(W_fc2_k)) - tf.matmul(tf.multiply(-g_3k_neg2 + self.b_fc1, filt3_k), tf.nn.relu(-W_fc2_k))
            g_4k_max = tf.maximum(g_4k_pos, g_4k_neg)
            g_4k = tf.nn.max_pool(tf.reshape(g_4k_max, [1, tf.shape(g_4k_max)[0], self.num_classes, 1]),
                    [1, self.num_features, 1, 1], [1, self.num_features, 1, 1], "SAME") + \
                   tf.reshape(self.b_fc2 - self.b_fc2[k], [1, 1, self.num_classes, 1])

            robust_acc += tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(g_4k, tf.constant([0.0])), axis = 2), tf.float32))
            robust_objective += tf.reduce_sum(tf.reduce_logsumexp(g_4k, axis = 2))

        self.acc_bound = robust_acc/tf.cast(tf.shape(labels)[0], tf.float32)
        self.loss = robust_objective/tf.cast(tf.shape(labels)[0], tf.float32)
        return self.loss, self.acc_bound

    def set_mode(self, mode='train'):
        self.mode = mode

    @staticmethod
    def _weight_variable(shape, name):
        initial = tf.keras.initializers.GlorotUniform()
        return tf.Variable(initial_value=initial(shape), name=name, trainable=True)

    @staticmethod
    def _bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                            ksize = [1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    @staticmethod
    def _avg_pool_4x4( x):
        return tf.nn.max_pool(x,
                              ksize = [1,4,4,1],
                              strides=[1,4,4,1],
                              padding='SAME')


