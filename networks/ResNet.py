"""
The model is the tensroflow v2 version of ResNet in Cifar10 Robustness challenge
https://github.com/MadryLab/cifar10_challenge
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import robustify_network


class robustResNet(robustify_network.RobustifyNetwork):

    def __init__(self, config):
        super().__init__(config['num_classes'], config['epsilon'])

        initial_learning_rate = config['initial_learning_rate']
        training_batch_size = config['training_batch_size']
        batch_decrease_learning_rate = config['batch_decrease_learning_rate']

        self.activate_before_residual = [True, False, False]
        self.filters = [16, 160, 320, 640]
        self.strides = [1, 2, 2]
        self.mode = 'train'

        self.train_variables = []
        self.vars = {}

        #First convolution:
        self.vars['init_conv'] = self._weight_variable([3, 3, 3, 16], 'conv1')
        self.train_variables += [self.vars['init_conv']]

        #Blocks:
        for block in range(3):
            self.vars['unit_' + str(block+1) + '_0'] = \
                self._residual_variables(self.filters[block], self.filters[block+1],
                                         self.activate_before_residual[block],
                                         'unit_' + str(block+1) + '_0')
            for i in range(1, 5):
                self.vars['unit_' + str(block+1) + '_' + str(i)] = \
                    self._residual_variables(self.filters[block+1], self.filters[block+1], False,
                                             'unit_' + str(block+1) + '_' + str(i))

        #Read-out:
        self.vars['final_bn'] = self._batch_norm_variables('final_bn', [self.filters[block+1]])
        self.train_variables += [self.vars['final_bn']['gamma']]
        self.train_variables += [self.vars['final_bn']['beta']]

        self.vars['logit_W'] = self._dense_variable([self.filters[3], 10], 'logit_W')
        self.vars['logit_bias'] = self._bias_variable([10], 'logit_bias')
        self.train_variables += [self.vars['logit_W']]
        self.train_variables += [self.vars['logit_bias']]

        # Setting up the optimizer
        self.learning_rate = \
          tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                         training_batch_size * batch_decrease_learning_rate, 0.85, staircase=True)

        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, momentum=0.9, nesterov=False,)

    def feedforward_pass(self, input):
        self.x_input_image = tf.cast(tf.reshape(input, [-1, 32, 32, 3]), dtype=tf.float32)

        #First convolution:
        x = self._conv2d(self.x_input_image, self.vars['init_conv'],  self._stride_arr(1))

        #Blocks:
        for block in range(3):
            x = self._residual(x, self.filters[block], self.filters[block+1],
                               self.vars['unit_' + str(block+1) + '_0'],
                               self._stride_arr(self.strides[block]), self.mode=='train',
                               self.activate_before_residual[block])
            for i in range(1, 5):
                x = self._residual(x, self.filters[block+1], self.filters[block+1],
                                   self.vars['unit_' + str(block+1) + '_' + str(i)],
                                   self._stride_arr(1), self.mode=='train', False)

        #Read-out:
        x = self._batch_norm(x, self.vars['final_bn'], 0.9, self.mode=='train')
        x = self._relu(x, 0.1)
        x = self._global_avg_pool(x)
        self.pre_softmax = self._fully_connected(x, self.vars['logit_W'], self.vars['logit_bias'])

        return self.pre_softmax

    def set_mode(self, mode='train'):
        self.mode = mode

    def _residual_variables(self, in_filter, out_filter, activate_before_residual, name):
        res_vars = {}
        if activate_before_residual:
            res_vars['init_bn'] = self._batch_norm_variables(name + 'init_bn', [in_filter])
        else:
            res_vars['init_bn'] = self._batch_norm_variables(name + 'init_bn', [in_filter])
        self.train_variables += [res_vars['init_bn']['gamma']]
        self.train_variables += [res_vars['init_bn']['beta']]

        res_vars['conv1'] = self._weight_variable([3, 3, in_filter, out_filter], name + 'conv1')
        self.train_variables += [res_vars['conv1']]

        res_vars['bn2'] = self._batch_norm_variables(name + 'bn2', [out_filter])
        self.train_variables += [res_vars['bn2']['gamma']]
        self.train_variables += [res_vars['bn2']['beta']]
        res_vars['conv2'] = self._weight_variable([3, 3, out_filter, out_filter], name + 'conv2')
        self.train_variables += [res_vars['conv2']]

        return res_vars

    def _residual(self, x, in_filter, out_filter, res_vars, stride, mode, activate_before_residual=False):
        if activate_before_residual:
            x = self._batch_norm(x, res_vars['init_bn'], 0.9, mode)
            x = self._relu(x, 0.1)
            orig_x = x
        else:
            orig_x = x
            x = self._batch_norm(x, res_vars['init_bn'], 0.9, mode)
            x = self._relu(x, 0.1)

        x = self._conv2d(x, res_vars['conv1'], stride)

        x = self._batch_norm(x, res_vars['bn2'], 0.9, mode)
        x = self._relu(x, 0.1)
        x = self._conv2d(x, res_vars['conv2'], [1, 1, 1, 1])

        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        x += orig_x
        return x

    @staticmethod
    def _batch_norm_variables(name, shape):
        """Batch normalization create variables."""
        #with tf.name_scope(name):
        bn_vars = {}
        bn_vars['gamma'] = tf.Variable(initial_value=tf.constant(1.0, shape=shape), name='scale_' + name, trainable=True)
        bn_vars['beta'] = tf.Variable(initial_value=tf.constant(0.0, shape=shape), name='beta_' + name, trainable=True)
        bn_vars['moving_mean'] = tf.Variable(initial_value=tf.constant(0.0, shape=shape), name='mean_' + name, trainable=False)
        bn_vars['moving_var'] = tf.Variable(initial_value=tf.constant(0.0, shape=shape), name='var_' + name, trainable=False)
        return bn_vars

    @staticmethod
    def _batch_norm(x, bn_vars, momentum, train):
        if train:
            batch_mean, batch_var = tf.nn.moments(x, [0])
            x = tf.nn.batch_normalization(
                x, batch_mean, batch_var, bn_vars['beta'], bn_vars['gamma'], 0.001)
            bn_vars['moving_mean'] = bn_vars['moving_mean'] * momentum + batch_mean * (1 - momentum)
            bn_vars['moving_var'] = bn_vars['moving_var'] * momentum + batch_var * (1 - momentum)
        else:
            x = tf.nn.batch_normalization(
                x, bn_vars['moving_mean'], bn_vars['moving_var'], bn_vars['beta'], bn_vars['gamma'], 0.001)
        return x

    @staticmethod
    def _weight_variable(shape, name):
        num_weights = shape[0]*shape[1]*shape[3]
        initial = tf.random_normal_initializer(stddev=np.sqrt(2.0/num_weights))
        return tf.Variable(initial_value=initial(shape), name=name, trainable=True)

    @staticmethod
    def _dense_variable(shape, name):
        initial = tf.compat.v1.uniform_unit_scaling_initializer(factor=1.0)
        return tf.Variable(initial_value=initial(shape), name=name, trainable=True)

    @staticmethod
    def _bias_variable(shape, name):
        initial = tf.constant(0.0, shape=shape, name=name)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W, strides):
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

    @staticmethod
    def _relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def _fully_connected(x, W, b):
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        return tf.matmul(x, W) + b

    @staticmethod
    def _global_avg_pool(x):
        return tf.reduce_mean(x, [1, 2])

    @staticmethod
    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def load_Madry(self, model_dir):
        from tensorflow.python.training import py_checkpoint_reader
        reader = py_checkpoint_reader.NewCheckpointReader(
            tf.train.latest_checkpoint(model_dir + '/checkpoints/'))

        self.W_conv1.assign(reader.get_tensor("Variable"))
        self.b_conv1.assign(reader.get_tensor("Variable_1"))
        self.W_conv2.assign(reader.get_tensor("Variable_2"))
        self.b_conv2.assign(reader.get_tensor("Variable_3"))
        self.W_fc1.assign(reader.get_tensor("Variable_4"))
        self.b_fc1.assign(reader.get_tensor("Variable_5"))
        self.W_fc2.assign(reader.get_tensor("Variable_6"))
        self.b_fc2.assign(reader.get_tensor("Variable_7"))

