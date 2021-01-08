"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import numpy as np

with open('config.json') as config_file:
    config = json.load(config_file)
eps = config['epsilon']

num_classes = 10


class Model(tf.keras.Model):

  def __init__(self):
    self.train_variables = []

    # first convolutional layer
    self.W_conv1 = self._weight_variable([5,5,1,32])
    self.train_variables += self.W_conv1
    self.b_conv1 = self._bias_variable([32])
    self.train_variables += self.b_conv1

    # second convolutional layer
    self.W_conv2 = self._weight_variable([5,5,32,64])
    self.train_variables += self.W_conv2
    self.b_conv2 = self._bias_variable([64])
    self.train_variables += self.b_conv2

    # first fully connected layer
    self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    self.train_variables += self.W_fc1
    self.b_fc1 = self._bias_variable([1024])
    self.train_variables += self.b_fc1

    # output layer
    self.W_fc2 = self._weight_variable([1024,10])
    self.train_variables += self.W_fc2
    self.b_fc2 = self._bias_variable([10])
    self.train_variables += self.b_fc2

  def __call__(self, x_input, y_input):
    with tf.GradientTape() as self.tape:

      self.x_input = x_input
      self.y_input = y_input
      self.y_input1 = tf.cast(self.y_input, tf.int32)
      self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

      h1 = self._conv2d(self.x_image, self.W_conv1) + self.b_conv1
      h_conv1 = tf.nn.relu(h1)
      h_pool1 = self._max_pool_2x2(h_conv1)

      h_conv2 = tf.nn.relu(self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
      h_pool2 = self._max_pool_2x2(h_conv2)

      h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

      self.pre_softmax = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
      y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.y_input, logits=self.pre_softmax)
      self.xent = tf.reduce_mean(y_xent)

  def grad(self):
    return self.xent, self.tape.gradient(self.xent, self.train_variables)

  def evaluate(self):
    #Evaluation
    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Compute linear approximation for robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)

    sum_exps=0
    for i in range(num_classes):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
    self.robust_xent = tf.reduce_mean(tf.log(sum_exps))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.glorot_uniform_initializer()
      return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e15)))

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
