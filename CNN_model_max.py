"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json
import numpy as np

with open('config.json') as config_file:
    config = json.load(config_file)
eps = config['epsilon']

num_classes = 10


class Model(object):
  def __init__(self, num_features):
    self.eps_l1 = np.sqrt(num_features)*eps
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    batch_size = tf.shape(self.x_input)[0]
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.y_input1 = tf.cast(self.y_input, tf.int32)
    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
    self.labels = tf.repeat( self.y_input , repeats = num_features, axis = 0)

    # first convolutional layer
    self.W_conv1 = self._weight_variable([5,5,1,32])
    self.b_conv1 = self._bias_variable([32])

    z1 = self._conv2d(self.x_image, self.W_conv1) + self.b_conv1
    h_conv1 = tf.nn.relu(z1)
    h_pool1 = self._max_pool_2x2(h_conv1)
    self.indices_1 = tf.reshape(tf.vectorized_map(self.pool_argmax, h_conv1), tf.shape(h_pool1))


    new_shape = [batch_size*num_features, tf.shape(z1)[1], tf.shape(z1)[2], tf.shape(z1)[3]]
    offset = self._conv2d(tf.reshape(tf.eye(num_features), [-1, 28, 28, 1]), self.W_conv1)
    self.g_1_pos = tf.reshape(self.eps_l1*offset[None] + z1[:, None], new_shape)
    self.g_1_neg = tf.reshape(-self.eps_l1*offset[None] + z1[:, None], new_shape)
    filt1 = tf.repeat(tf.sign(h_pool1), repeats = num_features*tf.ones(batch_size, tf.int32), axis = 0)
    shape1 = tf.shape(filt1)
    
    # second convolutional layer
    self.W_conv2 = self._weight_variable([5,5,32,64])
    self.b_conv2 = self._bias_variable([64])

    z2 = self._conv2d(h_pool1, self.W_conv2) + self.b_conv2
    h_conv2 = tf.nn.relu(z2)
    h_pool2 = self._max_pool_2x2(h_conv2)
    self.indices_2 = tf.reshape(tf.vectorized_map(self.pool_argmax, h_conv2), tf.shape(h_pool2))

    g_1_pos_max = tf.reshape(tf.vectorized_map(self._gather_max, (self.g_1_pos, tf.repeat(self.indices_1, repeats=num_features, axis=0))), shape1)
    g_1_neg_max = tf.reshape(tf.vectorized_map(self._gather_max, (self.g_1_neg, tf.repeat(self.indices_1, repeats=num_features, axis=0))), shape1)


    self.g_2_pos1 = self._conv2d(self._max_pool_2x2(tf.nn.relu(self.g_1_pos)), tf.nn.relu(self.W_conv2)) - self._conv2d(tf.multiply(g_1_pos_max, filt1), tf.nn.relu(-self.W_conv2))
    self.g_2_pos2 = self._conv2d(self._max_pool_2x2(tf.nn.relu(self.g_1_pos)), tf.nn.relu(-self.W_conv2)) - self._conv2d(tf.multiply(g_1_pos_max, filt1), tf.nn.relu(self.W_conv2))
    self.g_2_neg1 = self._conv2d(self._max_pool_2x2(tf.nn.relu(self.g_1_neg)), tf.nn.relu(self.W_conv2)) - self._conv2d(tf.multiply(g_1_neg_max, filt1), tf.nn.relu(-self.W_conv2))
    self.g_2_neg2 = self._conv2d(self._max_pool_2x2(tf.nn.relu(self.g_1_neg)), tf.nn.relu(-self.W_conv2)) - self._conv2d(tf.multiply(g_1_neg_max, filt1), tf.nn.relu(self.W_conv2))
    filt2 = tf.repeat(tf.sign(h_pool2), repeats = num_features*tf.ones(batch_size, tf.int32), axis = 0)
    shape2 = tf.shape(filt2)


    # first fully connected layer
    self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    self.b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

    g_2_pos1_flat =  tf.reshape(self._max_pool_2x2(tf.nn.relu(self.g_2_pos1 + self.b_conv2)), [-1, 7 * 7 * 64])
    g_2_neg1_flat =  tf.reshape(self._max_pool_2x2(tf.nn.relu(self.g_2_neg1 + self.b_conv2)), [-1, 7 * 7 * 64])

    g_2_pos2_pool = tf.reshape(tf.vectorized_map(self._gather_max, (-self.g_2_pos2+ self.b_conv2, tf.repeat(self.indices_2, repeats=num_features, axis=0))), shape2)
    g_2_neg2_pool = tf.reshape(tf.vectorized_map(self._gather_max, (-self.g_2_neg2+ self.b_conv2, tf.repeat(self.indices_2, repeats=num_features, axis=0))), shape2)
    g_2_pos2_flat = tf.reshape(tf.multiply(g_2_pos2_pool, filt2), [-1, 7 * 7 * 64])
    g_2_neg2_flat = tf.reshape(tf.multiply(g_2_neg2_pool, filt2), [-1, 7 * 7 * 64])


    self.g_3_pos1 = tf.matmul(g_2_pos1_flat, tf.nn.relu(self.W_fc1)) - tf.matmul(g_2_pos2_flat, tf.nn.relu(-self.W_fc1))
    self.g_3_pos2 = tf.matmul(g_2_pos1_flat, tf.nn.relu(-self.W_fc1)) - tf.matmul(g_2_pos2_flat, tf.nn.relu(self.W_fc1))
    self.g_3_neg1 = tf.matmul(g_2_neg1_flat, tf.nn.relu(self.W_fc1)) - tf.matmul(g_2_neg2_flat, tf.nn.relu(-self.W_fc1))
    self.g_3_neg2 = tf.matmul(g_2_neg1_flat, tf.nn.relu(-self.W_fc1)) - tf.matmul(g_2_neg2_flat, tf.nn.relu(self.W_fc1))
    filt3 = tf.repeat(tf.sign(h_fc1), repeats = num_features*tf.ones(batch_size, tf.int32), axis = 0)

    # output layer
    self.W_fc2 = self._weight_variable([1024,10])
    self.b_fc2 = self._bias_variable([10])
    self.pre_softmax = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)

    #Evaluation
    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    robust_objective = 0
    robust_acc = 0

    for k in range(num_classes):
      mask = tf.equal(self.labels, k)
      g_3k_pos1 = tf.boolean_mask(self.g_3_pos1, mask)
      g_3k_pos2 = tf.boolean_mask(self.g_3_pos2, mask)
      g_3k_neg1 = tf.boolean_mask(self.g_3_neg1, mask)
      g_3k_neg2 = tf.boolean_mask(self.g_3_neg2, mask)
      filt3_k = tf.boolean_mask(filt3, mask)
      W_fc2_k = self.W_fc2 - tf.gather(self.W_fc2, [k], axis=1)

      g_4k_pos = tf.matmul(tf.nn.relu(g_3k_pos1 + self.b_fc1), tf.nn.relu(W_fc2_k)) - tf.matmul(tf.multiply(-g_3k_pos2 + self.b_fc1, filt3_k), tf.nn.relu(-W_fc2_k))
      g_4k_neg = tf.matmul(tf.nn.relu(g_3k_neg1 + self.b_fc1), tf.nn.relu(W_fc2_k)) - tf.matmul(tf.multiply(-g_3k_neg2 + self.b_fc1, filt3_k), tf.nn.relu(-W_fc2_k))
      g_4k_max = tf.maximum(g_4k_pos, g_4k_neg)
      g_4k = tf.nn.max_pool(tf.reshape(g_4k_max, [1, tf.shape(g_4k_max)[0], num_classes, 1]), [1, num_features, 1, 1], [1, num_features, 1, 1], "SAME") + tf.reshape(self.b_fc2  - self.b_fc2[k], [1, 1, num_classes, 1])
      

      robust_acc +=  tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(g_4k, tf.constant([0.0])), axis = 2), tf.float32))
      robust_objective +=  tf.reduce_sum( tf.reduce_logsumexp(g_4k, axis = 2))

    self.robust_acc = robust_acc/tf.cast(tf.shape(self.y_input)[0], tf.float32)
    self.robust_l1_xent = robust_objective/tf.cast(tf.shape(self.y_input)[0], tf.float32)


    #Compute linear approximation for robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)

    sum_exps = 0
    sum_exps1 = 0
    for i in range(num_classes):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = self.eps_l1*tf.reduce_max(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      exponent1 = eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
      sum_exps1+=tf.exp(exponent1)
    self.robust_l1_xent_approx = tf.reduce_mean(tf.log(sum_exps))  #l1 robust approximation using our gradient method (no theoretical guarantees)
    self.robust_linf_xent_approx = tf.reduce_mean(tf.log(sum_exps1))#linf robust approximation using our gradient method (no theoretical guarantees)


  @staticmethod
  def _weight_variable(shape):
      initial = tf.glorot_uniform_initializer()
      return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e15)))

  @staticmethod
  def _bias_variable(shape):
      initial = tf.glorot_uniform_initializer()
      return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(2e15)))

  @tf.function 
  def _gather_max(self, args):
    x, indices = args
    return tf.gather(tf.reshape(x, [-1]), indices)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @tf.function
  def pool_argmax(self, args):
      return tf.nn.max_pool_with_argmax(args[None], ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME', include_batch_in_index=True)[1]

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

  @staticmethod
  def _avg_pool_4x4( x):
      return tf.nn.max_pool(x,
                            ksize = [1,4,4,1],
                            strides=[1,4,4,1],
                            padding='SAME')
