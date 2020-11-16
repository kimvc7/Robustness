"""
The model is a multiclass perceptron for 10 classes.
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import numpy as np
import json


with open('config.json') as config_file:
    config = json.load(config_file)
eps = config['epsilon']

l1_size = 200
l2_size = 200
num_classes = 10

class Model(object):
  def __init__(self, num_features):
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])


    # Fully connected layers.
    self.W1 = self._weight_variable([num_features, l1_size])
    self.b1 = self._bias_variable([l1_size])
    self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1) + self.b1)

    self.W2 = self._weight_variable([l1_size, l2_size])
    self.b2 = self._bias_variable([l2_size])
    self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)

    self.W3 = self._weight_variable([l2_size, num_classes])
    self.b3 = self._bias_variable([num_classes])
    self.pre_softmax = tf.matmul(self.h2, self.W3) + self.b3

    #Prediction 
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    #Compute linear approximation for robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)

    sum_exps = 0
    for i in range(num_classes):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      exponent1 = eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
    self.robust_xent = tf.reduce_mean(tf.log(sum_exps))


    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  @staticmethod
  def _weight_variable(shape):
      initial = tf.glorot_uniform_initializer()
      return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e10)))

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)