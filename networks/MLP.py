"""
The model is a multiclass perceptron for 10 classes.
"""
import tensorflow as tf


import numpy as np
import json

import robust_net

with open('config.json') as config_file:
    config = json.load(config_file)
eps = config['epsilon']

l1_size = 200
l2_size = 200
num_classes = 10


class robustMLP(robust_net.RobustNet):

  def __init__(self, num_features, initial_learning_rate, training_batch_size):
    super(robustMLP, self).__init__()

    self.train_variables = []

    self.W1 = self._weight_variable([num_features, l1_size])
    self.train_variables += [self.W1]
    self.b1 = self._bias_variable([l1_size])
    self.train_variables += [self.b1]

    self.W2 = self._weight_variable([l1_size, l2_size])
    self.train_variables += [self.W2]
    self.b2 = self._bias_variable([l2_size])
    self.train_variables += [self.b2]

    self.W3 = self._weight_variable([l2_size, num_classes])
    self.train_variables += [self.W3]
    self.b3 = self._bias_variable([num_classes])
    self.train_variables += [self.b3]

    # Setting up the optimizer
    self.learning_rate = \
      tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, training_batch_size * 5, 0.85, staircase=True)

    self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

  def feedforward_pass(self, input):
    # Fully connected layers.
    self.h1 = tf.nn.relu(tf.matmul(input, self.W1) + self.b1)
    self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)
    self.pre_softmax = tf.matmul(self.h2, self.W3) + self.b3
    return self.pre_softmax

  @staticmethod
  def _weight_variable(shape):
    initial = tf.keras.initializers.GlorotUniform()
    return tf.Variable(initial_value=initial(shape), name=str(np.random.randint(1e10)), trainable=True)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
