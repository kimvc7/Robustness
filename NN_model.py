"""
The model is a multiclass perceptron for 10 classes.
"""
import tensorflow as tf


import numpy as np
import json


with open('config.json') as config_file:
    config = json.load(config_file)
eps = config['epsilon']

l1_size = 200
l2_size = 200
num_classes = 10

class Model(tf.keras.Model):

  def __init__(self, num_features):
    super(Model, self).__init__()

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

  def __call__(self, input):

    self.x_input = input

    # Fully connected layers.
    self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1) + self.b1)
    self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)
    self.pre_softmax = tf.matmul(self.h2, self.W3) + self.b3
    return self.pre_softmax


  def feedfowrard_robust(self, input, label=None, robust=True):

    self.x_input = input
    if label is not None:
      self.y_input = label

    with tf.GradientTape() as self.tape:
      with tf.GradientTape(persistent=True) as self.second_tape:

        self.second_tape.watch(self.x_input)

        # Fully connected layers.
        self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1) + self.b1)
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)
        self.pre_softmax = tf.matmul(self.h2, self.W3) + self.b3

        if robust:
          # Compute linear approximation for robust cross-entropy.
          data_range = tf.range(tf.shape(self.y_input)[0])
          indices = tf.map_fn(lambda n: tf.stack([n, tf.cast(self.y_input[n], tf.int32)]), data_range)

          self.nom_exponent = []
          sum_exps = 0
          for i in range(num_classes):
            self.nom_exponent += [self.pre_softmax[:,i] - tf.gather_nd(self.pre_softmax, indices)]

      if robust:
        for i in range(num_classes):
          grad = self.second_tape.gradient(self.nom_exponent[i], self.x_input)
          exponent = eps * tf.reduce_sum(tf.abs(grad), axis=1) + self.nom_exponent[i]
          sum_exps += tf.exp(exponent)

        self.loss = tf.reduce_mean(tf.math.log(sum_exps))

      else:
        if robust == False:
          y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.y_input, tf.int32), logits=self.pre_softmax)
          self.loss = tf.reduce_mean(y_xent)
    return self.pre_softmax

  def evaluate(self, y_input):
    #Evaluation
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.cast(y_input, tf.int32), logits=self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def grad(self):
    return zip(self.tape.gradient(self.loss, self.train_variables), self.train_variables)

  @staticmethod
  def _weight_variable(shape):
    initial = tf.keras.initializers.GlorotUniform()
    return tf.Variable(initial_value=initial(shape), name=str(np.random.randint(1e10)), trainable=True)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)