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

  def __init__(self, num_features, initial_learning_rate, training_batch_size):
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

  def __call__(self, input):
    self.x_input = input
    return self.feedforward_pass(input)

  @tf.function
  def train_step(self, input, label, robust=True):
    self.x_input = input
    self.y_input = label

    with tf.GradientTape() as self.tape:
      with tf.GradientTape(persistent=True) as self.second_tape:

        if robust:
          self.second_tape.watch(self.x_input)

        self.feedforward_pass(self.x_input)

        if robust:
          # Compute linear approximation for robust cross-entropy.
          data_range = tf.range(tf.shape(self.y_input)[0])
          indices = tf.map_fn(lambda n: tf.stack([n, tf.cast(self.y_input[n], tf.int32)]), data_range)

          self.nom_exponent = []
          for i in range(num_classes):
            self.nom_exponent += [self.pre_softmax[:,i] - tf.gather_nd(self.pre_softmax, indices)]

      if robust:
        sum_exps = 0
        for i in range(num_classes):
          grad = self.second_tape.gradient(self.nom_exponent[i], self.x_input)
          exponent = eps * tf.reduce_sum(tf.abs(grad), axis=1) + self.nom_exponent[i]
          sum_exps += tf.math.exp(exponent)

        self.loss = tf.reduce_mean(tf.math.log(sum_exps))

      else:
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.cast(self.y_input, tf.int32), logits=self.pre_softmax)
        self.loss = tf.reduce_mean(y_xent)

    self.optimizer.apply_gradients(zip(self.tape.gradient(self.loss, self.train_variables), self.train_variables))
    print("Graph Created!")

  def evaluate(self, y_input):
    #Evaluation
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.cast(y_input, tf.int32), logits=self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
    initial = tf.keras.initializers.GlorotUniform()
    return tf.Variable(initial_value=initial(shape), name=str(np.random.randint(1e10)), trainable=True)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)