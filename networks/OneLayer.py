"""
The model is a multiclass perceptron for 10 classes.
"""
import tensorflow as tf
import numpy as np
import robustify_network


class robustOneLayer(robustify_network.RobustifyNetwork):

  def __init__(self, config, num_features):
    super().__init__(config['num_classes'], config['epsilon'])

    l1_size = config['l1_size']
    initial_learning_rate = config['initial_learning_rate']
    training_batch_size = config['training_batch_size']
    num_classes = config['num_classes']
    batch_decrease_learning_rate = config['batch_decrease_learning_rate']

    self.mode = 'train'
    self.num_features = num_features
    self.l1_size = l1_size

    self.train_variables = []

    self.W1 = self._weight_variable([num_features, l1_size])
    self.train_variables += [self.W1]
    self.b1 = self._bias_variable([l1_size])
    self.train_variables += [self.b1]

    self.W2 = self._weight_variable([l1_size, num_classes])
    self.train_variables += [self.W2]
    self.b2 = self._bias_variable([num_classes])
    self.train_variables += [self.b2]

    # Setting up the optimizer
    self.learning_rate = \
      tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                     training_batch_size * batch_decrease_learning_rate, 0.85, staircase=True)

    self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

  def feedforward_pass(self, input):
    # Fully connected layers.
    self.z1 = tf.matmul(input, self.W1) + self.b1
    self.h1 = tf.nn.relu(self.z1)
    self.pre_softmax = tf.matmul(self.h1, self.W2) + self.b2
    return self.pre_softmax

  def set_mode(self, mode='train'):
      self.mode = mode

  @staticmethod
  def _weight_variable(shape):
    initial = tf.keras.initializers.GlorotUniform()
    return tf.Variable(initial_value=initial(shape), name=str(np.random.randint(1e10)), trainable=True)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
