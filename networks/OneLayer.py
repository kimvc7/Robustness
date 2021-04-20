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
    initial_learning_rate = float(config['initial_learning_rate'])
    training_batch_size = config['training_batch_size']
    num_classes = config['num_classes']
    batch_decrease_learning_rate = float(config['batch_decrease_learning_rate'])

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

  def certificate_loss(self, epsilon, labels):

    labels_r = tf.repeat(labels, repeats = self.num_features, axis = 0)

    robust_objective = 0
    robust_acc = 0

    for k in range(self.num_classes):
      mask = tf.equal(labels_r, k)
      z_k = tf.boolean_mask(self.z1, mask)
      W2_k = self.W2 - tf.gather(self.W2, [k], axis=1)
      h_1_pos = tf.reshape(epsilon*self.W1[None] + z_k[:, None], [self.num_features * tf.shape(z_k)[0], self.l1_size])
      h_1_neg = tf.reshape(-epsilon*self.W1[None] + z_k[:, None], [self.num_features * tf.shape(z_k)[0], self.l1_size])
      filt = tf.repeat(tf.nn.relu(tf.sign(z_k)), repeats = self.num_features * tf.ones(tf.shape(z_k)[0], tf.int32), axis = 0)
      objectives_pos = tf.matmul(tf.nn.relu(h_1_pos), tf.nn.relu(W2_k)) - tf.matmul(tf.multiply(h_1_pos, filt), tf.nn.relu(-W2_k))
      objectives_neg = tf.matmul(tf.nn.relu(h_1_neg), tf.nn.relu(W2_k)) - tf.matmul(tf.multiply(h_1_neg, filt), tf.nn.relu(-W2_k))
      objectives_max = tf.maximum(objectives_pos, objectives_neg)
      objectives = tf.nn.max_pool(tf.reshape(objectives_max, [1, self.num_features * tf.shape(z_k)[0], self.num_classes, 1]), [1, self.num_features, 1, 1], [1, self.num_features, 1, 1], "SAME")
      logits_diff = objectives + tf.reshape(self.b2 - self.b2[k], [1, 1, self.num_classes, 1])
      robust_acc += tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(logits_diff, tf.constant([0.0])), axis = 2), tf.float32))
      robust_objective += tf.reduce_sum(tf.reduce_logsumexp(objectives + tf.reshape(self.b2 - self.b2[k], [1, 1, self.num_classes, 1]), axis = 2))

    self.acc_bound = robust_acc/tf.cast(tf.shape(labels)[0], tf.float32)
    self.loss = robust_objective/tf.cast(tf.shape(labels)[0], tf.float32)
    return self.loss, self.acc_bound

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
