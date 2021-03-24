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
num_classes = 10

class Model(object):
  def __init__(self, num_features):
    self.eps_l1 = np.sqrt(num_features)*eps
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])


    # Fully connected layers.
    self.W1 = self._weight_variable([num_features, l1_size])
    self.b1 = self._bias_variable([l1_size])
    self.z1 = tf.matmul(self.x_input, self.W1) + self.b1
    self.h1 = tf.nn.relu(self.z1)

    self.W2 = self._weight_variable([l1_size, num_classes])
    self.b2 = self._bias_variable([num_classes])

    self.pre_softmax = tf.matmul(self.h1, self.W2) + self.b2

    #Prediction 
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    robust_objective = 0
    robust_acc = 0

    for k in range(num_classes):
      mask = tf.equal(self.y_input, k)
      z_k = tf.boolean_mask(self.z1, mask)
      W2_k = self.W2 - tf.gather(self.W2, [k], axis=1)
      h_1_pos = tf.reshape(self.eps_l1*self.W1[None] + z_k[:, None], [num_features* tf.shape(z_k)[0], l1_size])
      h_1_neg = tf.reshape(-self.eps_l1*self.W1[None] + z_k[:, None], [num_features* tf.shape(z_k)[0], l1_size])
      filt = tf.repeat(tf.nn.relu(tf.sign(z_k)), repeats = num_features*tf.ones(tf.shape(z_k)[0], tf.int32), axis = 0)
      objectives_pos = tf.matmul(tf.nn.relu(h_1_pos), tf.nn.relu(W2_k)) - tf.matmul(tf.multiply(h_1_pos, filt), tf.nn.relu(-W2_k))
      objectives_neg = tf.matmul(tf.nn.relu(h_1_neg), tf.nn.relu(W2_k)) - tf.matmul(tf.multiply(h_1_neg, filt), tf.nn.relu(-W2_k))
      objectives_max = tf.maximum(objectives_pos, objectives_neg)
      objectives = tf.nn.max_pool(tf.reshape(objectives_max, [1, num_features* tf.shape(z_k)[0], num_classes, 1]), [1, num_features, 1, 1], [1, num_features, 1, 1], "SAME")
      logits_diff = objectives + tf.reshape(self.b2  - self.b2[k], [1, 1, num_classes, 1])
      robust_acc +=  tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(logits_diff, tf.constant([0.0])), axis = 2), tf.float32))
      robust_objective +=  tf.reduce_sum( tf.reduce_logsumexp(objectives + tf.reshape(self.b2  - self.b2[k], [1, 1, num_classes, 1]), axis = 2))

    self.robust_acc = robust_acc/tf.cast(tf.shape(self.y_input)[0], tf.float32)

    self.robust_l1_xent = robust_objective/tf.cast(tf.shape(self.y_input)[0], tf.float32)

    #self.robust_l1_xent  can also be computed with these  two lines:
    #robust_outputs = tf.transpose(tf.reshape(tf.vectorized_map(self._loss_fn, (self.x_input, self.y_input, self.z1)), (-1, 10)))
    #self.robust_l1_xent = tf.reduce_mean( tf.reduce_logsumexp(robust_outputs, axis = 0)) 

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
      exponent1 = self.eps_l1*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
      sum_exps1+=tf.exp(exponent1)
    self.robust_l1_xent_approx = tf.reduce_mean(tf.log(sum_exps))  #l1 robust approximation using our gradient method (no theoretical guarantees)
    self.robust_linf_xent_approx = tf.reduce_mean(tf.log(sum_exps1))#linf robust approximation using our gradient method (no theoretical guarantees)


    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  @tf.function 
  def _loss_fn(self, args):
      x, y, z= args
      z = tf.reshape(z, (1, -1))
      filt = tf.nn.relu(tf.sign(z))
      W12 = self.W2 - tf.gather(self.W2, [y], axis=1)
      g1 = tf.matmul(tf.nn.relu(self.eps_l1*self.W1+ z), tf.nn.relu(W12)) - tf.matmul(tf.multiply(self.eps_l1*self.W1 + z, filt), tf.nn.relu(-W12))
      g2 = tf.matmul(tf.nn.relu(-self.eps_l1*self.W1+ z), tf.nn.relu(W12)) - tf.matmul(tf.multiply(-self.eps_l1*self.W1 + z, filt), tf.nn.relu(-W12))
      return tf.reduce_max( tf.maximum(g1 , g2), axis = 0) + self.b2 - self.b2[y]

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.03)
      return tf.Variable(initial)
      #initial = tf.glorot_uniform_initializer()
      #return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e10)))

  @staticmethod
  def _bias_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)