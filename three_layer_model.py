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

l1_size = 100
l2_size = 100
l3_size = 100
num_classes = 10

class Model(object):
  def __init__(self, num_features):
    self.eps_l1 = np.sqrt(num_features)*eps
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    batch_size = tf.shape(self.x_input)[0]
    self.labels = tf.repeat( self.y_input , repeats = num_features, axis = 0)


    self.W1 = self._weight_variable([num_features, l1_size])
    self.b1 = self._bias_variable([l1_size])
    self.z1 = tf.matmul(self.x_input, self.W1) + self.b1
    self.h1 = tf.nn.relu(self.z1)

    self.g_1_pos = tf.reshape(self.eps_l1*self.W1[None] + self.z1[:, None], [num_features* batch_size , l1_size])
    self.g_1_neg = tf.reshape(-self.eps_l1*self.W1[None] + self.z1[:, None], [num_features* batch_size , l1_size])
    filt = tf.repeat(tf.sign(self.h1), repeats = num_features*tf.ones(batch_size, tf.int32), axis = 0)

    self.W2 = self._weight_variable([l1_size, l2_size])
    self.b2 = self._bias_variable([l2_size])
    self.z2 = tf.matmul(self.h1, self.W2) + self.b2
    self.h2 = tf.nn.relu(self.z2)

    self.g_2_pos1 = tf.matmul(tf.nn.relu(self.g_1_pos), tf.nn.relu(self.W2)) - tf.matmul(tf.multiply(self.g_1_pos, filt), tf.nn.relu(-self.W2))
    self.g_2_pos2 = tf.matmul(tf.nn.relu(self.g_1_pos), tf.nn.relu(-self.W2)) - tf.matmul(tf.multiply(self.g_1_pos, filt), tf.nn.relu(self.W2))
    self.g_2_neg1 = tf.matmul(tf.nn.relu(self.g_1_neg), tf.nn.relu(self.W2)) - tf.matmul(tf.multiply(self.g_1_neg, filt), tf.nn.relu(-self.W2))
    self.g_2_neg2 = tf.matmul(tf.nn.relu(self.g_1_neg), tf.nn.relu(-self.W2)) - tf.matmul(tf.multiply(self.g_1_neg, filt), tf.nn.relu(self.W2))
    filt2 = tf.repeat(tf.sign(self.h2), repeats = num_features*tf.ones(batch_size, tf.int32), axis = 0)
    

    self.W3 = self._weight_variable([l2_size, l3_size])
    self.b3 = self._bias_variable([l3_size])
    self.z3 = tf.matmul(self.h2, self.W3) + self.b3
    self.h3 = tf.nn.relu(self.z3)

    self.g_3_pos1 = tf.matmul(tf.nn.relu(self.g_2_pos1 + self.b2), tf.nn.relu(self.W3)) - tf.matmul(tf.multiply(-self.g_2_pos2 + self.b2, filt2), tf.nn.relu(-self.W3))
    self.g_3_pos2 = tf.matmul(tf.nn.relu(self.g_2_pos1 + self.b2), tf.nn.relu(-self.W3)) - tf.matmul(tf.multiply(-self.g_2_pos2 + self.b2, filt2), tf.nn.relu(self.W3))
    self.g_3_neg1 = tf.matmul(tf.nn.relu(self.g_2_neg1 + self.b2), tf.nn.relu(self.W3)) - tf.matmul(tf.multiply(-self.g_2_neg2 + self.b2, filt2), tf.nn.relu(-self.W3))
    self.g_3_neg2 = tf.matmul(tf.nn.relu(self.g_2_neg1 + self.b2), tf.nn.relu(-self.W3)) - tf.matmul(tf.multiply(-self.g_2_neg2 + self.b2, filt2), tf.nn.relu(self.W3))
    filt3 = tf.repeat(tf.sign(self.h3), repeats = num_features*tf.ones(batch_size, tf.int32), axis = 0)

    self.W4 = self._weight_variable([l3_size, num_classes])
    self.b4 = self._bias_variable([num_classes])
    self.pre_softmax = tf.matmul(self.h3, self.W4) + self.b4

    #Prediction 
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    robust_objective = 0
    robust_acc = 0
    self.mask = tf.equal(self.labels, 0)

    for k in range(num_classes):
      mask = tf.equal(self.labels, k)
      g_3k_pos1 = tf.boolean_mask(self.g_3_pos1, mask)
      g_3k_pos2 = tf.boolean_mask(self.g_3_pos2, mask)
      g_3k_neg1 = tf.boolean_mask(self.g_3_neg1, mask)
      g_3k_neg2 = tf.boolean_mask(self.g_3_neg2, mask)
      filt3_k = tf.boolean_mask(filt3, mask)
      W4_k = self.W4 - tf.gather(self.W4, [k], axis=1)

      g_4k_pos = tf.matmul(tf.nn.relu(g_3k_pos1 + self.b3), tf.nn.relu(W4_k)) - tf.matmul(tf.multiply(-g_3k_pos2 + self.b3, filt3_k), tf.nn.relu(-W4_k))
      g_4k_neg = tf.matmul(tf.nn.relu(g_3k_neg1 + self.b3), tf.nn.relu(W4_k)) - tf.matmul(tf.multiply(-g_3k_neg2 + self.b3, filt3_k), tf.nn.relu(-W4_k))
      g_4k_max = tf.maximum(g_4k_pos, g_4k_neg)
      g_4k = tf.nn.max_pool(tf.reshape(g_4k_max, [1, tf.shape(g_4k_max)[0], num_classes, 1]), [1, num_features, 1, 1], [1, num_features, 1, 1], "SAME") + tf.reshape(self.b4  - self.b4[k], [1, 1, num_classes, 1])
      

      robust_acc +=  tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(g_4k, tf.constant([0.0])), axis = 2), tf.float32))
      robust_objective +=  tf.reduce_sum( tf.reduce_logsumexp(g_4k, axis = 2))

    self.robust_acc = robust_acc/tf.cast(tf.shape(self.y_input)[0], tf.float32)
    self.robust_l1_xent = robust_objective/tf.cast(tf.shape(self.y_input)[0], tf.float32)

    #self.robust_l1_xent  can also be computed with these  two lines:
    #robust_outputs = tf.transpose(tf.reshape(tf.vectorized_map(self._loss_fn, (self.x_input, self.y_input, self.z1, self.h1, self.h2, self.h3)), (-1, 10)))
    #self.robust_l1_xent = tf.reduce_mean( tf.reduce_logsumexp(robust_outputs, axis = 0)) 

    #Compute linear approximation for robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)

    sum_exps_l1 = 0
    sum_exps_linf = 0
    sum_exps_l2 = 0

    for i in range(num_classes):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)

      exponent_l1 = self.eps_l1*tf.reduce_max(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      exponent_linf = eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      exponent_l2  = eps*tf.sqrt(tf.reduce_sum(tf.square(grad[0]), axis=1)) + self.nom_exponent[i]

      sum_exps_l1 +=tf.exp(exponent_l1)
      sum_exps_linf+=tf.exp(exponent_linf)
      sum_exps_l2+=tf.exp(exponent_l2)

    self.robust_l1_xent_approx = tf.reduce_mean(tf.log(sum_exps_l1))  #l1 robust approximation using our gradient method (no theoretical guarantees)
    self.robust_linf_xent_approx = tf.reduce_mean(tf.log(sum_exps_linf))#linf robust approximation using our gradient method (no theoretical guarantees)
    self.robust_l2_xent_approx = tf.reduce_mean(tf.log(sum_exps_l2)) #l2 robust approximation using our gradient method (no theoretical guarantees)

    gradient = tf.gradients(self.xent, self.x_input)[0]
    self.robust_l1_xent_baseline = self.xent + eps* tf.reduce_max(tf.abs(gradient[0])) 
    self.robust_linf_xent_baseline = self.xent + eps* tf.reduce_sum(tf.abs(gradient[0])) 
    self.robust_l2_xent_baseline = self.xent + eps* tf.sqrt(tf.reduce_sum(tf.square(gradient[0])))

    self.robust_l1_xent_baseline_squared = self.xent + eps* tf.square(tf.reduce_max(tf.abs(gradient[0])))
    self.robust_linf_xent_baseline_squared = self.xent + eps* tf.square(tf.reduce_sum(tf.abs(gradient[0])))
    self.robust_l2_xent_baseline_squared = self.xent + eps* tf.reduce_sum(tf.square(gradient[0]))


    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  @tf.function 
  def _loss_fn(self, args):
      x, y, z1, h1, h2, h3, = args
      z1 = tf.reshape(z1, (1, -1))
      h1 = tf.reshape(h1, (1, -1))
      h2 = tf.reshape(h2, (1, -1))
      h3 = tf.reshape(h3, (1, -1))
      filt = tf.sign(h1)
      filt2 = tf.sign(h2)
      filt3 = tf.sign(h3)
      W14 = self.W4 - tf.gather(self.W4, [y], axis=1)
      g2pos1 = tf.matmul(tf.nn.relu(self.eps_l1*self.W1+ z1), tf.nn.relu(self.W2)) - tf.matmul(tf.multiply(self.eps_l1*self.W1 + z1, filt), tf.nn.relu(-self.W2))
      g2pos2 = tf.matmul(tf.nn.relu(self.eps_l1*self.W1+ z1), tf.nn.relu(-self.W2)) - tf.matmul(tf.multiply(self.eps_l1*self.W1 + z1, filt), tf.nn.relu(self.W2))
      g2neg1 = tf.matmul(tf.nn.relu(-self.eps_l1*self.W1+ z1), tf.nn.relu(self.W2)) - tf.matmul(tf.multiply(-self.eps_l1*self.W1 + z1, filt), tf.nn.relu(-self.W2))
      g2neg2 = tf.matmul(tf.nn.relu(-self.eps_l1*self.W1+ z1), tf.nn.relu(-self.W2)) - tf.matmul(tf.multiply(-self.eps_l1*self.W1 + z1, filt), tf.nn.relu(self.W2))
      g3pos1 = tf.matmul(tf.nn.relu(g2pos1 + self.b2), tf.nn.relu(self.W3)) - tf.matmul(tf.multiply(-g2pos2 + self.b2, filt2), tf.nn.relu(-self.W3))
      g3pos2 = tf.matmul(tf.nn.relu(g2pos1 + self.b2), tf.nn.relu(-self.W3)) - tf.matmul(tf.multiply(-g2pos2 + self.b2, filt2), tf.nn.relu(self.W3))
      g3neg1 = tf.matmul(tf.nn.relu(g2neg1 + self.b2), tf.nn.relu(self.W3)) - tf.matmul(tf.multiply(-g2neg2 + self.b2, filt2), tf.nn.relu(-self.W3))
      g3neg2 = tf.matmul(tf.nn.relu(g2neg1 + self.b2), tf.nn.relu(-self.W3)) - tf.matmul(tf.multiply(-g2neg2 + self.b2, filt2), tf.nn.relu(self.W3))
      g_4k_pos = tf.matmul(tf.nn.relu(g3pos1 + self.b3), tf.nn.relu(W14)) - tf.matmul(tf.multiply(-g3pos2 + self.b3, filt3), tf.nn.relu(-W14))
      g_4k_neg = tf.matmul(tf.nn.relu(g3neg1 + self.b3), tf.nn.relu(W14)) - tf.matmul(tf.multiply(-g3neg2 + self.b3, filt3), tf.nn.relu(-W14))
      return tf.reduce_max( tf.maximum(g_4k_pos , g_4k_neg), axis = 0) + self.b4 - self.b4[y]


  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.05)
      return tf.Variable(initial)
      #initial = tf.glorot_uniform_initializer()
      #return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e10)))

  @staticmethod
  def _bias_variable(shape):
      initial = tf.truncated_normal(shape, stddev=1)
      return tf.Variable(initial)