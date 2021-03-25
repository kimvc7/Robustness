"""
The abstract robust model
"""
import tensorflow as tf
import numpy as np


class RobustifyNetwork(tf.keras.Model):

  def __init__(self, num_classes, epsilon):
    super(RobustifyNetwork, self).__init__()

    self.num_classes = num_classes
    self.train_variables = []
    pass

  def feedforward_pass(self, input):
    pass

  def __call__(self, input):
    self.x_input = input
    return self.feedforward_pass(input)

  @tf.function
  def train_step(self, input, label, epsilon, robust=True, clipping=False, l1=False, certificate=False):
    self._full_call(input, label, epsilon, robust=robust, clipping=clipping,
                    evaluate=False, l1=l1, certificate=certificate)

  @tf.function
  def evaluate(self, input, label, epsilon, step=-1, summary=None, clipping=False):
    self._full_call(input, label, epsilon, step=step, robust=True, clipping=clipping,
                    evaluate=True, summary=summary, l1=False, certificate=False)

  def _full_call(self, input, label, epsilon, robust=True, clipping=False, evaluate=False,
                 summary=None, step=-1, l1=False, certificate=False):

    self.x_input = input
    self.y_input = label

    if clipping:
      self.M = tf.math.minimum(1 - self.x_input, epsilon)
      self.m = tf.math.maximum(-self.x_input, -epsilon)

    if not certificate:
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
            for i in range(self.num_classes):
              self.nom_exponent += [self.pre_softmax[:,i] - tf.gather_nd(self.pre_softmax, indices)]

        if robust:
          sum_exps = 0
          for i in range(self.num_classes):
            grad = self.second_tape.gradient(self.nom_exponent[i], self.x_input)
            if clipping:
              positive_terms = tf.math.multiply(self.M, tf.nn.relu(grad[0]))
              negative_terms = tf.math.multiply(self.m, tf.nn.relu(-grad[0]))
              exponent = tf.reduce_sum(positive_terms - negative_terms, axis=1) + self.nom_exponent[i]
            elif l1:
              exponent = epsilon * tf.reduce_max(tf.abs(grad), axis=1) + self.nom_exponent[i]
            else:
              exponent = epsilon * tf.reduce_sum(tf.abs(grad), axis=1) + self.nom_exponent[i]
            sum_exps += tf.math.exp(exponent)

          self.loss = tf.reduce_mean(tf.math.log(sum_exps))

        else:
          y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.y_input, tf.int32), logits=self.pre_softmax)
          self.loss = tf.reduce_mean(y_xent)

    else: #certificate objective:
      with tf.GradientTape() as self.tape:

          self.feedforward_pass(self.x_input)

          robust_objective = 0
          robust_acc = 0

          for k in range(self.num_classes):
            mask = tf.equal(self.y_input, k)
            z_k = tf.boolean_mask(self.z1, mask)
            W2_k = self.W2 - tf.gather(self.W2, [k], axis=1)
            h_1_pos = tf.reshape(epsilon*self.W1[None] + z_k[:, None], [self.num_features* tf.shape(z_k)[0], self.l1_size])
            h_1_neg = tf.reshape(-epsilon*self.W1[None] + z_k[:, None], [self.num_features* tf.shape(z_k)[0], self.l1_size])
            filt = tf.repeat(tf.nn.relu(tf.sign(z_k)), repeats = self.num_features*tf.ones(tf.shape(z_k)[0], tf.int32), axis = 0)
            objectives_pos = tf.matmul(tf.nn.relu(h_1_pos), tf.nn.relu(W2_k)) - tf.matmul(tf.multiply(h_1_pos, filt), tf.nn.relu(-W2_k))
            objectives_neg = tf.matmul(tf.nn.relu(h_1_neg), tf.nn.relu(W2_k)) - tf.matmul(tf.multiply(h_1_neg, filt), tf.nn.relu(-W2_k))
            objectives_max = tf.maximum(objectives_pos, objectives_neg)
            objectives = tf.nn.max_pool(tf.reshape(objectives_max, [1, self.num_features* tf.shape(z_k)[0], self.num_classes, 1]), [1, self.num_features, 1, 1], [1, self.num_features, 1, 1], "SAME")
            logits_diff = objectives + tf.reshape(self.b2  - self.b2[k], [1, 1, self.num_classes, 1])
            robust_acc += tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(logits_diff, tf.constant([0.0])), axis = 2), tf.float32))
            robust_objective += tf.reduce_sum( tf.reduce_logsumexp(objectives + tf.reshape(self.b2  - self.b2[k], [1, 1, self.num_classes, 1]), axis = 2))

          self.loss = robust_objective/tf.cast(tf.shape(self.y_input)[0], tf.float32)


    if not evaluate:
      self.optimizer.apply_gradients(zip(self.tape.gradient(self.loss, self.train_variables), self.train_variables))
      #print("\n Graph Created! \n")

    else:
      # Evaluation
      y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(label, tf.int32), logits=self.pre_softmax)
      self.xent = tf.reduce_mean(y_xent)

      self.y_pred = tf.argmax(self.pre_softmax, 1)
      correct_prediction = tf.equal(self.y_pred, label)
      self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      if summary:
        with summary.as_default():
          tf.summary.scalar('Cross Entropy', self.xent, step)
          tf.summary.scalar('Accuracy', self.accuracy, step)
          tf.summary.scalar('Robust Loss', self.loss, step)
          tf.summary.scalar('Learning Rate', self.optimizer.learning_rate(step), step)
          tf.summary.scalar('Epsilon', epsilon, step)

      #print("\n Evaluate Graph Created! \n")

  def load_all(self, path, load_optimizer=True):

    if load_optimizer:
      opt_weights = np.load(path + '_optimizer.npy', allow_pickle=True)

      grad_vars = self.trainable_weights
      zero_grads = [tf.zeros_like(w) for w in grad_vars]
      self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
      self.optimizer.set_weights(opt_weights)

    self.load_weights(path)

  def save_all(self, path):
    self.save_weights(path)
    np.save(path + '_optimizer.npy', self.optimizer.get_weights())