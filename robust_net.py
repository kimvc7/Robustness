"""
The abstract robust model
"""
import tensorflow as tf
import json


with open('config.json') as config_file:
    config = json.load(config_file)
eps = config['epsilon']
num_classes = 10


class RobustNet(tf.keras.Model):

  def __init__(self):
    super(RobustNet, self).__init__()

    self.train_variables = []
    pass

  def feedforward_pass(self, input):
    pass

  def __call__(self, input):
    self.x_input = input
    return self.feedforward_pass(input)

  @tf.function
  def train_step(self, input, label, robust=True):
    self._full_call(input, label, robust=robust, evaluate=False, summary=None, step=0)
    print("Graph Created!")

  @tf.function
  def evaluate(self, input, label, summary=None, step=-1):
    self._full_call(input, label, robust=True, evaluate=True, summary=summary, step=step)

  def _full_call(self, input, label, robust=True, evaluate=False, summary=None, step=0):
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

    if not evaluate:
      self.optimizer.apply_gradients(zip(self.tape.gradient(self.loss, self.train_variables), self.train_variables))
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
