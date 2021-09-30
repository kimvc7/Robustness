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
    def train_step(self, input, label, epsilon, robust=True, type_robust='linf',):
        self._full_call(input, label, epsilon, robust=robust,
                        evaluate=False, type_robust=type_robust)

    @tf.function
    def evaluate(self, input, label, epsilon, step=-1, summary=None, type_robust='linf', evaluate_bound=False):
        self._full_call(input, label, epsilon, step=step, robust=True,
                        evaluate=True, summary=summary, type_robust=type_robust)

    def evaluate_bound(self, input, label, epsilon):
        self._full_call(input, label, epsilon,  robust=True,  type_robust='certificate',
                      evaluate=True)

    def evaluate_approx_bound(self, input, label, epsilon, type_robust):
        self._full_call(input, label, epsilon,  robust=True,  type_robust=type_robust,
                      evaluate=True, evaluate_bound=True)

    def _full_call(self, input, label, epsilon, robust=True, evaluate=False, evaluate_bound=False,
                 summary=None, step=-1, type_robust='linf'):

        self.x_input = input
        self.y_input = label

        if not robust: #vanilla training

            with tf.GradientTape() as self.tape:

                self.feedforward_pass(self.x_input)

                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                      labels=tf.cast(self.y_input, tf.int32), logits=self.pre_softmax)
                self.loss = tf.reduce_mean(y_xent)

        else: #robust training

            if type_robust == "clipping":
                self.M = tf.math.minimum(1 - self.x_input, epsilon)
                self.m = tf.math.maximum(-self.x_input, -epsilon)

            if not type_robust == "certificate":

                with tf.GradientTape() as self.tape:
                    with tf.GradientTape(persistent=True) as self.second_tape:

                        self.second_tape.watch(self.x_input)
                        self.feedforward_pass(self.x_input)

                        if type_robust == 'grad': #linf
                            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.y_input, logits=self.pre_softmax)
                            self.xent = tf.reduce_mean(y_xent)
                            self.loss = self.xent + epsilon*self.second_tape.gradient(self.xent, self.x_input)

                        else:
                            # Compute linear approximation for robust cross-entropy.
                            data_range = tf.range(tf.shape(self.y_input)[0])
                            indices = tf.map_fn(lambda n: tf.stack([n, tf.cast(self.y_input[n], tf.int32)]), data_range)

                            self.nom_exponent = []
                            for i in range(self.num_classes):
                                self.nom_exponent += [self.pre_softmax[:,i] - tf.gather_nd(self.pre_softmax, indices)]

                            sum_exps = 0
                            for i in range(self.num_classes):
                                grad = self.second_tape.gradient(self.nom_exponent[i], self.x_input)
                                if type_robust == 'clipping': #linf
                                    positive_terms = tf.math.multiply(self.M, tf.nn.relu(grad[0]))
                                    negative_terms = tf.math.multiply(self.m, tf.nn.relu(-grad[0]))
                                    exponent = tf.reduce_sum(positive_terms - negative_terms, axis=1) + self.nom_exponent[i]
                                elif type_robust == 'l1':
                                    exponent = np.sqrt(self.num_features) * epsilon * tf.reduce_max(tf.abs(grad), axis=1) + \
                                               self.nom_exponent[i]
                                elif type_robust == 'l1+inf':
                                    exponent = np.sqrt(self.num_features) * epsilon * tf.reduce_max(tf.abs(grad), axis=1) + \
                                             epsilon * tf.reduce_sum(tf.abs(grad), axis=1) + self.nom_exponent[i]
                                else: #linf
                                    exponent = epsilon * tf.reduce_sum(tf.abs(grad), axis=1) + self.nom_exponent[i]
                                sum_exps += tf.math.exp(exponent)

                            self.loss = tf.reduce_mean(tf.math.log(sum_exps))

            else: #certificate objective:
                if not evaluate:
                    with tf.GradientTape() as self.tape:

                        self.feedforward_pass(self.x_input)

                        #L1 certificate -- epsilon is converted
                        self.loss, self.acc_bound = self.certificate_loss(np.sqrt(self.num_features) * epsilon, label)

                else:
                    self.feedforward_pass(self.x_input)

                    self.loss, self.acc_bound = self.certificate_loss(epsilon, label)

                    self.acc_bound = (self.acc_bound).numpy()

        if not evaluate:

            self.optimizer.apply_gradients(zip(self.tape.gradient(self.loss, self.train_variables),
                                               self.train_variables))
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

            if evaluate_bound==True:
                self.eval_approx_bound = (self.loss).numpy()
                self.eval_xent = (self.xent).numpy()

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