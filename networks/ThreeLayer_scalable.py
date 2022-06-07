"""
The model is a multiclass perceptron for 10 classes.
"""
import tensorflow as tf
import numpy as np
import robustify_network


class robustThreeLayer_scalable(robustify_network.RobustifyNetwork):

    def __init__(self, config, num_features):
        super().__init__(config['num_classes'], config['epsilon'])

        l1_size = config['l1_size']
        initial_learning_rate = float(config['initial_learning_rate'])
        training_batch_size = config['training_batch_size']
        num_classes = config['num_classes']
        batch_decrease_learning_rate = float(config['batch_decrease_learning_rate'])

        self.batch_size = training_batch_size
        self.mode = 'train'
        self.num_features = num_features
        self.l1_size = l1_size

        self.train_variables = []

        self.W1 = self._weight_variable([num_features, l1_size])
        self.train_variables += [self.W1]
        self.b1 = self._bias_variable([l1_size])
        self.train_variables += [self.b1]

        self.W2 = self._weight_variable([l1_size, l1_size])
        self.train_variables += [self.W2]
        self.b2 = self._bias_variable([l1_size])
        self.train_variables += [self.b2]

        self.W3 = self._weight_variable([l1_size, l1_size])
        self.train_variables += [self.W3]
        self.b3 = self._bias_variable([l1_size])
        self.train_variables += [self.b3]

        self.W4 = self._weight_variable([l1_size, num_classes])
        self.train_variables += [self.W4]
        self.b4 = self._bias_variable([num_classes])
        self.train_variables += [self.b4]

        # Setting up the optimizer
        self.learning_rate = \
            tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                    training_batch_size * batch_decrease_learning_rate, 0.85, staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def feedforward_pass(self, input):

        self.batch_size = input.shape[0]

        self.z1 = tf.matmul(input, self.W1) + self.b1
        self.h1 = tf.nn.relu(self.z1)

        self.z2 = tf.matmul(self.h1, self.W2) + self.b2
        self.h2 = tf.nn.relu(self.z2)

        self.z3 = tf.matmul(self.h2, self.W3) + self.b3
        self.h3 = tf.nn.relu(self.z3)

        self.pre_softmax = tf.matmul(self.h3, self.W4) + self.b4

        return self.pre_softmax

    def set_mode(self, mode='train'):
        self.mode = mode

    def sigma(self, x):
        return tf.nn.relu(x)

    def certificate_loss(self, epsilon, labels):

        labels_r = tf.repeat(labels, repeats = self.num_features, axis = 0)

        self.g_1_pos = tf.reshape(epsilon*self.W1[None] + self.z1[:, None],
                                  [self.num_features*self.batch_size, self.l1_size])
        self.g_1_neg = tf.reshape(-epsilon*self.W1[None] + self.z1[:, None],
                                  [self.num_features*self.batch_size, self.l1_size])

        self.g_2_pos1 = tf.matmul(self.sigma(self.g_1_pos),
                                  self.sigma(self.W2)) - tf.matmul(self.sigma(self.g_1_pos), self.sigma(-self.W2))
        self.g_2_pos2 = tf.matmul(self.sigma(self.g_1_pos),
                                  self.sigma(-self.W2)) - tf.matmul(self.sigma(self.g_1_pos), self.sigma(self.W2))
        self.g_2_neg1 = tf.matmul(self.sigma(self.g_1_neg),
                                  self.sigma(self.W2)) - tf.matmul(self.sigma(self.g_1_neg), self.sigma(-self.W2))
        self.g_2_neg2 = tf.matmul(self.sigma(self.g_1_neg),
                                  self.sigma(-self.W2)) - tf.matmul(self.sigma(self.g_1_neg), self.sigma(self.W2))

        self.g_3_pos1 = tf.matmul(self.sigma(self.g_2_pos1 + self.b2),
                                  self.sigma(self.W3)) - tf.matmul(self.sigma(-self.g_2_pos2 + self.b2), self.sigma(-self.W3))
        self.g_3_pos2 = tf.matmul(self.sigma(self.g_2_pos1 + self.b2),
                                  self.sigma(-self.W3)) - tf.matmul(self.sigma(-self.g_2_pos2 + self.b2), self.sigma(self.W3))
        self.g_3_neg1 = tf.matmul(self.sigma(self.g_2_neg1 + self.b2),
                                  self.sigma(self.W3)) - tf.matmul(self.sigma(-self.g_2_neg2 + self.b2), self.sigma(-self.W3))
        self.g_3_neg2 = tf.matmul(self.sigma(self.g_2_neg1 + self.b2),
                                  self.sigma(-self.W3)) - tf.matmul(self.sigma(-self.g_2_neg2 + self.b2), self.sigma(self.W3))

        robust_objective = 0
        robust_acc = 0

        for k in range(self.num_classes):
            mask = tf.equal(labels_r, k)
            g_3k_pos1 = tf.boolean_mask(self.g_3_pos1, mask)
            g_3k_pos2 = tf.boolean_mask(self.g_3_pos2, mask)
            g_3k_neg1 = tf.boolean_mask(self.g_3_neg1, mask)
            g_3k_neg2 = tf.boolean_mask(self.g_3_neg2, mask)
            W4_k = self.W4 - tf.gather(self.W4, [k], axis=1)

            g_4k_pos = tf.matmul(self.sigma(g_3k_pos1 + self.b3), self.sigma(W4_k)) - tf.matmul(self.sigma(-g_3k_pos2 + self.b3), self.sigma(-W4_k))
            g_4k_neg = tf.matmul(self.sigma(g_3k_neg1 + self.b3), self.sigma(W4_k)) - tf.matmul(self.sigma(-g_3k_neg2 + self.b3), self.sigma(-W4_k))
            g_4k_max = tf.maximum(g_4k_pos, g_4k_neg)
            g_4k = tf.nn.max_pool(tf.reshape(g_4k_max, [1, tf.shape(g_4k_max)[0], self.num_classes, 1]), [1, self.num_features, 1, 1], [1, self.num_features, 1, 1], "SAME") + \
                   tf.reshape(self.b4  - self.b4[k], [1, 1, self.num_classes, 1])

            robust_acc +=  tf.reduce_sum(tf.cast(tf.reduce_all(tf.less_equal(g_4k, tf.constant([0.0])), axis = 2), tf.float32))
            robust_objective += tf.reduce_sum(tf.reduce_logsumexp(g_4k, axis = 2))

        self.acc_bound = robust_acc/tf.cast(tf.shape(labels)[0], tf.float32)
        self.loss = robust_objective/tf.cast(tf.shape(labels)[0], tf.float32)
        return self.loss, self.acc_bound

    @staticmethod
    def _weight_variable(shape):
        initial = tf.keras.initializers.GlorotUniform()
        return tf.Variable(initial_value=initial(shape), name=str(np.random.randint(1e10)), trainable=True)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
