"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import json
import argparse
from timeit import default_timer as timer

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf

from networks.CNN import robustCNN as Net
import input_data

from foolbox import TensorFlowModel, accuracy, Model
from foolbox.attacks import LinfPGD, FGSM, FGM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="mnist",
                            help="number of subsets")
parser.add_argument("--val_size", type=float, default=10000,
                            help="validation percent of the data e.g., 0.25 means 0.25*traning size")

with open('config.json') as config_file:
    config = json.load(config_file)


args = parser.parse_args()
print(args)

# Setting up training parameters
seed = config['random_seed']
tf.random.set_seed(seed)
batch_size = config['batch_size']
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
data_set = args.data_set
initial_learning_rate = config['initial_learning_rate']
eta = config['constant_learning_rate']




#Setting up the data and the model
data = input_data.load_data_set(validation_size=args.val_size, data_set=data_set, seed=seed)
num_features = data.train.images.shape[1]
model = Net(num_features, initial_learning_rate, batch_size)

#Setting up attacks
pre = dict(std=None, mean=None)  # RGB to BGR
fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
fmodel = fmodel.transform_bounds((0, 255))
attack = LinfPGD()
epsilons = [
    0.0,
    0.1,
    2.0,
]


#Setting up data for testing and validation
val_dict = {'x_input': data.validation.images,
            'y_input': data.validation.labels.reshape(-1)}
x_test = data.test.images
y_test = data.test.labels.reshape(-1)


model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize the summary writer, global variables, and our time counter.
summary_writer = tf.summary.create_file_writer(model_dir + "/Natural")
summary_writer1 = tf.summary.create_file_writer(model_dir + "/Test")
summary_writer2 = tf.summary.create_file_writer(model_dir + "/Adversarial")


# Main training loop
best_val_acc, test_acc, num_iters = 0, 0, 0
training_time = 0.0
for ii in range(max_num_training_steps):
    x_batch, y_batch = data.train.next_batch(batch_size)

    if ii % num_output_steps == 0:
        print('Step {}:    ({})'.format(ii, datetime.now()))

        model.evaluate(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                        summary=summary_writer1, step=ii)
        model.evaluate(tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64),
                        summary=summary_writer, step=ii)


        raw_advs, clipped_advs, success = attack(fmodel, tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                                                 epsilons=epsilons)
        model.evaluate(tf.cast(clipped_advs[1], tf.float32), tf.cast(y_batch, tf.int64),
                         summary=summary_writer2, step=ii)
        robust_accuracy = 1 - success.numpy().mean(axis=-1)
        print("robust accuracy for perturbations with")
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

        if ii != 0:
            print('    {} examples per second'.format(
                num_output_steps * batch_size / training_time))
            training_time = 0.0

    # Actual training step
    start = timer()
    model.train_step(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64), robust=True)
    end = timer()
    training_time += end - start

