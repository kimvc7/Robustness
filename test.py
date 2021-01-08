"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import json
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf

from networks.MLP import robustMLP as Net
import input_data

from foolbox import TensorFlowModel, accuracy, Model
from foolbox.attacks import LinfPGD, FGSM, FGM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="mnist",
                            help="number of subsets")
parser.add_argument("--val_size", type=float, default=128,
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

attack_linfpgd = LinfPGD()
epsilons_inf = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]

attack_l2fgsm = FGM()
epsilons_inf = list(10*np.array([
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]))

#Setting up data for testing and validation
x_test = data.validation.images
y_test = data.validation.labels.reshape(-1)

x_batch, y_batch = data.train.next_batch(batch_size)

model.load_weights('./results/checkpoints/6000')

raw_advs, clipped_advs, success = attack_linfpgd(fmodel, tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64),
                                         epsilons=epsilons_inf)
robust_accuracy = 1 - success.numpy().mean(axis=-1)
print("robust accuracy for perturbations with")
for eps, acc in zip(epsilons_inf, robust_accuracy):
    print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


raw_advs, clipped_advs, success = attack_l2fgsm(fmodel, tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64),
                                         epsilons=epsilons_inf)
robust_accuracy = 1 - success.numpy().mean(axis=-1)
print("robust accuracy for perturbations with")
for eps, acc in zip(epsilons_inf, robust_accuracy):
    print(f"  L2 norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")




