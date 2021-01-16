"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""

import numpy as np
import os

import tensorflow as tf

import input_data
from networks.robust_network import get_network

from foolbox import TensorFlowModel, accuracy, Model
from foolbox.attacks import LinfPGD, FGSM, FGM

def test(config):

    # Setting up testing parameters
    seed = config['random_seed']
    tf.random.set_seed(seed)
    batch_size = config['training_batch_size']
    data_set = config['data_set']
    backbone_name = config['backbone']

    # Setting up the data and the model
    data = input_data.load_data_set(validation_size=config['validation_size'], data_set=data_set, seed=seed)
    num_features = data.train.images.shape[1]
    model = get_network(backbone_name, config, num_features)

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

    model.load_weights(tf.train.latest_checkpoint('./results/checkpoints/', latest_filename=None))


    raw_advs, clipped_advs, success = attack_linfpgd(fmodel, tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64),
                                             epsilons=epsilons_inf)
    robust_accuracy = 1 - success.numpy().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons_inf, robust_accuracy):
        print(f"  Linf norm < {eps:<6}: {acc.item() * 100:4.1f} %")


    raw_advs, clipped_advs, success = attack_l2fgsm(fmodel, tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64),
                                             epsilons=epsilons_inf)
    robust_accuracy = 1 - success.numpy().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons_inf, robust_accuracy):
        print(f"  L2 norm < {eps:<6}: {acc.item() * 100:4.1f} %")




