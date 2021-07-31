"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""

import numpy as np
import pickle
import tensorflow as tf
import sys
import os
import numpy as np

import input_data
from networks.robust_network import get_network


def test_bound(config):

    FACTOR_MEMORY = 2
    config['training_batch_size'] = int(256/FACTOR_MEMORY) #***FOR GPU MEMORY CONSTRAINTS**
    # Setting up testing parameters
    seed = config['random_seed']
    tf.random.set_seed(seed)
    batch_size = config['training_batch_size']
    backbone_name = config['backbone']

    if not os.path.isfile(config["model_dir"] + '/results/training.done'):
        print("Not trained")
        return

    if os.path.isfile(config["model_dir"] + '/results/testing_bound.done') and not config["restart"]:
        print("Already tested")
        #return

    # Setting up the data and the model
    data = input_data.load_data_set(results_dir=config['results_dir'], data_set=config['data_set'],
                                    standarized=config["standarize"], multiplier=config["standarize_multiplier"], seed=seed)
    num_features = data.train.images.shape[1]
    model = get_network(backbone_name, config, num_features)

    if config['backbone'] == 'Madry': # Load pretrained model for fair comparison
        model.load_Madry(config['model_dir'])
    else:
        model.load_all(tf.train.latest_checkpoint(config['model_dir'] + '/checkpoints/'), load_optimizer=False)

    model.set_mode('test')

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

    tf.executing_eagerly()
    num_iter = 10*FACTOR_MEMORY
    for dataset in ["val", "test"]:

        for iter in range(num_iter):

            if dataset == "val":
                x_batch, y_batch = data.validation.next_batch(batch_size)
            else:
                x_batch, y_batch = data.test.next_batch(batch_size)

            tmp_acc = []

            for epsilon in epsilons_inf:

                model.evaluate_bound(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                                                 epsilon=epsilon*model.num_features)
                tmp_acc += [model.acc_bound]

            if iter == 0:
                acc = dict(zip(epsilons_inf, tmp_acc))
            else:
                tmp = dict(zip(epsilons_inf, tmp_acc))
                acc = {k: acc[k] + tmp[k] for k in acc.keys()}

        acc = {k: acc[k]/num_iter for k in acc.keys()}

        print(acc)

        with open(config['model_dir'] + '/results/acc_' + dataset + '_' + 'bound' + '.pkl', 'wb') as f:
            pickle.dump(acc, f)

    print("\n Bound accuracy done")
    sys.stdout.flush()

    open(config['model_dir'] + '/results/testing_bound.done', 'w').close()




