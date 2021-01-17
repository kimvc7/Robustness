"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""

import numpy as np
import pickle
import tensorflow as tf
import sys

import input_data
from networks.robust_network import get_network

from foolbox import TensorFlowModel, Model
from foolbox.attacks import LinfPGD, FGSM, FGM, L2PGD

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

    if config['name_net'] == 'Madry': # Load pretrained model for fair comparison
        model.load_Madry(config['model_dir'])
    else:
        model.load_all(tf.train.latest_checkpoint(config['model_dir'] + '/checkpoints/'), load_optimizer=False)

    #Setting up attacks
    pre = dict(std=None, mean=None)  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
    fmodel = fmodel.transform_bounds((0, 255))

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
    epsilons_l2 = list(10*np.array(epsilons_inf))

    attacks = [LinfPGD(), L2PGD(), FGSM(), FGM()]
    name_attacks = ["linf_pgd", "l2_pgd", "linf_fgsm", "l2_fgm"]
    epsilons = [epsilons_inf, epsilons_l2, epsilons_inf, epsilons_l2]

    num_iter = 10
    for attack, name_attack, epsilon in zip(attacks, name_attacks, epsilons):

        for dataset in ["val", "test"]:

            for iter in range(num_iter):

                if dataset == "val":
                    x_batch, y_batch = data.validation.next_batch(batch_size)
                else:
                    x_batch, y_batch = data.test.next_batch(batch_size)

                clipped_advs_all = []
                raw_advs_all = []

                raw_advs, clipped_advs, success = attack(fmodel, tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                                                         epsilons=epsilon)

                robust_accuracy = 1 - success.numpy().mean(axis=-1)

                if iter == 0:
                    acc = dict(zip(epsilon, robust_accuracy))
                else:
                    tmp = dict(zip(epsilon, robust_accuracy))
                    acc = {k: acc[k] + tmp[k] for k in acc.keys()}

                if (iter == 0) and dataset == 'test':
                    with open(config['model_dir'] + '/results/examples' + name_attack + '.pkl', 'wb') as f:
                        pickle.dump([clipped_advs_all, raw_advs_all], f)

            acc = {k: acc[k]/num_iter for k in acc.keys()}

            with open(config['model_dir'] + '/results/acc_' + dataset + '_' + name_attack + '.pkl', 'wb') as f:
                pickle.dump(acc, f)

        print("\n Attack " + name_attack + " done")
        sys.stdout.flush()

    open(config['model_dir'] + '/results/testing.done', 'w').close()




