"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""

import numpy as np
import pickle
import tensorflow as tf
import sys
import os

import input_data
from networks.robust_network import get_network

from foolbox import TensorFlowModel, Model
from foolbox.attacks import LinfPGD, FGSM, FGM, L2PGD, L1PGD, L1FastGradientAttack


def test(config):

    # Setting up testing parameters
    seed = config['random_seed']
    tf.random.set_seed(seed)
    batch_size = config['training_batch_size']
    backbone_name = config['backbone']

    if not os.path.isfile(config["model_dir"] + '/results/training.done'):
        print("Not trained") 
        #return

    if os.path.isfile(config["model_dir"] + '/results/testing_extra.done') and not config["restart"]:
        print("Already tested")
        return

    # Setting up the data and the model
    data = input_data.load_data_set(results_dir=config['results_dir'], data_set=config['data_set'],
                                    standarized=config["standarize"], multiplier=config["standarize_multiplier"],
                                    re_size=config["re_size"], seed=seed)

    num_features = data.train.images.shape[1]
    model = get_network(backbone_name, config, num_features)

    if config['backbone'] == 'Madry': # Load pretrained model for fair comparison
        model.load_Madry(config['model_dir'])
    else:
        model.load_all(tf.train.latest_checkpoint(config['model_dir'] + '/checkpoints/'), load_optimizer=False)

    model.set_mode('test')
    #Setting up attacks
    pre = dict(std=None, mean=None)  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(config["bound_lower"], config["bound_upper"]), preprocessing=pre)
    fmodel = fmodel.transform_bounds((config["bound_lower"], config["bound_upper"]))

    ''' 
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
    '''

    epsilons_inf = [
            2.0,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            50.0
        ]
    epsilons_l2 = list(np.sqrt(num_features) * np.array(epsilons_inf))#list(10 * np.array(epsilons_inf))#

    attacks = [L1PGD(), L1FastGradientAttack(), LinfPGD(), L2PGD(), FGSM(), FGM()]#[L1PGD(), L1FastGradientAttack(), L2PGD(), FGM()] #[LinfPGD(), L2PGD(), FGSM(), FGM()]#
    name_attacks = ["l1_pgd_norm", "l1_fgm_norm", "linf_pgd", "l2_pgd_norm", "linf_fgsm", "l2_fgm_norm"] #["l1_pgd_norm", "l1_fgm_norm", "l2_pgd_norm", "l2_fgm_norm"] #["linf_pgd", "l2_pgd", "linf_fgsm", "l2_fgm"]#
    epsilons = [epsilons_l2, epsilons_l2, epsilons_inf, epsilons_l2, epsilons_inf, epsilons_l2]#[epsilons_l2, epsilons_l2, epsilons_l2, epsilons_l2]#[epsilons_inf, epsilons_l2, epsilons_inf, epsilons_l2]#

    num_iter = int(10)# * (int(256/batch_size)))
    for attack, name_attack, epsilon in zip(attacks, name_attacks, epsilons):

        for dataset in ["val", "test"]:

            for iter in range(num_iter):

                if dataset == "val":
                    x_batch, y_batch = data.validation.next_batch(batch_size)
                else:
                    x_batch, y_batch = data.test.next_batch(batch_size)

                raw_advs, clipped_advs, success = attack(fmodel, tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                                                         epsilons=epsilon)

                robust_accuracy = 1 - success.numpy().mean(axis=-1)

                if iter == 0:
                    acc = dict(zip(epsilon, robust_accuracy))
                else:
                    tmp = dict(zip(epsilon, robust_accuracy))
                    acc = {k: acc[k] + tmp[k] for k in acc.keys()}

                if (iter == 0) and dataset == 'test':
                    with open(config['model_dir'] + '/results/examples2' + name_attack + '.pkl', 'wb') as f:
                        pickle.dump([clipped_advs, raw_advs], f)

            acc = {k: acc[k]/num_iter for k in acc.keys()}

            with open(config['model_dir'] + '/results/acc2_' + dataset + '_' + name_attack + '.pkl', 'wb') as f:
                pickle.dump(acc, f)

        print("\n Attack " + name_attack + " done")
        sys.stdout.flush()

    open(config['model_dir'] + '/results/testing_extra.done', 'w').close()




