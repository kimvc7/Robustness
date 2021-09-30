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
from foolbox import TensorFlowModel, Model
from foolbox.attacks import LinfPGD, FGSM, FGM, L2PGD, L1PGD, L1FastGradientAttack

def test_approx_bound(config):

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

    if os.path.isfile(config["model_dir"] + '/results/testing_approx_bound.done') and not config["restart"]:
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
    #Setting up attacks
    pre = dict(std=None, mean=None)  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(config["bound_lower"], config["bound_upper"]), preprocessing=pre)
    fmodel = fmodel.transform_bounds((config["bound_lower"], config["bound_upper"]))


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
    epsilons_l1 = list(np.sqrt(num_features) * np.array(epsilons_inf))

    tf.executing_eagerly()
    num_iter = 2*FACTOR_MEMORY

    for type_robust in [ "l1", "linf"]:
        for dataset in ["test"]:
            out = {}
            for iter in range(num_iter):

                if dataset == "val":
                    x_batch, y_batch = data.validation.next_batch(batch_size)
                else:
                    x_batch, y_batch = data.test.next_batch(batch_size)

                tmp_bound = []
                tmp_xent = []

                if type_robust == "l1":
                    epsilons = epsilons_l1
                    attack = L1PGD()
                else:
                    epsilons = epsilons_inf
                    attack = LinfPGD()

                for epsilon in epsilons:

                    model.evaluate_approx_bound(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                                                     epsilon=epsilon, type_robust=type_robust)
                    tmp_bound += [model.eval_approx_bound]

                raw_advs, clipped_advs, success = attack(fmodel, tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                                                         epsilons=epsilons)
                for idx, epsilon in enumerate(epsilons):
                    model.evaluate_approx_bound(tf.cast(clipped_advs[idx], tf.float32), tf.cast(y_batch, tf.int64),
                                   epsilon=tf.cast(epsilon, tf.float32), type_robust=type_robust)
                    tmp_xent += [model.eval_xent]

                if iter == 0:
                    out['bound'] = dict(zip(epsilons_inf, tmp_bound))
                    out['xent'] = dict(zip(epsilons_inf, tmp_xent))
                else:
                    tmp = dict(zip(epsilons_inf, tmp_bound))
                    out['bound'] = {k: out['bound'][k] + tmp[k] for k in out['bound'].keys()}
                    tmp = dict(zip(epsilons_inf, tmp_xent))
                    out['xent'] = {k: out['xent'][k] + tmp[k] for k in out['xent'].keys()}

            out['bound'] = {k: out['bound'][k]/num_iter for k in out['bound'].keys()}
            out['xent'] = {k: out['xent'][k] / num_iter for k in out['xent'].keys()}
            print(out)

            with open(config['model_dir'] + '/results/acc_' + dataset + '_' + 'approx_bound_' + type_robust + '.pkl', 'wb') as f:
                pickle.dump(out, f)

    print("\n Bound accuracy done")
    sys.stdout.flush()

    open(config['model_dir'] + '/results/testing_approx_bound.done', 'w').close()




