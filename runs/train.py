import tensorflow as tf

import os
import input_data
from timeit import default_timer as timer
from networks.robust_network import get_network
from datetime import datetime
import sys


def train(config):

    # Setting up training parameters
    seed = config['random_seed']
    tf.random.set_seed(seed)
    batch_size = config['training_batch_size']
    max_num_training_steps = config['max_num_training_steps']
    num_output_steps = config['num_output_steps']
    data_set = config['data_set']
    eval_attack_during_training = config['eval_attack_during_training']
    backbone_name = config['backbone']
    robust_training = config['robust_training']

    if eval_attack_during_training:
        from foolbox import TensorFlowModel, accuracy, Model
        from foolbox.attacks import LinfPGD, FGSM, FGM

    # Setting up the data and the model
    data = input_data.load_data_set(validation_size=config['validation_size'], data_set=data_set, seed=seed)
    num_features = data.train.images.shape[1]
    model = get_network(backbone_name, config, num_features)

    # Setting up attacks
    if eval_attack_during_training:
        pre = dict(std=None, mean=None)  # RGB to BGR
        fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
        fmodel = fmodel.transform_bounds((0, 255))
        attack = LinfPGD()
        epsilons = [0.1]

    model_dir = config['model_dir']
    if not os.path.exists(model_dir + '/checkpoints/'):
        os.makedirs(model_dir + '/checkpoints/')
        os.makedirs(model_dir + '/results/')
        start_iteration = 0
    elif config["restart"]:
        print("Restart training")
        import shutil
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        start_iteration = 0
    else:
        start_iteration = int(tf.train.latest_checkpoint(model_dir + '/checkpoints/').split('/')[-1])
        print("Reload existing checkpoint " + str(start_iteration))
        model.load_all(tf.train.latest_checkpoint(model_dir + '/checkpoints/'))

    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.create_file_writer(model_dir + "/Natural")
    summary_writer1 = tf.summary.create_file_writer(model_dir + "/Test")
    if eval_attack_during_training:
        summary_writer2 = tf.summary.create_file_writer(model_dir + "/Adversarial")
    sys.stdout.flush()

    # Main training loop
    training_time = 0.0
    for ii in range(start_iteration, max_num_training_steps):
        x_batch, y_batch = data.train.next_batch(batch_size)

        if ii % num_output_steps == 0:
            print('\n Step {} {}:    ({})\n'.format(model.optimizer.iterations.numpy(), ii, datetime.now()))

            # Setting up data for testing and validation
            x_test, y_test = data.validation.next_batch(batch_size)

            model.evaluate(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                            summary=summary_writer1, step=ii, robust=robust_training)
            model.evaluate(tf.cast(x_test, tf.float32), tf.cast(y_test, tf.int64),
                            summary=summary_writer, step=ii, robust=robust_training)

            if eval_attack_during_training:
                raw_advs, clipped_advs, success = attack(fmodel, tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64),
                            epsilons=epsilons)

                model.evaluate(tf.cast(clipped_advs[0], tf.float32), tf.cast(y_batch, tf.int64),
                            summary=summary_writer2, step=ii, robust=robust_training)

                robust_accuracy = 1 - success.numpy().mean(axis=-1)
                print("robust accuracy for perturbations with")
                for eps, acc in zip(epsilons, robust_accuracy):
                    print(f"  Linf norm < {eps:<6}: {acc.item() * 100:4.1f} %")

            if training_time != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))

                training_time = 0.0

                model.save_all(model_dir + '/checkpoints/' + str(ii))

            sys.stdout.flush()

        # Actual training step
        start = timer()
        model.train_step(tf.cast(x_batch, tf.float32), tf.cast(y_batch, tf.int64), robust=robust_training)
        end = timer()
        training_time += end - start

    open(model_dir + '/results/training.done', 'w').close()


