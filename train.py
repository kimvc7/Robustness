"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import sys
import subprocess

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#from CNN_model import Model
from NN_model import Model

from pgd_attack import LinfPGDAttack


import numpy as np
import input_data
import itertools

import argparse



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
tf.set_random_seed(seed)
batch_size = config['batch_size']
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
data_set = args.data_set
initial_learning_rate = config['initial_learning_rate']
eta = config['constant_learning_rate']
learning_rate = tf.train.exponential_decay(initial_learning_rate, 0, 5, 0.85, staircase=True)

global_step = tf.Variable(1, name="global_step")

#Setting up the data and the model
data = input_data.load_data_set(validation_size=args.val_size, data_set=data_set, seed=seed)
num_features = data.train.images.shape[1]
model = Model(num_features)

attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])


#Setting up data for testing and validation
val_dict = {model.x_input: data.validation.images,
                model.y_input: data.validation.labels.reshape(-1)}
test_dict = {model.x_input: data.test.images,
                model.y_input: data.test.labels.reshape(-1)}

# Setting up the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.robust_xent, global_step=global_step)


model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

with tf.Session() as sess:


  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir + "/Natural")
  summary_writer1 = tf.summary.FileWriter(model_dir+ "/Robust")
  summary_writer2 = tf.summary.FileWriter(model_dir+ "/Adversarial")

  sess.run(tf.global_variables_initializer())
  training_time = 0.0
    
  # Main training loop
  best_val_acc, test_acc, num_iters = 0, 0, 0

  for ii in range(max_num_training_steps):
    x_batch, y_batch = data.train.next_batch(batch_size)
    nat_dict = {model.x_input: x_batch, model.y_input: y_batch}

    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    adv_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}
    
    # Output
    if ii % num_output_steps == 0:

      #Arversarial batch dictionary
      x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      adv_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}

      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      test_acc = sess.run(model.accuracy, feed_dict=test_dict)
      nat_xent = sess.run(model.xent, feed_dict=nat_dict)
      test_xent = sess.run(model.xent, feed_dict=test_dict)
      adv_xent = sess.run(model.xent, feed_dict=adv_dict)
      robust_xent = sess.run(model.robust_xent, feed_dict=nat_dict)
      test_robust_xent = sess.run(model.robust_xent, feed_dict=test_dict)


      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    batch nat accuracy {:.4}'.format(nat_acc * 100))
      print('    batch adv accuracy {:.4}'.format(adv_acc * 100))
      print('    testing nat accuracy {:.4}'.format(test_acc * 100))
      print('    Batch Nat Xent {:.4}'.format(nat_xent))
      print('    Batch Adv Xent {:.4}'.format(adv_xent))
      print('    Batch Robust Xent {:.4}'.format(robust_xent))
      print('    Testing Nat Xent {:.4}'.format(test_xent))
      print('    Testing Robust Xent {:.4}'.format(test_robust_xent))


      
      summary = tf.Summary(value=[
          tf.Summary.Value(tag='Xent', simple_value= nat_xent),
          tf.Summary.Value(tag='Acc', simple_value= nat_acc),
          tf.Summary.Value(tag='Test Xent', simple_value= test_xent),
          tf.Summary.Value(tag='Test Acc', simple_value= test_acc)])
      summary1 = tf.Summary(value=[
          tf.Summary.Value(tag='Xent', simple_value= robust_xent),
          tf.Summary.Value(tag='Testing Xent', simple_value= test_xent)])
      summary2 = tf.Summary(value=[
          tf.Summary.Value(tag='Xent', simple_value= adv_xent),
          tf.Summary.Value(tag='Acc', simple_value= adv_acc)])
      summary_writer.add_summary(summary, global_step.eval(sess))
      summary_writer1.add_summary(summary1, global_step.eval(sess))
      summary_writer2.add_summary(summary2, global_step.eval(sess))
      


      if ii != 0:
        print('    {} examples per second'.format(
        num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Actual training step
    start = timer()
    sess.run(optimizer, feed_dict=nat_dict)
    end = timer()
    training_time += end - start


  x_batch_adv_test = attack.perturb(data.test.images, data.test.labels, sess)
  adv_test_dict = {model.x_input: x_batch_adv_test, model.y_input: data.test.labels}
  adv_test_acc = sess.run(model.accuracy, feed_dict=adv_test_dict)
  test_acc = sess.run(model.accuracy, feed_dict=test_dict)
  print("test adv accuracy:", adv_test_acc)
  print("test nat accuracy:", test_acc)
    

