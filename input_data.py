import tensorflow as tf

import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy
import json
import pickle

_Datasets = collections.namedtuple('_Datasets', ['train', 'validation', 'test'])

class _DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype,
               reshape,
               num_features,
               seed):
    """Construct a _DataSet.

    Args:
      images: The images
      labels: The labels
      dtype: Output image dtype. One of [uint8, float32]. `uint8` output has
        range [0,255]. float32 output has range [0,1].
      reshape: Bool. If True returned images are returned flattened to vectors.
      num_subsets: Number of training subsets for stability
      subset_ratio: fraction of original training set that must be in each subset.
    """
     # Convert shape from [num examples, rows, columns, depth]
     # to [num examples, rows*columns] (assuming depth == 1)

    seed1, seed2 = random_seed.get_seed(seed)
    numpy.random.seed(seed)
    if reshape:
      labels = labels.reshape(labels.shape[0])    
      images = images.reshape(images.shape[0], num_features)

    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch

    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self._images[perm0]
      self._labels = self._labels[perm0]

    # Go to the next epoch
    if start + batch_size > self._num_examples:

      # Finished epoch
      self._epochs_completed += 1

      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]

      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch

      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part),
                               axis=0), numpy.concatenate(
                                   (labels_rest_part, labels_new_part), axis=0)

    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def load_data_set(results_dir, data_set, seed=None, reshape=True, standarized=False, multiplier=1, dtype=dtypes.float32):

  with open(results_dir + 'configs_datasets/' + str(data_set) + '.json') as config_file:
    config = json.load(config_file)

  if config["dataset_name"] == "cifar" or config["dataset_name"] == "mnist" \
          or config["dataset_name"] == "fashion_mnist":
    if config["dataset_name"] == "cifar":
      (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
      num_features = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
    if config["dataset_name"] == "mnist":
      (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
      num_features = X_train.shape[1]*X_train.shape[2]
    if config["dataset_name"] == "fashion_mnist":
      (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
      num_features = X_train.shape[1]*X_train.shape[2]

    X_val = X_train[:config["validation_size"]]
    y_val = y_train[:config["validation_size"]]
    X_train = X_train[config["validation_size"]:]
    y_train = y_train[config["validation_size"]:]

    if standarized:

      def standarize(X):
        import numpy as np
        m = np.mean(X, axis=(1, 2))
        X = X - m[:, np.newaxis, np.newaxis]
        d = np.std(X, axis=(1, 2))
        X = X / d[:, np.newaxis, np.newaxis]
        X[np.isnan(X)] = 0
        return X

      X_train = multiplier*standarize(X_train)
      X_val = multiplier*standarize(X_val)
      X_test = multiplier*standarize(X_test)

  elif config["dataset_name"] == "Gauss_MLP-1":

    for set in ["train", "val", "test"]:
      with open(results_dir + 'datasets/' + set + "_" + config["name_file"], 'rb') as dataset_file:
        tmp = pickle.load(dataset_file)

      if set == "train":
        X_train = tmp["data"]
        y_train = tmp["labels"]
      elif set == "val":
        X_val = tmp["data"][:config["validation_size"]]
        y_val = tmp["labels"][:config["validation_size"]]
      else:
        X_test = tmp["data"][:config["testing_size"]]
        y_test = tmp["labels"][:config["testing_size"]]

      del tmp

  elif config["dataset_name"] == "UCI":

    for set in ["train", "test"]:
      import numpy as np
      tmpX = np.genfromtxt(results_dir + 'datasets/UCI/imp_' + config["name_file"] + '_' + set + "X.csv", delimiter=',')
      tmpX = np.nan_to_num(tmpX, nan=0.0)
      tmpY = np.genfromtxt(results_dir + 'datasets/UCI/' + config["name_file"] + '_' + set + "Y.csv", delimiter=',')

      if set == "train":
        num_samples = np.shape(tmpY)[0]
        num_train_samples = int(np.round(num_samples*0.75))
        X_train = tmpX[:num_train_samples]
        y_train = tmpY[:num_train_samples] - 1
        X_val = tmpX[num_train_samples:]
        y_val = tmpY[num_train_samples:] - 1
      else:
        X_test = tmpX
        y_test = tmpY - 1


      del tmpX
      del tmpY

    num_features = X_train.shape[1]


  print("There are", X_train.shape[0], "samples in the training set.")
  print("There are", X_val.shape[0], "samples in the validation set.")
  print("There are", X_train.shape[1], "features.")

  options = dict(dtype=dtype, reshape=reshape, num_features=num_features, seed=seed)

  train = _DataSet(X_train, y_train, **options)
  validation = _DataSet(X_val, y_val, **options)
  test = _DataSet(X_test, y_test, **options)

  return _Datasets(train=train, validation=validation, test=test)