# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities."""

import os
import pickle
import re
from os.path import join

import numpy as np
import tensorflow as tf
from absl import flags, logging
from tensorflow.python.client import device_lib

_GPUS = None
FLAGS = flags.FLAGS
flags.DEFINE_bool('log_device_placement', False, 'For debugging purpose.')


class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config():
    config = tf.ConfigProto()
    if len(get_available_gpus()) > 1:
        config.allow_soft_placement = True
    if FLAGS.log_device_placement:
        config.log_device_placement = True
    config.gpu_options.allow_growth = True
    return config


def setup_main():
    pass


def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.set_verbosity(logging.ERROR)


def smart_shape(x):
    s = x.shape
    st = tf.shape(x)
    return [s[i] if s[i].value is not None else st[i] for i in range(4)]


def ilog2(x):
    """Integer log2."""
    return int(np.ceil(np.log2(x)))


def find_latest_checkpoint(dir, glob_term='model.ckpt-*.meta'):
    """Replacement for tf.train.latest_checkpoint.

    It does not rely on the "checkpoint" file which sometimes contains
    absolute path and is generally hard to work with when sharing files
    between users / computers.
    """
    r_step = re.compile('.*model\.ckpt-(?P<step>\d+)\.meta')
    matches = tf.gfile.Glob(os.path.join(dir, glob_term))
    matches = [(int(r_step.match(x).group('step')), x) for x in matches]
    ckpt_file = max(matches)[1][:-5]
    return ckpt_file


def get_latest_global_step(dir):
    """Loads the global step from the latest checkpoint in directory.

    Args:
      dir: string, path to the checkpoint directory.

    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(find_latest_checkpoint(dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


def get_latest_global_step_in_subdir(dir):
    """Loads the global step from the latest checkpoint in sub-directories.

    Args:
      dir: string, parent of the checkpoint directories.

    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    sub_dirs = (x for x in tf.gfile.Glob(os.path.join(dir, '*')) if os.path.isdir(x))
    step = 0
    for x in sub_dirs:
        step = max(step, get_latest_global_step(x))
    return step


def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.

    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.

    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def gpu(x):
    return '/gpu:%d' % (x % max(1, len(get_available_gpus())))


def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        local_device_protos = device_lib.list_local_devices(session_config=config)
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS


def get_gpu():
    gpus = get_available_gpus()
    pos = 0
    while 1:
        yield gpus[pos]
        pos = (pos + 1) % len(gpus)


def average_gradients(tower_grads):
    # Adapted from:
    #  https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. For each tower, a list of its gradients.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    if len(tower_grads) <= 1:
        return tower_grads[0]

    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grad = tf.reduce_mean([gv[0] for gv in grads_and_vars], 0)
        average_grads.append((grad, grads_and_vars[0][1]))
    return average_grads


def para_list(fn, *args):
    """Run on multiple GPUs in parallel and return list of results."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return zip(*[fn(*args)])
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    return zip(*outputs)


def para_mean(fn, *args):
    """Run on multiple GPUs in parallel and return means."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.reduce_mean(x, 0) for x in zip(*outputs)]
    return tf.reduce_mean(outputs, 0)


def para_cat(fn, *args):
    """Run on multiple GPUs in parallel and return concatenated outputs."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.concat(x, axis=0) for x in zip(*outputs)]
    return tf.concat(outputs, axis=0)


def interleave(x, batch):
    s = x.get_shape().as_list()
    return tf.reshape(tf.transpose(tf.reshape(x, [-1, batch] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] + s[1:])


def de_interleave(x, batch):
    s = x.get_shape().as_list()
    return tf.reshape(tf.transpose(tf.reshape(x, [batch, -1] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] +s[1:])


def combine_dicts(*args):
    # Python 2 compatible way to combine several dictionaries
    # We need it because currently TPU code does not work with python 3
    result = {}
    for d in args:
        result.update(d)
    return result

def save_list(log_dir, name, list, save_float = False, save_string=False):
    """
    save list to path with name and timestamp
    :param name:
    :param out_path:
    :param list:
    :return:
    """

    # can save as numpy list if is numpy, at most 2 dimensions and number values
    save_as_csv = type(list) is np.ndarray and list.ndim <= 2
    file = join(log_dir, "%s.%s" % (name, "csv" if save_as_csv else 'pkl'))
    if save_as_csv:
        if not save_float and not save_string:
            np.savetxt(file, list.astype(int), delimiter=',', fmt="%i")
        if save_float:
            np.savetxt(file, list, delimiter=',', fmt="%0.5f")
        if save_string:
            np.savetxt(file, list, delimiter=',', fmt="%s")

    else:
        with open(file, 'wb') as f:
            pickle.dump(list, f)

def load_list(log_dir, name, is_string=False):
    possible_files = os.listdir(log_dir)
    # print(possible_files)
    # print(name)

    for file in possible_files:
        if name in file:
            # open file
            # print("open %s" % file)
            complete_path = join(log_dir,file)
            if ".csv" in file:
                list = np.loadtxt(complete_path, delimiter=',', dtype=str if is_string else float, comments="§")
            else:
                with open(complete_path, 'rb') as f:
                    list = pickle.load(f)

            # return
            return list
    return None

def combine_file_name(*parts):
    """
    combine filename with parts, but ignore nones
    :param parts:
    :return:
    """

    parts = [p for p in parts if p is not None]

    return "_".join(parts)


my_datasets = ['cifar10h', 'miceBone', 'plankton', 'turkey']

my_class_labels = {
    'turkey': sorted(['0','1']),
    "cifar10h" :  sorted(['bird', 'truck', 'automobile', 'airplane', 'dog', 'deer', 'cat', 'ship', 'frog', 'horse']),
    "miceBone":  sorted(['nr', 'g', 'ug']),
    "plankton" : sorted(['cop', 'pro_rhizaria_phaeodaria', 'collodaria_black', 'phyto_puff', 'det', 'no_fit', 'shrimp', 'bubbles', 'phyto_tuft', 'collodaria_globule']),
    }


my_class_weights = {"cifar10h" : [1,1,1,1,1, 1,1,1,1,1],
                   "miceBone":  [108, 453, 163],
                   "plankton" :  [514, 853, 709, 1405, 1057, 3710, 572, 1602, 1115, 743],
                    "turkey" :  [1300, 6740],
                   }


def calc_needs_balancing(name):
    if "balanced" in name:
        return True
    return False

def calc_size(name):
    """

    :param name:
    :return: w, h
    """
    w_h = (96,96)
    if "cifar10h" in name:
        w_h = (32,32)
    if "miceBone" in name:
        w_h = (192,192)
    return w_h

def calc_class_weights(name):
    """
    calculate the class weights for given dataset name
    :param name:
    :return:
    """

    assert name in my_datasets

    for key in my_class_weights:
        if key in name:

            # get numbers
            numbers = np.array(my_class_weights[key])

            weights =  np.sum(numbers) / (len(numbers) * numbers)
            print(f"Use weights {weights} for {name}")
            return weights

    assert False, "Should not be reached, the given dataset has no class labels provided for %s in %s" % (name,my_class_weights)

def calc_class_label(name):
    """
    calculate the class labels for given dataset name
    :param name:
    :return:
    """

    assert name in my_datasets

    for key in my_class_labels:
        if key in name:
            return my_class_labels[key]

    assert False, "Should not be reached, the given dataset has no class labels provided"