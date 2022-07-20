#!/usr/bin/env python

# Copyright 2018 Google LLC
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

"""Script to download all datasets and create .tfrecord files.
"""

import collections
import copy
import json

import cv2
import gzip
import os
import tarfile
import tempfile
from os.path import join
from urllib import request

import imageio
import numpy as np
import scipy.io
import tensorflow as tf
from absl import app
from tqdm import trange

from libml import data as libml_data
from libml.utils import EasyDict, my_datasets, calc_size

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
    'stl10': 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
}


def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits


def _load_stl10():
    def unflatten(images):
        return np.transpose(images.reshape((-1, 3, 96, 96)),
                            [0, 3, 2, 1])

    with tempfile.NamedTemporaryFile() as f:
        if tf.gfile.Exists('stl10/stl10_binary.tar.gz'):
            f = tf.gfile.Open('stl10/stl10_binary.tar.gz', 'rb')
        else:
            request.urlretrieve(URLS['stl10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_X = tar.extractfile('stl10_binary/train_X.bin')
        train_y = tar.extractfile('stl10_binary/train_y.bin')

        test_X = tar.extractfile('stl10_binary/test_X.bin')
        test_y = tar.extractfile('stl10_binary/test_y.bin')

        unlabeled_X = tar.extractfile('stl10_binary/unlabeled_X.bin')

        train_set = {'images': np.frombuffer(train_X.read(), dtype=np.uint8),
                     'labels': np.frombuffer(train_y.read(), dtype=np.uint8) - 1}

        test_set = {'images': np.frombuffer(test_X.read(), dtype=np.uint8),
                    'labels': np.frombuffer(test_y.read(), dtype=np.uint8) - 1}

        _imgs = np.frombuffer(unlabeled_X.read(), dtype=np.uint8)
        unlabeled_set = {'images': _imgs,
                         'labels': np.zeros(100000, dtype=np.uint8)}

        fold_indices = tar.extractfile('stl10_binary/fold_indices.txt').read()

    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    unlabeled_set['images'] = _encode_png(unflatten(unlabeled_set['images']))
    return dict(train=train_set, test=test_set, unlabeled=unlabeled_set,
                files=[EasyDict(filename="stl10_fold_indices.txt", data=fold_indices)])

def _load_my_dataset(name):
    def _load_internal():

        dataset_path = "/data-ssd/%s" % name

        def unflatten(images):
            return np.transpose(images.reshape((-1, 3, 96, 96)),
                                [0, 3, 2, 1])

        unlabeled_data_available = os.path.exists(join(dataset_path, "unlabeled"))
        test_data_available = os.path.exists(join(dataset_path, "test"))

        folders = ["train","val"] + (["unlabeled"] if unlabeled_data_available else []) + (["test"] if test_data_available else [])
        sets = []
        indices = []

        if not os.path.exists(dataset_path):
            print("DATA NOT FOUND - STOP")
            return None

        for folder in folders:
            classes = sorted([f for f in os.listdir(join(dataset_path,folder)) if os.path.isdir(join(dataset_path,folder,f)) ])

            print(folder, classes)

            imgs = []
            labels = []
            index = []
            for i, cl in enumerate(sorted(classes)):
                files = os.listdir(join(dataset_path,folder,cl))

                for file in sorted(files):
                    file_name = join(dataset_path,folder,cl,file)
                    im = imageio.imread(file_name)

                    w_h = calc_size(name)

                    im = cv2.resize(im, (w_h[1], w_h[0]))

                    if im.ndim == 2:
                        # should be loaded as bgr instead of grayscale
                        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

                    imgs.append(im)
                    labels.append(i)
                    index.append(file_name)

            set_entry = {'images': np.array(imgs),
                         'labels': np.array(labels)}
            sets.append(set_entry)
            indices.append(json.dumps(index))


        train_set = sets[0]
        test_set = sets[1]


        train_set['images'] = _encode_png(train_set['images'] )
        test_set['images'] = _encode_png(test_set['images'])

        result = dict(train=train_set, test=test_set,
                    files=[EasyDict(filename="%s_train_indices.txt" % name, data=indices[0]),
                           EasyDict(filename="%s_test_indices.txt" % name, data=indices[1]),
                           ])
        if unlabeled_data_available:
            unlabeled_set = sets[2]
            unlabeled_set['images'] = _encode_png(unlabeled_set['images'])
            result["unlabeled"] = unlabeled_set
            result["files"].append(EasyDict(filename="%s_unlabeled_indices.txt" % name, data=indices[2]))

        if test_data_available:
            unlabeled_set = sets[-1]
            unlabeled_set['images'] = _encode_png(unlabeled_set['images'])
            result["evaluation"] = unlabeled_set
            result["files"].append(EasyDict(filename="%s_evaluation_indices.txt" % name, data=indices[-1]))

        return result

    return _load_internal

def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar100():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar100'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/train.mat'))
        train_set = {'images': data_dict['data'],
                     'labels': data_dict['fine_labels'].flatten()}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/test.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['fine_labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_fashionmnist():
    def _read32(data):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(data.read(4), dtype=dt)[0]

    image_filename = '{}-images-idx3-ubyte'
    label_filename = '{}-labels-idx1-ubyte'
    split_files = [('train', 'train'), ('test', 't10k')]
    splits = {}
    for split, split_file in split_files:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['fashion_mnist'].format(image_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2051
                n_images = _read32(data)
                row = _read32(data)
                col = _read32(data)
                images = np.frombuffer(data.read(n_images * row * col), dtype=np.uint8)
                images = images.reshape((n_images, row, col, 1))
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['fashion_mnist'].format(label_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2049
                n_labels = _read32(data)
                labels = np.frombuffer(data.read(n_labels), dtype=np.uint8)
        splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(libml_data.DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(libml_data.DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not tf.gfile.Exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        tf.gfile.MakeDirs(os.path.join(libml_data.DATA_DIR, folder))
    for filename, contents in files.items():
        with tf.gfile.Open(os.path.join(libml_data.DATA_DIR, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return tf.gfile.Exists(os.path.join(libml_data.DATA_DIR, name, folder))


CONFIGS = dict(
    # commented out because not of interest
    # cifar10=dict(loader=_load_cifar10, checksums=dict(train=None, test=None)),
    # cifar100=dict(loader=_load_cifar100, checksums=dict(train=None, test=None)),
    # svhn=dict(loader=_load_svhn, checksums=dict(train=None, test=None, extra=None)),
    # stl10=dict(loader=_load_stl10, checksums=dict(train=None, test=None)),
)


def main(argv):
    # add my datasets
    for name in my_datasets:
        CONFIGS[name] = {"loader": _load_my_dataset(name), "checksums": dict(train=None, test=None)}

    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    tf.gfile.MakeDirs(libml_data.DATA_DIR)

    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        if datas is None:
            # save guard against ill loaders
            continue
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(libml_data.DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with tf.gfile.Open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(libml_data.DATA_DIR, file_and_data.filename)
                    with tf.gfile.Open(path, "wb") as f:
                        f.write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))


if __name__ == '__main__':
    app.run(main)
