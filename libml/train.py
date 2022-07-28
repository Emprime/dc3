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
"""Training loop, checkpoint saving and loading, evaluation code."""
import datetime
import functools
import json
import os.path
import shutil
from os.path import join
from datetime import datetime
import numpy as np
import tensorflow as tf
from absl import flags
from sklearn import metrics
from tqdm import trange, tqdm

from libml import data, utils
from libml.utils import EasyDict, calc_class_label, save_list, combine_file_name

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('train_kimg', 1 << 14, 'Training duration in kibi-samples.')
flags.DEFINE_integer('report_kimg', 64, 'Report summary period in kibi-samples.')
flags.DEFINE_integer('save_kimg', 64, 'Save checkpoint period in kibi-samples.')
flags.DEFINE_integer('keep_ckpt', 5, 'Number of checkpoints to keep.')
flags.DEFINE_string('eval_ckpt', '', 'Checkpoint to evaluate. If provided, do not do training, just do eval.')
flags.DEFINE_string('rerun', '', 'A string to identify a run if running multiple ones with same parameters.')

flags.DEFINE_integer('overcluster_factor',5, 'Factor to multiply with the number of classes.')


class Model:
    def __init__(self, train_dir: str, dataset: data.DataSets, **kwargs):
        train_dir = join(train_dir, *FLAGS.IDs)
        self.train_dir =  train_dir#os.path.join(train_dir, FLAGS.rerun, self.experiment_name(**kwargs))
        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.ops = self.model(**kwargs)
        self.ops.update_step = tf.assign_add(self.step, FLAGS.batch)
        self.add_summaries(**kwargs)

        print(' Config '.center(80, '-'))
        print('train_dir', self.train_dir)
        print('%-32s %s' % ('Model', self.__class__.__name__))
        print('%-32s %s' % ('Dataset', dataset.name))
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in utils.model_vars(None)]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        self._create_initial_files()

    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    def train_print(self, text):
        self.tmp.print_queue.append(text)

    def _create_initial_files(self):
        for dir in (self.checkpoint_dir, self.arg_dir):
            tf.gfile.MakeDirs(dir)
        self.save_args()

    def _reset_files(self):
        shutil.rmtree(self.train_dir)
        self._create_initial_files()

    def save_args(self, **extra_params):
        with tf.gfile.Open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with tf.gfile.Open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    def eval_mode(self, ckpt=None):
        self.session = tf.Session(config=utils.get_config())
        saver = tf.train.Saver()
        if ckpt is None:
            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
        else:
            ckpt = os.path.abspath(ckpt)
        saver.restore(self.session, ckpt)
        self.tmp.step = self.session.run(self.step)
        print('Eval model %s at global_step %d' % (self.__class__.__name__, self.tmp.step))
        return self

    def model(self, **kwargs):
        raise NotImplementedError()

    def add_summaries(self, **kwargs):
        raise NotImplementedError()


class ClassifySemi(Model):
    """Semi-supervised classification."""

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int,  **kwargs):
        self.nclass = nclass
        self.overcluster_k  = int(FLAGS.overcluster_factor * self.nclass)
        self.num_heads  = int(FLAGS.num_heads)
        self.combined_output = FLAGS.combined_output
        Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

        # split the logging directory after log for a unqiue name
        self.log_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self.epoch = None # not init epoch


    def calculate_num_outputs(self):
        """
        calculate the outputs for for the semi-supervised model
        :return:
        """
        assert (not self.combined_output) or self.num_heads == 1,\
            "It is not allowed to use combined outputs with more than one head"

        if self.combined_output == 1 or self.combined_output == 3 or self.combined_output == 4: # sigmoid
            return self.nclass + self.overcluster_k + 1
        elif self.combined_output == 2: # combined
            return self.overcluster_k
        else:
            return self.nclass + (self.overcluster_k * self.num_heads)

    def determine_train_op(self):

        if self.ops.alt_train_op is None:
            return self.ops.train_op

        assert self.epoch is not None, "Epoch needs to be initiliazed"
        # alternating
        over_alternating = FLAGS.over_alternating
        normal_alternating = FLAGS.normal_alternating
        c = self.epoch % (over_alternating + normal_alternating)
        if c < normal_alternating:
            return self.ops.train_op
        else:
            return self.ops.alt_train_op

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()
        self.tmp.step = train_session.run([self.determine_train_op(), self.ops.update_step],
                                          feed_dict={self.ops.y: y['image'],
                                                     self.ops.xt: x['image'],
                                                     self.ops.label: x['label']})[1]

    def gen_labeled_fn(self, data_iterator):
        return self.dataset.train_labeled.numpy_augment(lambda: self.session.run(data_iterator))

    def gen_unlabeled_fn(self, data_iterator):
        return self.dataset.train_unlabeled.numpy_augment(lambda: self.session.run(data_iterator))

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt, pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.epoch = 1 + (self.tmp.step // report_nimg)
                    self.train_step(train_session, gen_labeled, gen_unlabeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def eval_checkpoint(self, ckpt=None):

        print("Eval Checkpoint")

        self.epoch = -1


        self.eval_mode(ckpt)
        self.cache_eval()
        ema = self.eval_stats()
        raw = self.eval_stats(classify_op=self.ops.classify_raw)

        print('%16s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % ('', 'acc labeled', 'f1 labeled', 'acc valid', 'f1 valid', 'acc test' ,'f1 test', 'acc unlabeled', 'f1 unlabeled', 'acc evaluation', 'f1 evaluation'))
        print('%16s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % (('raw',) + tuple('%.2f' % x for x in raw)))
        print('%16s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % (('ema',) + tuple('%.2f' % x for x in ema)))

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset, name):
            """Return numpy arrays of all the samples from a dataset."""
            pbar = tqdm(desc='Caching %s examples' % name)
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])
                pbar.update()

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            pbar.close()
            return images, labels

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test.parse(), name='test')
            self.tmp.cache.valid = collect_samples(self.dataset.valid.parse(), name='valid')
            if self.dataset._unlabeled_data is not None:
                self.tmp.cache._unlabeled = collect_samples(self.dataset._unlabeled_data.parse(), name='_unlabeled')

            print("dataset", self.dataset.evaluation_data)
            if self.dataset.evaluation_data is not None:
                self.tmp.cache.evaluation_data = collect_samples(self.dataset.evaluation_data.parse(), name='evaluation')
            self.tmp.cache.train_labeled = collect_samples(self.dataset.train_labeled.take(10000).parse(),
                                                           name='train_labeled')

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        eval_method = "ema" if classify_op is None else "raw"
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []

        print("cache names", self.tmp.cache.keys())
        for subset in ('train_labeled', 'valid', 'test', '_unlabeled','evaluation_data'):
            if subset not in self.tmp.cache:
                accuracies.extend([-1,-1]) # add invalid values
                continue

            images, labels = self.tmp.cache[subset]
            predicted = []
            over_predicted = []
            prob_predicted = []
            embds = []

            for x in range(0, images.shape[0], batch):
                p, p_over, p_all = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })

                if self.ops.embds_op is not None:
                    em = self.session.run(
                        self.ops.embds_op,
                        feed_dict={
                            self.ops.x: images[x:x + batch],
                            **(feed_extra or {})
                        })

                    embds.append(em)

                predicted.append(p)
                over_predicted.append(p_over)
                prob_predicted.append(p_all)

            predicted = np.concatenate(predicted, axis=0)
            over_predicted = np.concatenate(over_predicted, axis=0)
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)

            # save embeddings
            # embds = np.concatenate(embds, axis=0)
            # np.savetxt("/src/"+subset+"_embds.csv", embds)


            self.save_evaluations(self.epoch, subset, eval_method, "normal", predicted, labels)
            self.save_evaluations(self.epoch, subset, eval_method, "over", over_predicted, labels)
            if FLAGS.combined_output != 0:

                prob_predicted = np.concatenate(prob_predicted, axis=0)
                self.save_evaluations(self.epoch, subset, eval_method, "ambiguity", prob_predicted, labels)

            # Combine the overcluster classes if necessary, should have no impact for normal classes if they are not pretty wrong -> happens with class imbalcne
            predicted_classes = predicted.argmax(1)

            # f1 score
            cl_report = metrics.classification_report(labels, predicted_classes, digits=4, output_dict=True)
            accuracies.append(cl_report['macro avg']['f1-score'])

        if verbose:
            self.train_print('kimg %-5d  accuracy&f1 train/valid/test/unlabeled/evaluation  %.2f  %.2f  / %.2f  %.2f /  %.2f  %.2f  /  %.2f  %.2f  /  %.2f  %.2f' %
                             tuple([self.epoch] + accuracies))
        return np.array(accuracies, 'f')

    def save_evaluations(self, epoch, subset, eval_method, suffix,  predicted, labels):
        """
        save predicted results for evaluation
        :param epoch:
        :param subset:
        :param eval_method:
        :param predicted:
        :param labels:
        :param over_clustering:
        :param combined:
        :param combined_over_pred:
        :param combined_fuzzy_prediction:
        :return:
        """

        log_dir = f"/data1/{self.log_time}/ep-{epoch}"

        # load indices
        if subset == "train_labeled":
            path_format = "/data-ssd/fixmatch/%s_train_indices.txt"
            prefix = "train"
        elif subset == "test":
            path_format = "/data-ssd/fixmatch/%s_test_indices.txt"
            prefix = "val"
        elif subset == "_unlabeled":
            prefix = "unlabeled"
            path_format = "/data-ssd/fixmatch/%s_unlabeled_indices.txt"
        elif subset == "evaluation_data":
            prefix = "test"
            path_format = "/data-ssd/fixmatch/%s_evaluation_indices.txt"
        else:
            path_format = None
            prefix = "INVALID"

        if path_format is not None:


            with open(path_format % self.dataset._root_name, 'r') as outfile:
                file_names = json.load(outfile)

            assert len(file_names) == len(labels), "%s -> %d vs. %d" % (subset, len(file_names), len(labels))
            # TODO might not be correct if train samples are more than 10,000 (see caching above)

            os.makedirs(log_dir, exist_ok=True)
            print(f"log {prefix} {eval_method} {suffix}  to {log_dir}")

            save_list(log_dir, combine_file_name(prefix, eval_method, "gt", suffix), np.array(labels))
            save_list(log_dir, combine_file_name(prefix, eval_method,  "files", suffix), np.array(file_names),
                      save_string=True)
            save_list(log_dir, combine_file_name(prefix,  eval_method, "preds", suffix), np.array(predicted),
                      save_float=True)

    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats(classify_op=None, verbose=True):
            return self.eval_stats(feed_extra=feed_extra, classify_op=classify_op, verbose=verbose)

        accuracies = tf.py_func(functools.partial(gen_stats), [], tf.float32)

        tf.summary.scalar('accuracy/train_labeled', accuracies[0])
        tf.summary.scalar('f1/train_labeled', accuracies[1])
        tf.summary.scalar('accuracy/valid', accuracies[2])
        tf.summary.scalar('f1/valid', accuracies[3])
        tf.summary.scalar('accuracy', accuracies[4])
        tf.summary.scalar('f1', accuracies[5])
        if 'classify_raw' in self.ops:
            accuracies = tf.py_func(functools.partial(gen_stats,
                                                      classify_op=self.ops.classify_raw,
                                                      verbose=False), [], tf.float32)
            tf.summary.scalar('accuracy/raw/train_labeled', accuracies[0])
            tf.summary.scalar('f1/raw/train_labeled', accuracies[1])
            tf.summary.scalar('accuracy/raw/valid', accuracies[2])
            tf.summary.scalar('f1/raw/valid', accuracies[3])
            tf.summary.scalar('accuracy/raw', accuracies[4])
            tf.summary.scalar('f1/raw', accuracies[5])


