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

import numpy as np
from absl import flags
from sklearn import metrics

from fully_supervised.lib.train import ClassifyFullySupervised
from libml import data
from libml.augment import AugmentPoolCTA
from libml.ctaugment import CTAugment
from libml.train import ClassifySemi
from src.utils.losses import Combine_clusters_by_purity

FLAGS = flags.FLAGS

flags.DEFINE_integer('adepth', 2, 'Augmentation depth.')
flags.DEFINE_float('adecay', 0.99, 'Augmentation decay.')
flags.DEFINE_float('ath', 0.80, 'Augmentation threshold.')


class CTAClassifySemi(ClassifySemi):
    """Semi-supervised classification."""
    AUGMENTER_CLASS = CTAugment
    AUGMENT_POOL_CLASS = AugmentPoolCTA

    @classmethod
    def cta_name(cls):
        return '%s_depth%d_th%.2f_decay%.3f' % (cls.AUGMENTER_CLASS.__name__,
                                                FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        ClassifySemi.__init__(self, train_dir, dataset, nclass, **kwargs)
        self.augmenter = self.AUGMENTER_CLASS(FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def gen_labeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = True
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def gen_unlabeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = False
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()
        v = train_session.run([self.ops.classify_op, self.determine_train_op(), self.ops.update_step],
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0][0] # take normal part from classify op
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        eval_method = "ema" if classify_op is None else "raw"
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        for subset in ('train_labeled', 'valid', 'test', '_unlabeled','evaluation_data'):
            if subset not in self.tmp.cache:
                accuracies.extend([-1, -1])  # add invalid values
                continue

            images, labels = self.tmp.cache[subset]
            predicted = []
            over_predicted = []
            prob_predicted = []

            for x in range(0, images.shape[0], batch):
                p, p_over, p_all = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })

                predicted.append(p)
                over_predicted.append(p_over)
                prob_predicted.append(p_all)

            predicted = np.concatenate(predicted, axis=0)
            over_predicted = np.concatenate(over_predicted, axis=0)
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)

            self.save_evaluations(self.tmp.step >> 10, subset, eval_method, "normal", predicted, labels, over_clustering=False)
            self.save_evaluations(self.tmp.step >> 10, subset, eval_method, "over", over_predicted, labels, over_clustering=True)

            if FLAGS.combined_output != 0:
                prob_predicted = np.concatenate(prob_predicted, axis=0)
                self.save_evaluations(self.epoch, subset, eval_method, "combined", predicted, labels,
                                      over_clustering=True, combined=True, combined_over_pred=over_predicted,
                                      combined_fuzzy_prediction=prob_predicted)

            predicted_classes = predicted.argmax(1)

            # f1 score
            cl_report = metrics.classification_report(labels, predicted_classes, digits=4, output_dict=True)
            accuracies.append(cl_report['macro avg']['f1-score'])

        if verbose:
            self.train_print(
                'kimg %-5d  accuracy&f1 train/valid/test/unlabeled/evaluation  %.2f  %.2f  / %.2f  %.2f /  %.2f  %.2f  /  %.2f  %.2f  /  %.2f  %.2f' %
                tuple([self.tmp.step >> 10] + accuracies))
        self.train_print(self.augmenter.stats())
        return np.array(accuracies, 'f')


class CTAClassifyFullySupervised(ClassifyFullySupervised, CTAClassifySemi):
    """Fully-supervised classification."""

    def train_step(self, train_session, gen_labeled):
        x = gen_labeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)
