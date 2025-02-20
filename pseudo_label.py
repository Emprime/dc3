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
"""Pseudo-label: The simple and efficient semi-supervised learning method fordeep neural networks.

Reimplementation of http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf
"""

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import utils, data, models
from libml.overclustering_training import get_ops, split_normal_over, get_loss_scales, over_flags, \
    weighted_cross_entropy
from libml.utils import EasyDict

FLAGS = flags.FLAGS


class PseudoLabel(models.MultiModel):

    def model(self, batch, lr, wd, ema, warmup_pos, consistency_weight, threshold, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')
        l = tf.one_hot(l_in, self.nclass)

        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)
        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        logits_x = classifier(xt_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        logits_y = classifier(y_in, training=True)

        # split classes into normal and overclustering
        logits_x, logits_x_over, prob_fuzzy_train = split_normal_over(logits_x, self.nclass, self.overcluster_k, batch,
                                                       combined=FLAGS.combined_output)
        logits_y, logits_y_over, prob_fuzzy = \
            split_normal_over(logits_y, self.nclass, self.overcluster_k, batch, combined=FLAGS.combined_output, verbose=1)

        certain_scale, fuzzy_scale = get_loss_scales(prob_fuzzy)



        # Get the pseudo-label loss
        loss_pl = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(logits_y, axis=-1), logits=logits_y
        )
        # Masks denoting which data points have high-confidence predictions
        greater_than_thresh = tf.reduce_any(
            tf.greater(tf.nn.softmax(logits_y), threshold),
            axis=-1,
            keepdims=True,
        )
        greater_than_thresh = tf.cast(greater_than_thresh, loss_pl.dtype)
        # Only enforce the loss when the model is confident
        loss_pl *= greater_than_thresh * certain_scale
        # Note that we also average over examples without confident outputs;
        # this is consistent with the realistic evaluation codebase
        loss_pl = tf.reduce_mean(loss_pl)

        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = weighted_cross_entropy(labels=l_in, logits=logits_x, dataset_root_name = self.dataset._root_name)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/pl', loss_pl)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        # train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
        #     loss + loss_pl * warmup * consistency_weight + wd * loss_wd, colocate_gradients_with_ops=True)
        # with tf.control_dependencies([train_op]):
        #     train_op = tf.group(*post_ops)
        #
        # return EasyDict(
        #     xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
        #     classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
        #     classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

        return get_ops(loss + loss_pl * warmup * consistency_weight + wd * loss_wd,
                       post_ops=post_ops, ema_getter=ema_getter, lr=lr, classifier=classifier,
                       logits_x=logits_x, logits_x_over_all=logits_x_over,
                       logits_u=logits_y,
                       logits_u_over_all=logits_y_over,
                       nclass=self.nclass, batch=batch, overcluster_k=self.overcluster_k,
                       xt_in=xt_in, x_in=x_in, y_in=y_in, l_in=l_in, prob_fuzzy=prob_fuzzy, prob_fuzzy_train=prob_fuzzy_train,
                       logits_u2=logits_y, logits_u2_over_all=logits_y_over)


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = PseudoLabel(
        FLAGS.train_dir,
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        warmup_pos=FLAGS.warmup_pos,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        smoothing=FLAGS.smoothing,
        consistency_weight=FLAGS.consistency_weight,
        threshold=FLAGS.threshold,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        **over_flags())
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)
    model.eval_checkpoint()


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('consistency_weight', 1., 'Consistency weight.')
    flags.DEFINE_float('threshold', 0.95, 'Pseudo-label threshold.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.1, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
