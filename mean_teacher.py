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

"""Mean teachers are better role models:
    Weight-averaged consistency targets improve semi-supervised deep learning results.

Reimplementation of https://arxiv.org/abs/1703.01780
"""
import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import models, utils
from libml.data import PAIR_DATASETS
from libml.overclustering_training import get_ops, split_normal_over, over_flags, get_loss_scales, \
    weighted_cross_entropy
from libml.utils import EasyDict

FLAGS = flags.FLAGS


class MeanTeacher(models.MultiModel):

    def model(self, batch, lr, wd, ema, warmup_pos, consistency_weight, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch, 2] + hwc, 'y')  # unlabeled data
        l_in = tf.placeholder(tf.int32, [batch], 'labels')
        l = tf.one_hot(l_in, self.nclass)

        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)
        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        logits_x = classifier(xt_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        y_1, y_2 = tf.split(y, 2)
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        logits_y = classifier(y_1, training=True, getter=ema_getter)
        logits_teacher = tf.stop_gradient(logits_y)
        logits_student = classifier(y_2, training=True)

        # split
        logits_x, logits_x_over,  prob_fuzzy_train = split_normal_over(logits_x, self.nclass, self.overcluster_k, batch,  combined=FLAGS.combined_output)
        logits_student, logits_student_over, prob_fuzzy = \
            split_normal_over(logits_student, self.nclass, self.overcluster_k, batch, combined=FLAGS.combined_output,
                              verbose=1)
        logits_teacher, logits_teacher_over,  _ = \
            split_normal_over(logits_teacher, self.nclass, self.overcluster_k, batch, combined=FLAGS.combined_output)

        certain_scale, fuzzy_scale = get_loss_scales(prob_fuzzy)

        loss_mt = tf.reduce_mean((tf.nn.softmax(logits_teacher) - tf.nn.softmax(logits_student)) ** 2, -1)
        loss_mt = tf.reduce_mean(loss_mt * certain_scale)

        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = weighted_cross_entropy(labels=l_in, logits=logits_x, dataset_root_name = self.dataset._root_name)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/mt', loss_mt)
        tf.summary.scalar('losses-scaled/mt', warmup * consistency_weight * loss_mt)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)
        tf.summary.scalar('losses-scaled/wd', wd * loss_wd)

        post_ops.append(ema_op)

        # features
        embedder = lambda x, **kw: self.classifier(x, **kw, **kwargs).embeds

        # train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
        #     loss + loss_mt * warmup * consistency_weight + wd * loss_wd, colocate_gradients_with_ops=True)
        # with tf.control_dependencies([train_op]):
        #     train_op = tf.group(*post_ops)

        # return EasyDict(
        #     xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
        #     classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
        #     classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

        return get_ops(loss + loss_mt * warmup * consistency_weight + wd * loss_wd,
                       post_ops=post_ops, ema_getter=ema_getter, lr=lr, classifier=classifier,
                       logits_x=logits_x, logits_x_over_all=logits_x_over,
                       logits_u=logits_student,
                       logits_u_over_all=logits_student_over,
                       nclass=self.nclass, batch=batch, overcluster_k=self.overcluster_k,
                       xt_in=xt_in, x_in=x_in, y_in=y_in, l_in=l_in, prob_fuzzy=prob_fuzzy,prob_fuzzy_train=prob_fuzzy_train,
                       logits_u2=logits_teacher, logits_u2_over_all=logits_teacher_over, embedder=embedder)


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MeanTeacher(
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

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        **over_flags())
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)
    model.eval_checkpoint()


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('consistency_weight', 50., 'Consistency weight.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
