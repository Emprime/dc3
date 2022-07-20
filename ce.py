import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import utils, data, models
from libml.data import PAIR_DATASETS
from libml.overclustering_training import get_ops, split_normal_over, over_flags, get_loss_scales, \
    weighted_cross_entropy
from libml.utils import EasyDict

FLAGS = flags.FLAGS


class CE(models.MultiModel):

    def model(self, batch, lr, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch, 2] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')
        l = tf.one_hot(l_in, self.nclass)

        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        logits_x = classifier(xt_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        y_1, y_2 = tf.split(y, 2)
        logits_y1 = classifier(y_1, training=True)
        logits_y1 = tf.stop_gradient(logits_y1)
        logits_y2 = classifier(y_2, training=True)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        # split
        logits_x, logits_x_over,  prob_fuzzy_train = split_normal_over(logits_x, self.nclass,self.overcluster_k, batch,  combined=FLAGS.combined_output)
        logits_y1, logits_y1_over,  _ = \
            split_normal_over(logits_y1, self.nclass,self.overcluster_k, batch, combined=FLAGS.combined_output)
        logits_y2, logits_y2_over,  prob_fuzzy = \
            split_normal_over(logits_y2, self.nclass,self.overcluster_k, batch, combined=FLAGS.combined_output,  verbose=1)
        certain_scale, fuzzy_scale = get_loss_scales(prob_fuzzy)

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
        loss = weighted_cross_entropy(labels=l_in, logits=logits_x, dataset_root_name = self.dataset._root_name)

        # print("loss", loss)
        loss = tf.reduce_mean(loss * certain_scale) * FLAGS.wc

        # print("loss", loss)
        tf.summary.scalar('losses/xe', loss)

        return get_ops(loss,
                       post_ops=post_ops, ema_getter=ema_getter, lr=lr, classifier=classifier,
                       logits_x=logits_x, logits_x_over_all=logits_x_over, logits_u=logits_y2,
                       logits_u_over_all=logits_y2_over,
                       nclass=self.nclass, batch=batch, overcluster_k=self.overcluster_k,
                       xt_in=xt_in, x_in=x_in, y_in=y_in, l_in=l_in, prob_fuzzy=prob_fuzzy,
                       logits_u2=logits_y1, logits_u2_over_all=logits_y1_over,prob_fuzzy_train=prob_fuzzy_train)


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = CE(
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
