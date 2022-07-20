import functools
import itertools
import os
import tensorflow as tf
from absl import flags

from libml import utils
from libml.utils import EasyDict
from src.utils.losses import inverse_ce

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('IDs', [], 'IDs or tags which identify this run')

flags.DEFINE_float('wou', 10, 'Inverse Cross-entropy loss weight for unlabeled examples')
flags.DEFINE_float('wol', 10, 'Inverse Cross-entropy loss weight for labeled examples')
flags.DEFINE_integer('ceinv_labels', 1, 'Inverse Cross-entropy uses labels instead of random selection')
flags.DEFINE_integer('over_alternating', 0, 'Deprecated: Number of epochs to train with overclustering objective, -1 means that the combined loss is taken')
flags.DEFINE_integer('normal_alternating', 5, 'Deprecated: Number of epochs to train with normal objective')
flags.DEFINE_integer('num_heads', 1, 'Deprecated: Number of heads overclustering heads internally used')
flags.DEFINE_integer('combined_output', 4, 'Describes how the output is combined to make a ambiguous image estiamtione. Only 0 for no output prediciton and 4 for the version presented in the paper are supported. Other version are deprecated.')
flags.DEFINE_float('prior_fuzzy_distribution', -1, 'Prior known distribution of fuzzy data in unlabeled data')
flags.DEFINE_float('wf', 0.1, 'Weight for influence of ambiguity prediction impact')
flags.DEFINE_float('wcf', 0, 'Deprecated: Weight for influence of ambiguity prediction impact on certain training data')
flags.DEFINE_float('ws', 0.1, 'Weight for influence of similarity measure between predictions')
flags.DEFINE_float('wc', 1, 'Weight for influence of ce')
flags.DEFINE_integer('prob_gradient', 0, 'Allow the flow of the gradient along the prob certain and prob fuzzy')
flags.DEFINE_integer('use_loss_rescale', 0, 'Rescale loss if multiplied with prob certain or prob fuzzy to reach similar values as before')
flags.DEFINE_integer('use_soft_prob', 1, 'use soft probabilites for certain and fuzziness')

flags.DEFINE_integer('use_weighted_xe',1, 'Use on the labeled training data the weighted version of xe')



def weighted_cross_entropy(labels, logits, dataset_root_name):

    if FLAGS.use_weighted_xe == 1:
        class_weights = tf.constant(utils.calc_class_weights(dataset_root_name))

        # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
        weights = tf.gather(class_weights, labels)

        # compute the loss
        xe = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)
    else:
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    return xe


def over_flags():

    if FLAGS.prior_fuzzy_distribution < 0:
        # use data specific fuzzy distribution
        dataset = FLAGS.dataset

        if "plankton" in dataset:
            FLAGS.prior_fuzzy_distribution = 0.4388
        elif "turkey" in dataset:
            FLAGS.prior_fuzzy_distribution = 0.22
        elif "micebone" in dataset:
            FLAGS.prior_fuzzy_distribution = 0.65
        elif "cifar" in dataset:
            FLAGS.prior_fuzzy_distribution = 0.4
        else:
            raise ValueError("No prior distribution given and not data specific one found")

    if FLAGS.wcf < 0:
        # use same for wcf and wf
        FLAGS.wcf = FLAGS.wf


    return EasyDict(wou=FLAGS.wou, wol=FLAGS.wol, ceinv=FLAGS.ceinv_labels, oa=FLAGS.over_alternating,
                    na=FLAGS.normal_alternating, nh=FLAGS.num_heads, wf=FLAGS.wf, ws=FLAGS.ws, co=FLAGS.combined_output, wc=FLAGS.wc, pf=FLAGS.prior_fuzzy_distribution,
                    pg=FLAGS.prob_gradient, ul=FLAGS.use_loss_rescale, us=FLAGS.use_soft_prob, wcf=FLAGS.wcf)


def get_loss_scales(prob_fuzzy, verbose=False):
    epsilon_ = tf.constant(1e-6)

    assert FLAGS.prob_gradient == 0 or FLAGS.combined_output != 0, "Allow prob gradient only if combined output"
    assert FLAGS.use_soft_prob == 0 or FLAGS.combined_output != 0, "Allow soft prob only if combined output"
    assert FLAGS.use_loss_rescale == 0 or FLAGS.combined_output != 0, "Allow loss scales only if combined output"

    if FLAGS.prob_gradient == 0:
        prob_fuzzy = tf.stop_gradient(prob_fuzzy)

    if FLAGS.use_soft_prob == 0:
        prob_fuzzy = tf.cast(tf.round(prob_fuzzy), dtype=tf.float32)

    prob_certain = 1 - prob_fuzzy

    if FLAGS.use_loss_rescale == 1:
        sum_fuzzy = tf.cast(tf.reduce_sum(prob_fuzzy), dtype=tf.float32)
        total = tf.cast(prob_fuzzy.shape[0], dtype=tf.float32)
        # print(total)
        sum_fuzzy = tf.clip_by_value(sum_fuzzy, 1, total - 1)
        scale_fuzzy = total / sum_fuzzy
        scale_certain = total / (total - sum_fuzzy)

    else:
        scale_certain = tf.constant(1., dtype=tf.float32)
        scale_fuzzy = tf.constant(1., dtype=tf.float32)

    if verbose:
        tf.summary.scalar('monitors/scale_certain', scale_certain)
        tf.summary.scalar('monitors/scale_fuzzy', scale_fuzzy)

    certain_scale = prob_certain * scale_certain
    fuzzy_scale = prob_fuzzy * scale_fuzzy
    # print(certain_scale)

    return certain_scale, fuzzy_scale


def split_normal_over(tensor, nclass, overcluster_k, batch, combined = 0, prob_fuzzy=None, verbose=0):
    """
    over clustering head my consist of multiple heads
    :param tensor: predictions without softmax activation
    :param nclass:
    :param combined: bool to indicate if the output is combined and should be used for fuzzy estimation
    :return:
    """


    if combined != 0:


        if prob_fuzzy is None:


            # combined array
            if combined == 2:
                # calculate softmax activation
                prob_tensor = tf.nn.softmax(tensor)

                # caluclate fuzzy prediction
                prob_fuzzy_tensor = tf.math.reduce_sum(prob_tensor[:, nclass:], axis=1)
            # print(prob_fuzzy_tensor)

            # sigmoid
            if combined == 1 or combined == 3 or combined == 4:
                prob_fuzzy_tensor = tf.nn.sigmoid(tensor[:,nclass + overcluster_k])
        else:
            # use given probability tensor
            prob_fuzzy_tensor = prob_fuzzy

        # classical splitting, just seperating but overclusterng is complete
        if combined == 2:
            tensor_over = tensor
        if combined == 1 or combined == 3 or combined == 4:
            tensor_over = tensor[:, nclass:nclass+overcluster_k]

        tensor_normal = tensor[:, :nclass]

        if verbose > 0:
            classified_fuzzy_tensor = tf.cast(tf.round(prob_fuzzy_tensor), dtype=tf.int32)
            tf.summary.scalar('monitors/num_fuzzy', tf.reduce_sum(classified_fuzzy_tensor))

    else:
        # classical splitting, just seperating
        tensor_over = tensor[:, nclass:]
        tensor_normal = tensor[:, :nclass]
        # fake all are certain
        prob_fuzzy_tensor = tf.zeros(batch)



    return tensor_normal, tensor_over, prob_fuzzy_tensor


def ce(p, q, epsilon_ = tf.constant(1e-6)):
    p = tf.clip_by_value(p, epsilon_, 1 - epsilon_)
    q = tf.clip_by_value(q, epsilon_, 1 - epsilon_)
    return -tf.reduce_sum(p * tf.log(q), axis=1)

def be(p, q, epsilon_ = tf.constant(1e-6)):
    p = tf.clip_by_value(p, epsilon_, 1 - epsilon_)
    q = tf.clip_by_value(q, epsilon_, 1 - epsilon_)
    l = p * tf.math.log(q)
    l += (1 - p) * tf.math.log(1-q)
    return  -l

def get_ops(loss_tensor, post_ops, ema_getter, lr, classifier,
            logits_x, logits_x_over_all,
            logits_u, logits_u_over_all,
            nclass, batch, overcluster_k, xt_in, x_in, y_in, l_in,
            prob_fuzzy, prob_fuzzy_train, logits_u2, logits_u2_over_all,
            uratio=1, pseudo_labels = None, pseudo_mask = None, embedder=None):
    """

    :param loss_tensor:
    :param post_ops:
    :param ema_getter:
    :param lr:
    :param classifier:
    :param logits_x: normal predictions labeled data, without softmax activateion
    :param logits_x_over_all: over cluster predictions labeled data , without softmax activateion, may contain multiple heads
    :param logits_u: normal preddictions unlabeled data , without softmax activateion
    :param logits_u_over_all: over cluster predictions unlabeled data , without softmax activateion,  may contain multiple heads
    :param batch:
    :param overcluster_k:
    :param xt_in:
    :param x_in:
    :param y_in:
    :param l_in:
    :param uratio:
    :param pseudo_labels:
    :param pseudo_mask:
    :param prob_fuzzy: tensor over unlabeled logits which was used to divide certain and fuzzy data, should be based on the same as logits_u
    :param prob_fuzzy: tensor over labeled logits which was used to divide certain and fuzzy data, should be based on the same as logits_x
    :return:
    """
    over_alternating = FLAGS.over_alternating
    ceinv_labels = FLAGS.ceinv_labels
    wol = FLAGS.wol
    wou = FLAGS.wou
    num_heads = FLAGS.num_heads
    combined_output = FLAGS.combined_output != 0
    prior_fuzzy = FLAGS.prior_fuzzy_distribution
    wf = FLAGS.wf
    ws = FLAGS.ws
    wcf = FLAGS.wcf

    certain_scale, fuzzy_scale = get_loss_scales(prob_fuzzy, verbose=True)

    # tf.summary.scalar('monitors/predicted_certain', tf.reduce_mean(prob_certain))

    loss_xeil_all = None
    loss_xeiu_all = None
    for head in range(num_heads):

        # split head
        # print(logits_u_over_all)
        logits_u_over = logits_u_over_all[:, (head * overcluster_k): (head +1) * overcluster_k]
        logits_x_over = logits_x_over_all[:, (head * overcluster_k): (head +1) * overcluster_k]


        if ceinv_labels == 1:
            print("Uses labels for Ceinv")
            if pseudo_labels is None:
                pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_u))
            args_pseudo = tf.argmax(pseudo_labels, axis=1)

            if pseudo_mask is None:
                pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= 0.95)

            # select negative examples based on labels and pseudolabels
            diff_class_element = []
            for i in range(batch):
                r_list = tf.random.shuffle(tf.where(tf.math.not_equal(l_in, l_in[i])))

                is_empty = tf.equal(tf.size(r_list), 0)  # ensure not degenerate case
                r_elemt = tf.cond(is_empty,
                                  lambda: tf.constant(0, dtype=tf.int64),  # use first in batch
                                  lambda: r_list[0])

                diff_class_element.append(r_elemt)
            over_logits_xinv_l = tf.gather(logits_u_over, tf.stack(diff_class_element))

            over_logits_xinv_l = tf.reshape(over_logits_xinv_l, shape=(batch, overcluster_k))
            # unlabeled
            diff_class_element = []
            for i in range(batch * uratio):
                r_list = tf.random.shuffle(tf.where(tf.math.not_equal(args_pseudo, args_pseudo[i])))

                is_empty = tf.equal(tf.size(r_list), 0) # ensure not degenerate case
                r_elemt = tf.cond(is_empty,
                                            lambda: tf.constant(0,dtype=tf.int64), # use first in batch
                                            lambda: r_list[0])

                diff_class_element.append(r_elemt)

            over_logits_xinv_u = tf.gather(logits_u_over, tf.stack(diff_class_element))
            over_logits_xinv_u = tf.reshape(over_logits_xinv_u, shape=(batch * uratio, overcluster_k))
        else:
            print("Uses no labels for Ceinv")
            # shuffle randomly
            # labeleled
            over_logits_xinv_l = tf.gather(logits_x_over, tf.random.shuffle(tf.range(tf.shape(logits_x_over)[0])))
            # unlabeled
            over_logits_xinv_u = tf.gather(logits_u_over, tf.random.shuffle(tf.range(tf.shape(logits_u_over)[0])))


        # 2. calculate loss
        loss_xeil = inverse_ce(tf.nn.softmax(logits_x_over), tf.nn.softmax(over_logits_xinv_l))
        loss_xeil = tf.reduce_mean(loss_xeil)
        tf.summary.scalar('losses/xeil-%d' % head, loss_xeil)
        tf.summary.scalar('losses-scaled/xeil-%d' % head, wol * loss_xeil)

        loss_xeiu = inverse_ce(tf.nn.softmax(logits_u_over), tf.nn.softmax(over_logits_xinv_u))
        if ceinv_labels == 1:
            tf.summary.scalar('monitors/mask_unlabeled', tf.reduce_mean(pseudo_mask))
            loss_xeiu = tf.reduce_mean(loss_xeiu * pseudo_mask * certain_scale)
        else:
            loss_xeiu = tf.reduce_mean(loss_xeiu * certain_scale)
        tf.summary.scalar('losses/xeiu-%d'% head, loss_xeiu)
        tf.summary.scalar('losses-scaled/xeiu-%d'% head, wou *loss_xeiu)

        # add head loss to all
        if loss_xeil_all is None or loss_xeiu_all is None:
            loss_xeil_all = loss_xeil
            loss_xeiu_all = loss_xeiu
        else:
            loss_xeil_all += loss_xeil
            loss_xeiu_all += loss_xeiu

    # divide by head number
    loss_xeil = loss_xeil_all / num_heads
    loss_xeiu = loss_xeiu_all / num_heads

    # calculate loss of fuzziness prediciton
    assert combined_output or wf < 0.0001, "The fuzziness loss makes only sense if the combined output is activated"
    # print(prob_fuzzy)
    assert prob_fuzzy.shape[0] == (batch * uratio), "Wrong size, expected a probability for each fuzzy element %s" % prob_fuzzy.shape
    # print(prob_fuzzy)
    predicted_fuzziness_prior = tf.math.reduce_mean(prob_fuzzy)
    # print(predicted_fuzziness_prior)

    tf.summary.scalar('monitors/predicted_fuzzy', predicted_fuzziness_prior)


    # compute kl divergende-> crossentropy would include entropy of p
    # print(combined_output)
    if FLAGS.combined_output == 1 or FLAGS.combined_output == 2:
        epsilon_ = tf.constant(1e-6)
        output = tf.clip_by_value(predicted_fuzziness_prior, epsilon_, 1. - epsilon_)
        prior_fuzzy = tf.clip_by_value(tf.constant(prior_fuzzy, dtype=output.dtype), epsilon_, 1. - epsilon_)
        # only two elements
        kl = prior_fuzzy * tf.math.log((output / prior_fuzzy))
        kl += (1 - prior_fuzzy) * tf.math.log(((1 - output) / (1-prior_fuzzy)))
        loss_fuzziness =  -kl
    if FLAGS.combined_output == 3:
        mse = (predicted_fuzziness_prior - prior_fuzzy) **2
        # print(mse)
        loss_fuzziness = mse

    if FLAGS.combined_output == 4:
        # use pseudo labels based on the number of fuzzy elements in each batch
        assert prior_fuzzy >= 0 and prior_fuzzy <= 1
        expected_num_fuzzy = min(batch*uratio -1, int(batch * uratio * prior_fuzzy)) # ensure between 0 and batch size -1
        print("Expected num Fuzzy per batch %d" % expected_num_fuzzy)
        cut_off_prob = tf.sort(prob_fuzzy, axis=0)[batch*uratio - expected_num_fuzzy] # should have only one dimension, or 1 as second

        tf.summary.scalar('monitors/cut_off_prob', cut_off_prob)

        pseudo_fuzzy_label = tf.where(prob_fuzzy >= cut_off_prob, tf.ones((batch * uratio)), tf.zeros((batch * uratio)) )

        tf.summary.scalar('monitors/average_pseudo_fuzzy', tf.reduce_mean(pseudo_fuzzy_label))

        loss_fuzziness = be(tf.stop_gradient(pseudo_fuzzy_label), prob_fuzzy)
        loss_fuzziness = tf.reduce_mean(loss_fuzziness)

        loss_certain_fuzziness = be(tf.stop_gradient(tf.zeros((batch))), prob_fuzzy_train)
        loss_certain_fuzziness = tf.reduce_mean(loss_certain_fuzziness)

    if FLAGS.combined_output == 0:
        loss_fuzziness = tf.constant(0., dtype=tf.float32)
        loss_certain_fuzziness = tf.constant(0., dtype=tf.float32)

    tf.summary.scalar('losses/xef', loss_fuzziness)
    tf.summary.scalar('losses/xecf', loss_certain_fuzziness)
    tf.summary.scalar('losses-scaled/xef', wf * loss_fuzziness)
    tf.summary.scalar('losses-scaled/xecf', wcf * loss_certain_fuzziness)

    # calculate loss on fuzzy unlabeled loss if seconds logits is available
    assert logits_u2_over_all is not None or ws < 0.0001, "The logits u2 must be set if ws > 0"

    if logits_u2_over_all is not None:
        epsilon_ = tf.constant(1e-6)
        # use ce to add entropy based on the logits u over
        sim_loss = ce(tf.nn.softmax(logits_u_over_all), tf.nn.softmax(logits_u2_over_all))
        # tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(tf.nn.softmax(logits_u2_over_all)), logits=logits_u_over_all)
        # tf.keras.losses.categorical_crossentropy
        tf.summary.scalar('monitors/similarity_loss', tf.reduce_mean(sim_loss))
        loss_fuzzy_similarity = tf.reduce_mean(sim_loss * fuzzy_scale)
    else:
        loss_fuzzy_similarity = tf.constant(0., dtype=tf.float32)

    tf.summary.scalar('losses/xes', loss_fuzzy_similarity)
    tf.summary.scalar('losses-scaled/xes', ws *loss_fuzzy_similarity)

    if over_alternating < 0:
        # combined
        loss = loss_tensor + wou * loss_xeiu + wol * loss_xeil + wf * loss_fuzziness + wcf * loss_certain_fuzziness + ws * loss_fuzzy_similarity
        tf.summary.scalar('losses/loss', loss)
        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss,
            colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)
        alt_train_op = None
    else:
        # alternating
        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_tensor,
            colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        alt_train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            wou * loss_xeiu + wol * loss_xeil + wf * loss_fuzziness + wcf * loss_certain_fuzziness + ws * loss_fuzzy_similarity,
            colocate_gradients_with_ops=True)
        with tf.control_dependencies([alt_train_op]):
            alt_train_op = tf.group(*post_ops)

    val_raw = classifier(x_in, training=False)
    val_ema = classifier(x_in, getter=ema_getter, training=False)


    raw_normal_out, raw_over_out, raw_prob_out = split_normal_over(val_raw, nclass, overcluster_k, batch * uratio, combined=FLAGS.combined_output)
    ema_normal_out, ema_over_out, ema_prob_out = split_normal_over(val_ema, nclass, overcluster_k, batch * uratio, combined=FLAGS.combined_output)




    return utils.EasyDict(
        xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op, alt_train_op=alt_train_op,
        classify_raw=(tf.nn.softmax(raw_normal_out),
                      tf.nn.softmax(raw_over_out),
                      raw_prob_out),  # No EMA, for debugging.
        classify_op=(tf.nn.softmax(ema_normal_out),
                     tf.nn.softmax(ema_over_out),
                     ema_prob_out),
    embds_op = embedder(x_in, getter=ema_getter, training=False) if embedder is not None else None)