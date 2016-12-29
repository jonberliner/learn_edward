import tensorflow as tf
from edward.models import Normal
import numpy as np
rng = np.random
import pdb
from copy import deepcopy
from util import sample_gumbel

def bnnet(sizes, act_fn=tf.nn.relu):
    n_hid = len(sizes) - 2
    W = []
    b = []
    qW = []
    qb = []
    kl_dict = {}
    prev_size = sizes[0]
    for size in sizes[1:]:
        W0 = Normal(mu=tf.zeros([prev_size, size]), 
                    sigma=tf.ones([prev_size, size]))
        b0 = Normal(mu=tf.zeros([size]),
                    sigma=tf.ones([size]))

        qW0 = Normal(mu=tf.Variable(tf.random_normal(tf.shape(W0))), 
                     sigma=tf.nn.softplus(tf.Variable(tf.random_normal(tf.shape(W0)))))
        qb0 = Normal(mu=tf.Variable(tf.random_normal(tf.shape(b0))),
                     sigma=tf.nn.softplus(tf.Variable(tf.random_normal(tf.shape(b0)))))

        W.append(W0)
        b.append(b0)
        qW.append(qW0)
        qb.append(qb0)

        kl_dict[W0] = qW0
        kl_dict[b0] = qb0
        prev_size = size

    def call(x):
        out = x
        li = 0
        for W0, b0 in zip(*[W, b]):
            out = tf.matmul(out, W0) + b0
            if li < n_hid:
                out = act_fn(out)
            li += 1

        def sample():
            li = 0
            out = x
            for qW0, qb0 in zip(*[qW, qb]):
                out = tf.matmul(out, qW0.sample()) + qb0.sample()
                if li < n_hid:
                    out = act_fn(out)
                li += 1
            return out

        setattr(out, 'sample', sample)
        setattr(out, 'call', call)
        return out

    setattr(call, 'act_fn', act_fn)
    setattr(call, 'W', W)
    setattr(call, 'b', b)
    setattr(call, 'qW', qW)
    setattr(call, 'qb', qb)
    setattr(call, 'kl_dict', kl_dict)
    setattr(call, 'n_hid', n_hid)
    return call


if __name__ == '__main__':
    import edward as ed
    from matplotlib import pyplot as plt
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


    # # MNIST
    # N_IN = 784
    # N_OUT = 10

    BS = 50
    N_IN = 1
    N_OUT = 1

    # CONTINUOUS HIDDEN
    HID = [2]
    act_fn0 = tf.nn.tanh

    # # DISCRETE HIDDEN
    # HID = [10, 10]
    # TEMP = 0.1
    # act_fn0 = lambda logits: gumbel_softmax(logits, TEMP)

    # build bayesian net

    net1 = bnnet([N_IN] + HID + [N_OUT], act_fn0)

    # placeholders
    input1 = tf.placeholder(tf.float32, (None, N_IN))

    # hook in inputs
    output0 = net1(input1)

    # stochastic output for training
    output1 = Normal(mu=output0, sigma=0.1 * tf.ones([BS, N_OUT]))
    # # MNIST
    # output1 = gumbel_softmax(output0, TEMP)
    # y_hat = gumbel_softmax(output0.sample(), TEMP)

    # build toy dataset
    x_train = np.linspace(-3, 3, BS)[:, None]
    y_train = np.cos(x_train) + rng.randn(*x_train.shape) * 0.1
    # # MNIST
    # x_train = mnist.train.images
    # y_train = mnist.train.labels

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # pre-training
        fd = {input1: x_train}
        hat0 = deepcopy(output0.sample().eval(feed_dict=fd))

        # inference
        fd = {input1: x_train, 
              output1: y_train}
        inference = ed.KLqp(output0.call.kl_dict, fd)
        inference.run(n_iter=1000, n_samples=20)
        # # MNIST
        # fd = {input1: x_train}
        # hat1 = y_hat.eval(feed_dict=fd)
        # hh = np.array([hh0.argmax() for hh0 in hat1])
        # tt = np.array([tt0.argmax() for tt0 in y_train])
        # percent = float((hh == tt).sum()) / len(tt)
        # print 'acc: %03f' % (percent)

        # post-training
        fd = {input1: x_train}
        hat1 = output0.sample().eval(feed_dict=fd)

        xr = x_train.ravel()
        plt.plot(xr, y_train.ravel(), 'k')
        plt.plot(xr, hat0.ravel(), 'r')
        plt.plot(xr, hat1.ravel(), 'g')
        plt.show()
