from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from termcolor import cprint
import math

"""
    Attention-related functions & utilities: Spatial Attention, Channel Attention,
    Self-Attention (Non-Local Network) and Compact Generalized Non-Local Network.
"""

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):

    weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    weight_regularizer = None

    with tf.variable_scope(scope):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def hw_flatten(x):
    batch_size = tf.shape(x)[0]
    return tf.reshape(x, shape=[batch_size, -1, x.shape[-1]])


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e3):
    """
        Adds a bunch of sinusoids of different frequencies to a Tensor.
        Each channel of the input Tensor is incremented by a sinusoid of a difft
        frequency and phase in one of the positional dimensions.
        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.
        The use of relative position is possible because sin(a+b) and cos(a+b) can
        be experessed in terms of b, sin(a) and cos(a).
        x is a Tensor with n "positional" dimensions, e.g. one dimension for a
        sequence or two dimensions for an image
        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels // (n * 2). For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.
            Args:
                x: a Tensor with shape [batch, d1 ... dn, channels]
                min_timescale: a float
                max_timescale: a float
            Returns:
                a Tensor the same shape as x.
    """
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in range(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x


def add_positional_embedding_nd(x, max_length, name):
    """
        Add n-dimensional positional embedding.
        Adds embeddings to represent the positional dimensions of the tensor.
        The input tensor has n positional dimensions - i.e. 1 for text, 2 for images,
        3 for video, etc.
            Args:
                x: a Tensor with shape [batch, p1 ... pn, depth]
                max_length: an integer.  static maximum size of any dimension.
                name: a name for this layer.
            Returns:
                a Tensor the same shape as x.
    """
    static_shape = x.get_shape().as_list()
    dynamic_shape = tf.shape(x)
    num_dims = len(static_shape) - 2
    depth = static_shape[-1]
    base_shape = [1] * (num_dims + 1) + [depth]
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [1] * num_dims + [depth]
    for i in range(num_dims):
        shape = base_shape[:]
        start = base_start[:]
        size = base_size[:]
        shape[i + 1] = max_length
        size[i + 1] = dynamic_shape[i + 1]
        var = (tf.get_variable(
            name + "_%d" % i,
            shape,
            initializer=tf.random_normal_initializer(0, depth ** -0.5)) *
               (depth ** 0.5))
        x += tf.slice(var, start, size)
    return x


def self_attention_block(x, scope, num_heads=8, sn=False):
    """
        Contains the implementation of Self-Attention block.
        As described in "Self-Attention Generative Adversarial Networks" (SAGAN) 
        https://arxiv.org/pdf/1805.08318.pdf.
    """

    ch = x.shape[-1]
    with tf.variable_scope(scope):
        f = conv(x, ch // num_heads, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        g = conv(x, ch // num_heads, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=tf.shape(x))  # [bs, h, w, C]
        x = o + x
        # x = gamma * o + x

    return x


def attach_attention_module(net, attention_module, block_scope=None):
    print("attach attention")
    if attention_module == 'se_block':  # SE_block
        se_block_scope = 'se_block' if block_scope is None else block_scope + '_SE'
        net = se_block(net, se_block_scope)
    elif attention_module == 'cbam_block':  # CBAM_block
        cbam_block_scope = 'cbam_block' if block_scope is None else block_scope + '_CBAM'
        net = cbam_block(net, cbam_block_scope)
    elif attention_module == 'sa_block':
        sa_block_scope = 'sa_block' if block_scope is None else block_scope + 'SA'
        net = self_attention_wrapper(net, sa_block_scope)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, name, ratio=8):
    """
        Contains the implementation of Squeeze-and-Excitation(SE) block.
        As described in https://arxiv.org/abs/1709.01507.
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)
        assert squeeze.get_shape()[1:] == (1, 1, channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel // ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel)
        scale = input_feature * excitation
    return scale


def cbam_block(input_feature, name, ratio=8):
    """
        Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')

    return attention_feature


def self_attention_wrapper(input_feature, name):
    """
        Self-Attention block wrapper
    """

    with tf.variable_scope(name):
        # attention_feature = input_feature
        # attention_feature = add_positional_embedding_nd(input_feature, max_length=1e3, name=name + 'positional')
        attention_feature = add_timing_signal_nd(input_feature)
        # print('positional embemdding')
        attention_feature = spatialCGNLx(attention_feature,
                                         groups=8)
        # attention_feature = self_attention_block(attention_feature, 'sp_at', num_heads=8) #
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keep_dims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat


def group_norm(x, G=32, eps=1e-5, scope='group_norm'):
    with tf.variable_scope(scope):
        shape = tf.shape(x)
        N, H, W = shape[0], shape[1], shape[2]
        C = x.get_shape().as_list()[3]
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x


def spatialCGNLx(input_x, out_channels=None, use_scale=False,
                 groups=None,
                 order=2):
    """
        Spatial CGNL block with Gaussian RBF kernel for image classification.
        As described in "Compact Generalized Non-local Network" https://arxiv.org/abs/1810.13125.
    """

    # input: batchsize, height, width, in_channels
    shape = tf.shape(input_x)
    batchsize, in_channels = shape[0], input_x.get_shape().as_list()[-1]

    if out_channels is None:
        out_channels = in_channels

    # conv g
    g = slim.conv2d(input_x, out_channels, [1, 1], stride=1)

    # conv phi
    phi = slim.conv2d(input_x, out_channels, [1, 1], stride=1)

    # conv theta
    theta = slim.conv2d(input_x, out_channels, [1, 1], stride=1)

    if groups is None:
        groups = 1
    _c = int(in_channels / groups)  # Number of channels C/P
    if groups and groups > 1:
        ts = tf.split(theta, num_or_size_splits=_c, axis=3)
        ps = tf.split(phi, num_or_size_splits=_c, axis=3)
        gs = tf.split(g, num_or_size_splits=_c, axis=3)
        _t_sequences = []
        for i in range(groups):
            _x = kernel(ts[i], ps[i], gs[i],
                        batchsize, _c, shape, order, use_scale)
            _t_sequences.append(_x)

        x = tf.concat(_t_sequences, axis=3)
    else:
        x = kernel(theta, phi, g,
                   batchsize, in_channels, shape, order, use_scale)

    # conv z
    zs = []
    xs = tf.split(x, num_or_size_splits=_c, axis=3)
    for i in range(groups):
        zs.append(slim.conv2d(xs[i],
                              int(out_channels / groups),
                              kernel_size=1, stride=1))
        z = tf.concat(zs, axis=3)

    gn = group_norm(z, groups)
    output = gn + input_x
    if use_scale:
        cprint("=> WARN: SpatialCGNLx block uses 'SCALE'",
               'yellow')
    if groups:
        cprint("=> WARN: SpatialCGNLx block uses '{}' groups".format(groups),
               'yellow')

    cprint('=> WARN: The Taylor expansion order in SpatialCGNLx block is {}'.format(order),
           'yellow')

    return output


def kernel(t, p, g, batchsize, num_channels, shape, num_order, use_scale):
    """
        The non-linear kernel (Gaussian RBF).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
    """

    shape = tf.shape(t)
    t = tf.reshape(t, [batchsize, 1, -1])
    p = tf.reshape(p, [batchsize, 1, -1])
    g = tf.reshape(g, [batchsize, -1, 1])

    # gamma
    gamma = tf.constant([1e-4])

    # NOTE:
    # We want to keep the high-order feature spaces in Taylor expansion to
    # rich the feature representation, so the l2 norm is not used here.
    #

    # beta
    beta = tf.exp(-2 * gamma)

    t_taylor = []
    p_taylor = []
    for order in range(num_order + 1):
        # alpha
        alpha = tf.multiply(
            tf.div(
                tf.pow(
                    (2 * gamma),
                    order),
                math.factorial(order)),
            beta)

        alpha = tf.sqrt(alpha)

        _t = tf.multiply(tf.pow(t, order), alpha)
        _p = tf.multiply(tf.pow(p, order), alpha)

        t_taylor.append(_t)
        p_taylor.append(_p)

    t_taylor = tf.concat(t_taylor, axis=1)
    p_taylor = tf.concat(p_taylor, axis=1)

    att = tf.matmul(p_taylor, g)

    # if use_scale:
    #     att = tf.div(att, (num_channels*h*w)**0.5)

    att = tf.reshape(att, [batchsize, 1, int(num_order + 1)])  # att.view(batchsize, 1, int(num_order+1))
    x = tf.matmul(att, t_taylor)
    x = tf.reshape(x, shape=shape)  # tf.reshape(x, (batchsize, h, w, num_channels))

    return x
