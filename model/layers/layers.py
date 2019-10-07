from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell

from .stn import spatial_transformer_network as stn 
from .sparse import sparse_conv


def feat_norm(input, dimZ):
    """

    """
    beta = tf.get_variable('beta', shape=(dimZ,), initializer=tf.constant_initializer(value=0.0))
    gamma = tf.get_variable('gamma', shape=(dimZ,), initializer=tf.constant_initializer(value=1.0))
    output, _, _ = tf.nn.fused_batch_norm(input, gamma, beta)

    return output

def conv2d_bn_lrn_drop(name_scope,
                       input_tensor,
                       kernel_size,
                       pooling=None, 
                       strides=[1, 1, 1, 1],
                       pool_strides=[1, 1, 1, 1],
                       activation=tf.nn.relu,
                       use_sparse_conv=False,
                       use_bn=False,
                       use_mvn=False,
                       is_training=True,
                       use_lrn=False,
                       keep_prob=1.0,
                       dropout_maps=False,
                       init_opt=0,
                       bias_init=0.1):
    """ Adds a 2-D convolutional layer given 4-D `inputs` and `kernel` with optional BatchNorm, LocalResponseNorm and Dropout.
        Args:
            scope_or_name: `string` or `VariableScope`, the scope to open.
            inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
            kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
            bias: `1-D Tensor`, [out_channels] bias.
            strides: list of `ints`, length 4, the stride of the sliding window for each dimension of `inputs`.
            activation: activation function to be used (default: `tf.nn.relu`).
            use_bn: `bool`, whether or not to include batch normalization in the layer.
            is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
            use_lrn: `bool`, whether or not to include local response normalization in the layer.
            keep_prob: `double`, dropout keep prob.
            dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
            padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.
        Returns:
            `4-D Tensor`, has the same type `inputs`.
    """
    print("Kernel size", kernel_size)
    with tf.variable_scope(name_scope):
        if init_opt == 0:
            stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]))
        
        elif init_opt == 1:
            init_opt == 5e-2 
        
        elif init_opt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2])), 5e-2)

        if use_sparse_conv:
            input_tensor, binary_mask = input_tensor
            output_tensor, binary_mask = sparse_conv(input_tensor, binary_mask, filters=kernel_size[3],
                                                    kernel_size=kernel_size[0], strides=strides[0])
        else:   
            kernel = tf.get_variable('weights', kernel_size,
                                    initializer=tf.random_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_tensor, kernel, strides, padding='SAME', name='conv')

            bias = tf.get_variable('bias', kernel_size[3],
                                    initializer=tf.constant_initializer(value=bias_init))
            output_tensor = tf.nn.bias_add(conv, bias, name='pre_activation')

        if use_bn:
            output_tensor = batch_norm(output_tensor, is_training=is_training, scale=True, fused=True, scope='batch_norm')
        
        if use_mvn:
            output_tensor = feat_norm(output_tensor, kernel_size[3])
        
        if activation:
            output_tensor = activation(output_tensor, name='activation')
        
        if use_lrn:
            output_tensor = tf.nn.local_response_normalization(output_tensor, name='local_responsive_norm')
        
        if dropout_maps:
            conv_shape = tf.shape(output_tensor)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            output_tensor = tf.nn.dropout(output_tensor, keep_prob, noise_shape=n_shape)
        else:
            output_tensor = tf.nn.dropout(output_tensor, keep_prob)
        
        if pooling:
            output_tensor = tf.nn.max_pool(output_tensor, ksize=pooling, strides=pool_strides, padding='VALID')

        if use_sparse_conv:
            return (output_tensor, binary_mask)
        else:
            return output_tensor

def rnn_layers(name_scope, 
               input_tensor, 
               num_hidden=256, 
               num_layer=1,
               keep_prob=1.0):
        """
            Create Bi-LSTM layers

            Args:
                :param input_tensor: input tensor (type: tensor)
                :param num_hidden  : number of hidden cells (type: int)
                :param num_layer  : number of hidden layers (type: int)
            :return 
        """

        with tf.variable_scope(name_scope):
            # forward direction cells 
            fw_cells = [LSTMCell(num_units=num_hidden, forget_bias=1.0, state_is_tuple=True) for _ in range(num_layer)]

            # backward direction cells 
            bw_cells = [LSTMCell(num_units=num_hidden, forget_bias=1.0, state_is_tuple=True) for _ in range(num_layer)]

            # bidirectional RNN 
            lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, input_tensor, dtype=tf.float32)

            # Dropout layer 
            if keep_prob:
                lstm_net = tf.nn.dropout(lstm_net, keep_prob=keep_prob)
            
            return lstm_net
            
def spatial_transformer_layer(name_scope,
                              input_tensor,
                              img_size, 
                              kernel_size,
                              pooling=None, 
                              strides=[1, 1, 1, 1],
                              pool_strides=[1, 1, 1, 1],
                              activation=tf.nn.relu,
                              use_bn=False,
                              use_mvn=False,
                              is_training=False,
                              use_lrn=False,
                              keep_prob=1.0,
                              dropout_maps=False,
                              init_opt=0,
                              bias_init=0.1):
    """
        Define spatial transformer network layer 
        Args:
        scope_or_name: `string` or `VariableScope`, the scope to open.
        inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
        kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
        img_size: 2D array, [image_width. image_height]
        bias: `1-D Tensor`, [out_channels] bias.
        strides: list of `ints`, length 4, the stride of the sliding window for each dimension of `inputs`.
        activation: activation function to be used (default: `tf.nn.relu`).
        use_bn: `bool`, whether or not to include batch normalization in the layer.
        is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
        use_lrn: `bool`, whether or not to include local response normalization in the layer.
        keep_prob: `double`, dropout keep prob.
        dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.
    Returns:
        `4-D Tensor`, has the same type `inputs`.
    """
    
    img_height = img_size[0]
    img_width = img_size[1]

    with tf.variable_scope(name_scope):
        if init_opt == 0:
            stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]))
        
        elif init_opt == 1:
            stddev = 5e-2 

        elif init_opt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2])), 5e-2)

        kernel = tf.get_variable('weights', kernel_size,
                    initializer=tf.random_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_tensor, kernel, strides, padding='SAME', name='conv')

        bias = tf.get_variable('bias', kernel_size[3],
                    initializer=tf.constant_initializer(value=bias_init))

        output_tensor = tf.nn.bias_add(conv, bias, name='pre_activation')

        if use_bn:
            output_tensor = batch_norm(output_tensor, is_training=is_training, scale=True, fused=True, scope='batch_norm')

        if use_mvn:
            output_tensor = feat_norm(output_tensor, kernel_size[3])

        if activation:
            output_tensor = activation(output_tensor, name='activation')
        
        if use_lrn:
            output_tensor = tf.nn.local_response_normalization(output_tensor, name='local_responsive_normalization')

        if dropout_maps:
            conv_shape = tf.shape(output_tensor)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            output_tensor = tf.nn.dropout(output_tensor, keep_prob=keep_prob, noise_shape=n_shape)
        else:
            output_tensor = tf.nn.dropout(output_tensor, keep_prob=keep_prob)
        
        if pooling:
            output_tensor = tf.nn.max_pool(output_tensor, ksize=pooling, strides=pool_strides, padding='VALID')

        output_tensor = tf.contrib.layers.flatten(output_tensor)

        output_tensor = tf.contrib.layers.fully_connected(output_tensor, 64, scope='fully_connected_layer_1')
        output_tensor = tf.nn.tanh(output_tensor)

        output_tensor = tf.contrib.layers.fully_connected(output_tensor, 6, scope='fully_connected_layer_2')
        output_tensor = tf.nn.tanh(output_tensor)

        stn_output = stn(input_fmap=input_tensor, theta=output_tensor, out_dims=(img_height, img_width))

        return stn_output

def dilated_conv2d_bn_lrn_drop(name_scope,
                               input_tensor,
                               kernel_size,
                               dilated_rate,
                               activation=tf.nn.relu,
                               use_bn=False,
                               use_mvn=False,
                               is_training=True,
                               use_lrn=False,
                               keep_prob=1.0,
                               dropout_maps=False,
                               init_opt=0,
                               bias_init=0.1):
    """ Adds a 2-D convolutional layer given 4-D `inputs` and `kernel` with optional BatchNorm, LocalResponseNorm and Dropout.
        Args:
            scope_or_name: `string` or `VariableScope`, the scope to open.
            inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
            kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
            dilated_rate: `int`, dilated factor 
            strides: list of `ints`, length 4, the stride of the sliding window for each dimension of `inputs`.
            activation: activation function to be used (default: `tf.nn.relu`).
            use_bn: `bool`, whether or not to include batch normalization in the layer.
            is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
            use_lrn: `bool`, whether or not to include local response normalization in the layer.
            keep_prob: `double`, dropout keep prob.
            dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
            padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.
        Returns:
            `4-D Tensor`, has the same type `inputs`.
    """
    with tf.variable_scope(name_scope):
        if init_opt == 0:
            stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]))

        elif init_opt == 1:
            stddev = 5e-2 
        
        elif init_opt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2])), 5e-2)

        kernel = tf.get_variable('weights', kernel_size, 
                    initializer=tf.random_normal_initializer(stddev=stddev))

        dilated_conv = tf.nn.atrous_conv2d(input_tensor, kernel, rate=dilated_rate, 
                    padding='SAME', name='dilated_conv')

        bias = tf.get_variable('bias', kernel_size[3], initializer=tf.constant_initializer(value=bias_init))

        output_tensor = tf.nn.bias_add(dilated_conv, bias, name='pre_activation')

        if use_bn:
            output_tensor = batch_norm(output_tensor, is_training=is_training, scale=True, fused=True, scope='batch_norm')

        if use_mvn:
            output_tensor = feat_norm(output_tensor, kernel_size[3])

        if activation:
            output_tensor = activation(output_tensor, name='activation')

        if use_lrn:
            output_tensor = tf.nn.local_response_normalization(output_tensor, name='local_response_normalization')

        if dropout_maps:
            conv_shape = tf.shape(output_tensor)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            output_tensor = tf.nn.dropout(output_tensor, keep_prob, noise_shape=n_shape)
        else:
            output_tensor = tf.nn.dropout(output_tensor, keep_prob)

        return output_tensor

def transposed_conv2d_bn_lrn_drop(name_scope, 
                                input_tensor,
                                kernel_size,
                                output_size,
                                stride=2,
                                activation=tf.nn.relu,
                                use_bn=False,
                                use_mvn=False,
                                is_training=True,
                                use_lrn=False,
                                keep_prob=1.0,
                                dropout_maps=False,
                                init_opt=0,
                                bias_init=0.1):
    """ Adds a 2-D convolutional layer given 4-D `inputs` and `kernel` with optional BatchNorm, LocalResponseNorm and Dropout.
        Args:
            scope_or_name: `string` or `VariableScope`, the scope to open.
            inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
            kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
            dilated_rate: `int`, dilated factor 
            stride: ints, length 1, the stride of the sliding window for each dimension of `inputs`.
            activation: activation function to be used (default: `tf.nn.relu`).
            use_bn: `bool`, whether or not to include batch normalization in the layer.
            is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
            use_lrn: `bool`, whether or not to include local response normalization in the layer.
            keep_prob: `double`, dropout keep prob.
            dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
            padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.
        Returns:
            `4-D Tensor`, has the same type `inputs`.
    """
    with tf.variable_scope(name_scope):
        if init_opt == 0:
            stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]))
        
        elif init_opt == 1:
            stddev = 5e-2
        
        elif init_opt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2])), 5e-2)

        kernel = tf.get_variable('weights', kernel_size, 
                    initializer=tf.random_normal_initializer(stddev=stddev))
        
        bias = tf.get_variable('bias', kernel_size[2],
                    initializer=tf.random_normal_initializer(stddev=stddev))
        
        transposed_conv = tf.nn.conv2d_transpose(input_tensor, kernel, output_size, 
                    strides=[1, stride, stride, 1], padding='SAME', name='transposed_conv2d')

        output_tensor = tf.nn.bias_add(transposed_conv, bias, name='pre_activation')

        if use_bn:
            output_tensor = batch_norm(output_tensor, is_training=is_training, scale=True, fused=True, scope='batch_normalization')
        
        if use_mvn:
            output_tensor = feat_norm(output_tensor, kernel_size[3])

        if activation:
            output_tensor = activation(output_tensor, name='activation')

        if use_lrn:
            output_tensor = tf.nn.local_response_normalization(output_tensor, name='local_response_localization')

        if dropout_maps:
            conv_shape = tf.shape(output_tensor)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            output_tensor = tf.nn.dropout(output_tensor, keep_prob, noise_shape=n_shape)
        else:
            output_tensor = tf.nn.dropout(output_tensor, keep_prob)

        return output_tensor


def downsample_avg(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    """

    """
    return tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding='SAME', name='down_avg')

def upsample_simple(input_tensor, output_size, up_rate, num_classes):
    """

    """
    filter_up = tf.constant(1.0, shape=[up_rate, up_rate, num_classes, num_classes])
    return tf.nn.conv2d_transpose(input_tensor, filter_up,
                                  output_shape=output_size,
                                  strides=[1, up_rate, up_rate, 1])

def horizontal_cell(images, num_filters_out, cell_fw, cell_bw, keep_prob=1.0, scope=None):
    """
        Run an LSTM bidirectionally over all the rows of each image.
            Args:
                images: (num_images, height, width, depth) tensor
                num_filters_out: output depth
                scope: optional scope name
            Returns:
                (num_images, height, width, num_filters_out) tensor, where
    """
    with tf.variable_scope(scope, "HorizontalGru", [images]):
        sequence = images_to_sequence(images)

        shapeT = tf.shape(sequence)
        sequence_length = shapeT[0]
        batch_sizeRNN = shapeT[1]
        sequence_lengths = tf.to_int64(
            tf.fill([batch_sizeRNN], sequence_length))
        forward_drop1 = DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        backward_drop1 = DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
        rnn_out1, _ = tf.nn.bidirectional_dynamic_rnn(forward_drop1, backward_drop1, sequence, dtype=tf.float32,
                                                      sequence_length=sequence_lengths, time_major=True,
                                                      swap_memory=True, scope=scope)
        rnn_out1 = tf.concat(rnn_out1, 2)
        rnn_out1 = tf.reshape(rnn_out1, shape=[-1, batch_sizeRNN, 2, num_filters_out])
        output_sequence = tf.reduce_sum(rnn_out1, axis=2)
        batch_size = tf.shape(images)[0]
        output = sequence_to_images(output_sequence, batch_size)
        return output

def images_to_sequence(tensor):
    """
        Convert a batch of images into a batch of sequences.
            Args:
                tensor: a (num_images, height, width, depth) tensor
            Returns:
                (width, num_images*height, depth) sequence tensor
    """
    transposed = tf.transpose(tensor, [2, 0, 1, 3])

    shapeT = tf.shape(transposed)
    shapeL = transposed.get_shape().as_list()
    # Calculate the ouput size of the upsampled tensor
    n_shape = tf.stack([
        shapeT[0],
        shapeT[1] * shapeT[2],
        shapeL[3]
    ])
    reshaped = tf.reshape(transposed, n_shape)
    return reshaped


def sequence_to_images(tensor, num_batches):
    """
    Convert a batch of sequences into a batch of images.
        Args:
            tensor: (num_steps, num_batchesRNN, depth) sequence tensor
            num_batches: the number of image batches
        Returns:
            (num_batches, height, width, depth) tensor
    """

    shapeT = tf.shape(tensor)
    shapeL = tensor.get_shape().as_list()
    # Calculate the ouput size of the upsampled tensor
    height = tf.to_int32(shapeT[1] / num_batches)
    n_shape = tf.stack([
        shapeT[0],
        num_batches,
        height,
        shapeL[2]
    ])

    reshaped = tf.reshape(tensor, n_shape)
    return tf.transpose(reshaped, [1, 2, 0, 3])

def separable_rnn(name_scope,
                  input_tensor, 
                  num_filters, 
                  keep_prob=1.0,
                  cell_type='LSTM'):
    """
    Run bidirectional LSTMs first horizontally then vertically.
        Args:
            images: (num_images, height, width, depth) tensor
            num_filters_out: output layer depth
            nhidden: hidden layer depth
            scope: optional scope name
        Returns:
        (num_images, height, width, num_filters_out) tensor
    """
    with tf.variable_scope(name_scope, 'SeparableLSTM', [input_tensor]):
        with tf.variable_scope('horizontal'):
            if 'LSTM' in cell_type:
                cell_fw = LSTMCell(num_filters, use_peepholes=True, state_is_tuple=True)
                cell_bw = LSTMCell(num_filters, use_peepholes=True, state_is_tuple=True)

            if 'GRU' in cell_type:
                cell_fw = GRUCell(num_filters)
                cell_bw = GRUCell(num_filters)

            hidden_cell = horizontal_cell(input_tensor, num_filters, cell_fw, cell_bw, keep_prob=keep_prob, scope=name_scope)

        with tf.variable_scope('vertical'):
            transposed = tf.transpose(hidden_cell, [0, 2, 1, 3])
            if 'LSTM' in cell_type:
                cell_fw = LSTMCell(num_filters, use_peepholes=True, state_is_tuple=True)
                cell_bw = LSTMCell(num_filters, use_peepholes=True, state_is_tuple=True)
            
            if 'GRU' in cell_type:
                cell_fw = GRUCell(num_filters)
                cell_bw = GRUCell(num_filters)

            output_transposed = horizontal_cell(transposed, num_filters, cell_fw, cell_bw, keep_prob=keep_prob, scope=name_scope)

        output_tensor = tf.transpose(output_transposed, [0, 2, 1, 3])

        return output_tensor

def up_sample_resnet(inp, channel_in, channel_out, out_shape, filter_size, pool_size, res_depth, activation):
    deconv = transposed_conv2d_bn_lrn_drop('deconv', inp, [filter_size, filter_size, channel_in, channel_in * 2],
                                         out_shape, pool_size, activation=activation)
    # print(pool_size)
    # x = conv2d_bn_lrn_drop('conv1', deconv, [filter_size, filter_size, pool_size * channel_in,
    #                                          channel_in], activation=tf.identity)
    x = deconv
    orig_x = x
    # x = tf.nn.relu(x, name='activation')
    for aRes in range(0, res_depth):
        if aRes < res_depth - 1:
            x = conv2d_bn_lrn_drop('convR_{}'.format(aRes), x, [filter_size, filter_size, channel_in,
                                                                 channel_in], activation=activation)
        else:
            x = conv2d_bn_lrn_drop('convR_{}'.format(aRes), x, [filter_size, filter_size, channel_in,
                                                                 channel_in], activation=tf.identity)
    x += orig_x
    x = activation(x, name='activation')

    x = conv2d_bn_lrn_drop('class_aux', x, [4, 4, channel_in, channel_out], activation=tf.identity)

    return x

def down_sample_resnet(x, channel_in, channel_out, filter_size, res_depth, pool_size, activation):
    # x = conv2d_bn_lrn_drop('conv1', inp, [filter_size, filter_size, channel_in // 2,
    #                                            channel_in], activation=tf.identity)
    orig_x = x
    # x = tf.nn.relu(x, name='activation')
    print("Down sample resnet")
    for aRes in range(0, res_depth):
        if aRes < res_depth - 1:
            x = conv2d_bn_lrn_drop('convR_{}'.format(aRes), x, [filter_size, filter_size, channel_in,
                                                                 channel_in], activation=activation)
        else:
            x = conv2d_bn_lrn_drop('convR_{}'.format(aRes), x, [filter_size, filter_size, channel_in,
                                                                 channel_in], activation=tf.identity)
    x += orig_x
    x = activation(x, name='activation')
    x = tf.nn.max_pool(x, pool_size, pool_size, padding='SAME', name='pool')

    x = conv2d_bn_lrn_drop('class_aux', x, [4, 4, channel_in, channel_out], activation=tf.nn.relu)

    return x
