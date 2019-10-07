#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" This work is an tensorflow (v1.12.0) implements Couple Unet,
a new connectivity pattern for the U-Net architecture. 
Given several stacked U-Nets, coupling each U-Net pair through 
the connections of their semantic blocks, resulting in the coupled 
U-Nets (CU-Net). The coupling connections could make the information 
flow more efficiently across U-Nets. The feature reuse across
U-Nets makes each U-Net very parameter efficient.
Ref: https://arxiv.org/abs/1808.06521
"""

import numpy as np 
import tensorflow as tf 
from collections import OrderedDict
import tensorflow.contrib.slim as slim

from .layers import layers 
from .layers.cspn import cspn
from .layers.attention import attach_attention_module

__author__ = "Kristopher"
__email__ = "kristopher@cinnamon.is"
__status__ = "Module"


def unet_block(input_tensor, use_residual, use_lstm, use_spn, 
               use_attention, in_channel, scale_space_num, res_depth, 
               feature_root, filter_size, pool_size, activation, last_block,
               prev_downsampling=None, prev_upsampling=None, binary_mask=None):
    """
        Define an U-Net block which consists of an encoder (down-sampling layers)
        and a decoder (up-sampling layers)
        :param input_tensor: input image
        :param use_residual: use residual connection (ResNet)
        :param use_lstm: run a separable LSTM horizontally then vertically across input features
        :param use_spn: use Spatial Propagation Network
        :param in_channel: number of input channels
        :param scale_space_num: number of down-sampling / up-sampling blocks
        :param res_depth: number of convolution layers in a down-sampling block
        :param feature_root: number of features in the first layers
        :param filter_size: convolution kernel size
        :param pool_size: pooling size
        :param activation: activation function
        :param prev_downsampling: previous down-sampling tower's outputs (used for coupling connection)
        :param prev_upsampling: previous up-sampling tower's outputs (used for coupling connection)
        :return:
    """
    ksize_pooling = [1, pool_size, pool_size, 1]
    stride_pooling = [1, pool_size, pool_size, 1]

    last_feature_num = in_channel
    act_feature_num = feature_root
    downsampling_conv = OrderedDict()
    upsampling_conv = OrderedDict()

    if binary_mask is not None:
        input_tensor = (input_tensor, binary_mask)
        use_sparse_conv = True 
    else:
        use_sparse_conv = False

    # Downsampling block
    print("Downsampling block")
    for layer in range(0, scale_space_num):
        with tf.variable_scope('unet_downsampling_{}'.format(layer)) as scope:
            if use_residual:
                x = layers.conv2d_bn_lrn_drop('conv_1', input_tensor, 
                    [filter_size, filter_size, last_feature_num, act_feature_num],
                    activation=tf.identity, use_sparse_conv=use_sparse_conv)

                if use_sparse_conv:
                    x, binary_mask = x
                
                original_x = x
                x = tf.nn.relu(x, name='activation')

                if use_sparse_conv:
                    x = (x, binary_mask)

                for depth in range(0, res_depth):
                    if depth < res_depth - 1:
                        x = layers.conv2d_bn_lrn_drop('conv_res_{}'.format(depth), x,
                            [filter_size, filter_size, act_feature_num, act_feature_num],
                            activation=activation, use_sparse_conv=use_sparse_conv)
                    else:
                        x = layers.conv2d_bn_lrn_drop('conv_res_{}'.format(depth), x, 
                            [filter_size, filter_size, act_feature_num, act_feature_num], 
                            activation=tf.identity, use_sparse_conv=use_sparse_conv)

                if use_sparse_conv:
                    x, binary_mask = x

                x += original_x
                x = activation(x, name='activation')

                if prev_downsampling is not None:
                    prev_downsampling_layer = prev_downsampling[layer]
                    x = tf.concat([prev_downsampling_layer, x], axis=3, name='concat')
                    x = layers.conv2d_bn_lrn_drop('conv1_1', x, 
                        [1, 1, 2*act_feature_num, act_feature_num], activation=activation)
                    
                if use_attention and layer > scale_space_num - 2:
                    downsampling_conv[layer] = attach_attention_module(x, 'sa_block')
                else:
                    downsampling_conv[layer] = x
                downsampling_conv[layer] = x
            else:
                conv1 = layers.conv2d_bn_lrn_drop('conv_1', input_tensor, 
                        [filter_size, filter_size, last_feature_num, act_feature_num], activation=activation)
                
                downsampling_conv[layer] = layers.conv2d_bn_lrn_drop('conv_2', conv1, 
                        [filter_size, filter_size, act_feature_num, act_feature_num], activation=activation)

            if layer < scale_space_num - 1:
                input_tensor = tf.nn.max_pool(x, ksize_pooling, stride_pooling, padding='SAME', name='pooling')
            else:
                input_tensor = x

            if use_sparse_conv:
                input_tensor = (input_tensor, binary_mask)
            
            last_feature_num = act_feature_num
            act_feature_num *= pool_size

    act_feature_num = int(last_feature_num/pool_size)

    if use_sparse_conv:
        input_tensor = input_tensor[0]
    
    # Bottleneck block
    print("Bottleneck block")

    if use_lstm:
        input_tensor = layers.separable_rnn('rnn', input_tensor, last_feature_num, cell_type='LSTM')

    if use_spn:
        # attach_attention_module(x, 'sa_block')
        # print("SPN in bottle block")
        num_guidance = 2
        with tf.variable_scope('CSPN') as scope:
            guidance_out = layers.down_sample_resnet(attach_attention_module(downsampling_conv[scale_space_num - 2], 'sa_block'),
                            act_feature_num, act_feature_num * num_guidance,
                            filter_size, res_depth, ksize_pooling, activation)
            input_tensor = cspn(input_tensor, None, guidance_out, kernel_size=3, num_layers=8)
    print("Upsampling block")
    # Upsampling block 
    for layer in range(scale_space_num - 2, -1, -1):
        with tf.variable_scope('unet_upsampling_{}'.format(layer)) as scope:
            downsampling_layer = downsampling_conv[layer]
            output_shape = tf.shape(downsampling_layer)

            transposed_conv = layers.transposed_conv2d_bn_lrn_drop('transposed_conv', input_tensor, 
                              [filter_size, filter_size, act_feature_num, last_feature_num], 
                              output_shape, pool_size, activation=activation)
            concat = tf.concat([downsampling_layer, transposed_conv], axis=3, name='concat')

            if use_residual:
                x = layers.conv2d_bn_lrn_drop('conv1', concat, 
                    [filter_size, filter_size, pool_size * act_feature_num, act_feature_num], 
                    activation=tf.identity)

                original_x = x
                x = tf.nn.relu(x, name='activation')

                for depth in range(0, res_depth):
                    if depth < res_depth - 1:
                        x = layers.conv2d_bn_lrn_drop('conv_res_{}'.format(depth), x, 
                            [filter_size, filter_size, act_feature_num, act_feature_num], 
                            activation=activation)
                    else:
                        x = layers.conv2d_bn_lrn_drop('conv_res_{}'.format(depth), x, 
                            [filter_size, filter_size, act_feature_num, act_feature_num], 
                            activation=tf.identity)
                    
                x += original_x 
                x = activation(x, name='activation')

                if prev_upsampling is not None:
                    prev_upsampling_layer = prev_upsampling[layer]
                    x = tf.concat([prev_upsampling_layer, x], axis=3, name='concat')
                    x = layers.conv2d_bn_lrn_drop('conv1_1', x,
                        [1, 1, 2 * act_feature_num, act_feature_num], activation=activation)

                upsampling_conv[layer] = x
                input_tensor = x
            else:
                conv1 = layers.conv2d_bn_lrn_drop('conv_1', concat, 
                        [filter_size, filter_size, pool_size * act_feature_num, act_feature_num], activation=activation)
            
                input_tensor = layers.conv2d_bn_lrn_drop('conv_2', conv1, 
                        [filter_size, filter_size, act_feature_num, act_feature_num], activation=activation)
        
            last_feature_num = act_feature_num
            act_feature_num /= pool_size

    # if not last_block:
    #     print("Adding attention + spatial transformer network")
    #     attach_attention_module(input_tensor, 'sa_block')
    #     act_feature_num = int(last_feature_num * pool_size)
    #     print("act_feature_num", act_feature_num)
    #     num_guidance = 2
    #     with tf.variable_scope('CSPN') as scope:
    #         guidance_out = layers.down_sample_resnet(upsampling_conv[scale_space_num - 3],
    #                         act_feature_num, int(act_feature_num / num_guidance),
    #                         filter_size, res_depth, ksize_pooling, activation)
    #         input_tensor = cspn(input_tensor, None, guidance_out, kernel_size=3, num_layers=8)

    return input_tensor, downsampling_conv, upsampling_conv 

def create_net(input_tensor, in_channel, n_class, scale_space_num, 
                res_depth, feature_root, filter_size, pool_size, activation):
    """
        Creates a CU-Net instance. This network can process images of arbitrarily sizes
        :param inp: input tensor, shape [?,?,?,channels]
        :param channels: number of channels of the input image
        :param n_class: number of output labels
        :param scale_space_num: number of down-sampling blocks in an encoder
        :param res_depth: depth of residual blocks
        :param featRoot: number of features in the first layer
        :param filter_size: size of the convolution filter
        :param pool_size: size of the max pooling operation
        :param activation: activation function to be used in convolution layers
    """
    input_tensor = tf.map_fn(lambda image: tf.image.per_image_standardization(image), input_tensor)

    use_residual = True 
    use_lstm = False
    use_spn = True

    num_blocks = 3
    use_attention = True
    binary_mask = None 

    input_scale_map = OrderedDict()
    input_scale_map[0] = input_tensor

    last_block = False
    with tf.variable_scope('feature_map') as scope:
        for block_id in range(0, num_blocks):
            with tf.variable_scope('feature_map_{}'.format(block_id)) as scope:
                if block_id == 0:
                    prev_downsampling = None 
                    prev_upsampling = None 
                    num_channels = in_channel
                else:
                    num_channels = n_class

                if use_spn and block_id >= num_blocks - 2:
                    enable_spn = True 
                else:
                    enable_spn = False

                if block_id == num_blocks - 1:
                    last_block = True
                output_tensor, prev_downsampling, prev_upsampling = unet_block(input_tensor, use_residual, 
                                                            use_lstm, enable_spn, use_attention, num_channels, 
                                                            scale_space_num, res_depth, feature_root, filter_size, 
                                                            pool_size, activation, last_block, prev_downsampling=prev_downsampling,
                                                            prev_upsampling=prev_upsampling, binary_mask=binary_mask) 

                output_tensor = layers.conv2d_bn_lrn_drop('n_class', output_tensor, [4, 4, feature_root, n_class], activation=tf.identity)
                input_tensor = output_tensor    
    return output_tensor

class CUNet(object):
    """
        Couple U-net implementation
        :param channels: (optional) number of channels in the input image
        :param n_class: (optional) number of output labels
        :param cost: (optional) name of the cost function. Default is 'cross_entropy'
        :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    def __init__(self, channels=1, n_class=1, model_kwargs={}):
        tf.reset_default_graph()

        self.n_class = n_class
        self.channels = channels

        self.input_tensor = tf.placeholder('float', shape=[None, None, None, self.channels], name='input_tensor')

        self.scale_space_num = model_kwargs.get('scale_space_num', 6)
        self.res_depth = model_kwargs.get('res_depth', 3)
        self.feature_root = model_kwargs.get('feature_root', 8)

        self.filer_size = model_kwargs.get('filter_size', 3)
        self.pool_size = model_kwargs.get('pool_size', 2)
        self.activation = model_kwargs.get('activation', 'relu')

        if self.activation is 'relu':
            self.activation = tf.nn.relu
        elif self.activation is 'elu':
            self.activation = tf.nn.elu 

        
        # self.num_scales = model_kwargs.get('num_scale', 3)
        self.final_activation = model_kwargs.get('final_activation', 'sigmoid')

        print("final activation", self.final_activation)
        logits = create_net(self.input_tensor, self.channels, self.n_class, 
                            self.scale_space_num, self.res_depth, self.feature_root, 
                            self.filer_size, self.pool_size, self.activation)

        print('Total trainable params', self.count_trainable_parameters())
        # self.summary()

        self.logits = tf.identity(logits, 'logits')

        if self.final_activation is "softmax":
            self.predictor = tf.nn.softmax(self.logits, name='output')

        elif self.final_activation is "sigmoid":
            self.predictor = tf.nn.sigmoid(self.logits, name='output')

        elif self.final_activation is "identity":
            self.predictor = tf.identity(self.logits, name='output')

        self.predictor = tf.nn.relu(self.logits, name='output')

        # self.predictor_class = tf.argmax(self.predictor, -1)

        self.predictor_class = self.predictor

    
    def count_trainable_parameters(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    def summary(self):
        """
            Summary layers and theirs params
        """
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def load_weights(self, sess, weights_dict):
        for var_name in weights_dict:
            print('Setting ' + var_name)
            scope = var_name.split('/')[0]
            var = '/'.join(var_name.split('/')[1:])
            if 'Const' in var: continue
            with tf.variable_scope(scope, reuse=True):
                var = tf.get_variable(var)
                try:
                    sess.run(var.assign(weights_dict[var_name]))
                except ValueError as e:
                    print('Error assign weight for {}: {}'.format(var_name, e))

    def save(self, sess, model_path):
        """
            Saves the current session to a checkpoint
            :param sess: current session
            :param model_path: path to file system location
        """
        saver = tf.train.Saver(max_to_keep=3)
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path, var_dict=None):
        """
            Restores a session from a checkpoint
            :param sess: current session instance
            :param model_path: path to file system checkpoint location
        """
        if var_dict is None:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(var_list=var_dict)
        saver.restore(sess, model_path)
        print('Model restored from file: {}'.format(model_path))


        









            
            

    

                    

