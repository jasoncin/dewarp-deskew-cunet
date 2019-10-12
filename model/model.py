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
import logging
import tensorflow as tf
from collections import OrderedDict
import tensorflow.contrib.slim as slim

from .layers import layers
from .layers.cspn import cspn
from .layers.attention import attach_attention_module

__author__ = "Kristopher"
__email__ = "kristopher@cinnamon.is"
__status__ = "Module"



class CUNet(object):
    """ Couple U-net implementation
    Aruguments:
        channels: (optional) number of channels in the input image
        n_class: (optional) number of output labels
        model_kwargs: (optional) kwargs passed to the model configuration
    """
    def __init__(self, channels=1, n_class=1, model_kwargs={}):
        tf.reset_default_graph()

        self.n_class = n_class
        self.channels = channels

        self.input_tensor = tf.placeholder('float', \
            shape=[None, None, None, self.channels], name='input_tensor')

        self.number_scale = model_kwargs.get('number_scale', 3)
        self.res_depth = model_kwargs.get('res_depth', 3)
        self.feature_root = model_kwargs.get('feature_root', 8)
        self.num_unets = model_kwargs.get('num_unets', 2)

        self.filter_size = model_kwargs.get('filter_size', 3)
        self.pool_size = model_kwargs.get('pool_size', 2)
        self.activation = model_kwargs.get('activation', 'relu')

        if self.activation is 'relu':
            self.activation = tf.nn.relu
        elif self.activation is 'elu':
            self.activation = tf.nn.elu

        self.dilated_rates = [2, 3, 4, 5]
        self.use_dilated_conv = model_kwargs.get('dilated', False)

        # self.num_scales = model_kwargs.get('num_scale', 3)
        self.final_activation = model_kwargs.get('final_activation', 'sigmoid')

        self.use_residual = model_kwargs.get('residual', False)
        self.use_spn = model_kwargs.get('spn', False)
        self.use_attention = model_kwargs.get('attention', False)

        print("final activation: {0}".format(self.final_activation))
        all_logits = self.stacking_unet(self.input_tensor)
        print("Total trainable params: {0}".format(self.count_trainable_parameters()))

        # self.logits = tf.identity(logits, 'logits')
        #import pdb; pdb.set_trace()
        self.all_logits = list(map(lambda item: \
            tf.identity(item[1], 'logits_{0}'.format(item[0])), all_logits))

        if self.final_activation is "softmax":
            self.all_predicts = list(map(lambda item: \
                tf.nn.softmax(item[1], name='output_{0}'.format(item[0])), all_logits))
        elif self.final_activation is "sigmoid":
            self.all_predicts = list(map(lambda item: \
                tf.nn.sigmoid(item[1], 'output_{0}'.format(item[0])), all_logits))
        elif self.final_activation is "identity":
            self.all_predicts = list(map(lambda item: \
                tf.identity(item[1], 'output_{0}'.format(item[0])), all_logits))
        elif self.final_activation is "relu":
            self.all_predicts = list(map(lambda item: \
                tf.nn.relu(item[1], 'output_{0}'.format(item[0])), all_logits))
        else:
            self.all_predicts = all_logits

        # self.predictor_class = tf.argmax(self.predictor, -1)
        self.predictor_class = self.all_predicts[-1]
        # self.summary()

    def unet(self, input_tensor, inchannel_num, \
        feature_num, activation, last_unet, \
        prev_downsampling=None, prev_upsampling=None):
        """ Define an U-Net block which consists of an encoder
        (down-sampling layers) and a decoder (up-sampling layers)
        Aruguments:
            input_tensor: input image
            inchannel_num: number of input channels
            feature_num: number of init features
            last_unet: Whether is the last layer block or not of each unet
            prev_downsampling: previous down-sampling tower's outputs (used for coupling connection)
            prev_upsampling: previous up-sampling tower's outputs (used for coupling connection)
        Returns:
            input_tensor: original input tensor
            downsampling_conv: Dict of Downsampling feature tensors
            upsampling_conv: Dict of Upsampling feature tensors
        """

        downsampling_conv = OrderedDict()
        upsampling_conv = OrderedDict()
        ksize_pooling = [1, self.pool_size, self.pool_size, 1]
        stride_pooling = [1, self.pool_size, self.pool_size, 1]

        # Downsampling block
        print("Downsampling block")
        for idx in range(0, self.number_scale):
            with tf.variable_scope('unet_downsampling_{}'.format(idx)) as scope:
                if self.use_residual:
                    x = layers.conv2d_bn_lrn_drop('conv_{}'.format(idx), \
                        input_tensor, [self.filter_size, self.filter_size, \
                        inchannel_num, feature_num], activation=tf.identity)

                    # Get original input used for skipped connection
                    original_x = x
                    x = tf.nn.relu(x, name='activation')

                    # Apply spatial transformer network
                    # x = layers.spatial_transformer_layer('stn_{}'.format(idx), \
                    #    x, tf.shape(x)[1:2], [self.filter_size, self.filter_size, \
                    #    feature_num, feature_num], activation=activation)

                    for depth in range(0, self.res_depth):
                        if (depth >= self.res_depth - 1):
                            activation = tf.identity

                        x = layers.conv2d_bn_lrn_drop('conv_res_{}'.format(depth), x, \
                            [self.filter_size, self.filter_size, feature_num, feature_num], \
                            activation=activation)

                    x += original_x
                    x = activation(x, name='activation')

                    if prev_downsampling is not None:
                        prev_downsampling_layer = prev_downsampling[idx]
                        x = tf.concat([prev_downsampling_layer, x], axis=3, name='concat')
                        x = layers.conv2d_bn_lrn_drop('conv1_1', x, \
                            [1, 1, 2 * feature_num, feature_num], activation=activation)

                    # if (self.use_attention and idx > self.number_scale - 2):
                    #     if last_unet:
                    #         downsampling_conv[idx] = attach_attention_module(x, 'sa_block')
                    # else:
                    #     downsampling_conv[idx] = x
                    downsampling_conv[idx] = x
                else:
                    conv1 = layers.conv2d_bn_lrn_drop('conv_1', \
                        input_tensor, [self.filter_size, self.filter_size, \
                        inchannel_num, inchannel_num], activation=activation)

                    downsampling_conv[idx] = layers.conv2d_bn_lrn_drop('conv_2', \
                        conv1, [self.filter_size, self.filter_size, \
                        feature_num, feature_num], activation=activation)

                if idx < self.number_scale - 1:
                    input_tensor = tf.nn.max_pool(x, ksize_pooling, \
                    stride_pooling, padding='SAME', name='pooling')
                else:
                    input_tensor = x

                inchannel_num = feature_num
                feature_num *= self.pool_size

        # Last number of feature maps
        last_feature_num = int(feature_num / self.pool_size)

        # Bottleneck block
        print("Bottleneck block")

        if self.use_spn:
            num_guidance = 2
            with tf.variable_scope('CSPN') as scope:
                guidance_out = layers.down_sample_resnet(attach_attention_module(\
                    downsampling_conv[self.number_scale - 2], 'sa_block'), \
                    feature_num, feature_num, * num_guidance, self.filter_size, \
                    self.res_depth, ksize_pooling, activation)
                input_tensor = cspn(input_tensor, None, guidance_out, kernel_size=3, num_layers=8)
        else:
            with tf.variable_scope('Bottleneck') as scope:
                original_x = x # Get original input used for skipped connection
                x = layers.conv2d_bn_lrn_drop('bottleneck_conv', \
                    input_tensor, [self.filter_size, self.filter_size, \
                    last_feature_num, feature_num], activation=activation)

                if self.use_attention:
                    x = attach_attention_module(x, 'sa_block')
                elif (self.use_dilated_conv and len(self.dilated_rates) > 0):
                    list_dilated_conv = list()
                    for i, d_rate in enumerate(self.dilated_rates):
                        dilated_conv = layers.dilated_conv2d_bn_lrn_drop(\
                            'bottleneck_dilated_conv_{}'.format(i), x, \
                            [self.filter_size, self.filter_size, feature_num, feature_num], \
                            dilated_rate=3, activation=activation)
                        list_dilated_conv.append(dilated_conv)
                    x = tf.concat(list_dilated_conv, axis=3, name='concat_dilated_convs')
                else:
                    x = layers.conv2d_bn_lrn_drop('bottleneck_conv_ac', x, \
                        [self.filter_size, self.filter_size, feature_num, feature_num], \
                        activation=activation)

                print(x, original_x, feature_num)
                x = layers.conv2d_bn_lrn_drop('bottleneck_conv_id', x, \
                        [1, 1, x.get_shape()[-1].value, last_feature_num], activation=tf.identity)
                x += original_x
                x = tf.nn.relu(x, name='activation_2')

        input_tensor = x
        feature_num = int(last_feature_num / self.pool_size)

        print("Upsampling block")
        # Upsampling block
        for idx in range(self.number_scale - 2, -1, -1):
            with tf.variable_scope('unet_upsampling_{}'.format(idx)) as scope:
                downsampling_layer = downsampling_conv[idx]
                output_shape = tf.shape(downsampling_layer)

                transposed_conv = layers.transposed_conv2d_bn_lrn_drop('transposed_conv_{}'.format(idx), \
                    input_tensor, [self.filter_size, self.filter_size, \
                    feature_num, last_feature_num], output_shape, self.pool_size, activation=activation)

                concat = tf.concat([downsampling_layer, transposed_conv], axis=3, name='concat')

                if self.use_residual:
                    x = layers.conv2d_bn_lrn_drop('conv1', \
                        concat, [self.filter_size, self.filter_size, \
                        self.pool_size * feature_num, feature_num],
                        activation=tf.identity)

                    original_x = x
                    x = tf.nn.relu(x, name='activation')

                    for depth in range(0, self.res_depth):
                        if depth >= self.res_depth - 1:
                            activation = tf.identity

                        x = layers.conv2d_bn_lrn_drop('conv_res_{}'.format(depth), \
                            x, [self.filter_size, self.filter_size, feature_num, feature_num], \
                            activation=activation)

                    x += original_x
                    x = activation(x, name='activation')

                    if prev_upsampling is not None:
                        prev_upsampling_layer = prev_upsampling[idx]
                        x = tf.concat([prev_upsampling_layer, x], axis=3, name='concat')
                        x = layers.conv2d_bn_lrn_drop('conv1_1', \
                            x, [1, 1, self.pool_size * feature_num, \
                            feature_num], activation=activation)

                    upsampling_conv[idx] = x
                    input_tensor = x
                else:
                    conv1 = layers.conv2d_bn_lrn_drop('conv_1', \
                        concat, [self.filter_size, self.filter_size, \
                        self.pool_size * feature_num, feature_num],\
                        activation=activation)

                    input_tensor = layers.conv2d_bn_lrn_drop('conv_2', \
                        conv1, [self.filter_size, self.filter_size, \
                        feature_num, feature_num], activation=activation)

                last_feature_num = feature_num
                feature_num /= self.pool_size

        return input_tensor, downsampling_conv, upsampling_conv

    def stacking_unet(self, input_tensor):
        """ Creates a CU-Net instance. This network
            can process images of arbitrarily sizes
            Aruguments:
                input_tensor:
            Returns:
                output_tensor: output tensor
        """

        input_scale_map = OrderedDict()
        input_scale_map[0] = input_tensor
        output_list = list()

        input_tensor = tf.map_fn(\
            lambda image: tf.image.per_image_standardization(image), input_tensor)

        last_unet = False
        with tf.variable_scope('feature_map') as scope:
            for net_id in range(0, self.num_unets):
                with tf.variable_scope('feature_map_{}'.format(net_id)) as scope:
                    if net_id == 0:
                        prev_downsampling = None
                        prev_upsampling = None
                        num_channels = self.channels
                    else:
                        num_channels = self.n_class

                    if net_id == self.num_unets - 1:
                        last_unet = True

                    output_tensor, prev_downsampling, prev_upsampling = \
                        self.unet(input_tensor, num_channels, \
                            self.feature_root, self.activation, last_unet, \
                            prev_downsampling=prev_downsampling, \
                            prev_upsampling=prev_upsampling)

                    output_tensor = layers.conv2d_bn_lrn_drop('n_class', \
                        output_tensor, [3, 3, self.feature_root, self.n_class], \
                        activation=tf.identity)
                    input_tensor = output_tensor
                    output_list.append((net_id, output_tensor))

        return output_list

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
