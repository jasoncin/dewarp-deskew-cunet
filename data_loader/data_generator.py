from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import errno
import json 
import random 
import threading 
import numpy as np
from queue import Queue
from random import uniform
from random import shuffle 
from scipy import misc
from scipy import ndimage 
from matplotlib import pyplot as plt 

from utils.path_util import read_image_list
from utils.generic_util import decision
from utils.image_util import (affine_transform, \
                elastic_transform, to_categorical)


class DataGenerator(object):
    """

    """
    def __init__(self, path_list_train, path_list_eval, n_classes, 
                thread_num=4, queue_capacity=64, label_prefix='mask', data_kwargs={}):
        """
            Define data generator parameters / flags
            Args:
                path_list_train: Path to training file list
                path_list_val: Path to validation file list
                n_classes: Number of output classes
                thread_num: Number of concurrent threads to generate data
                queue_capacity: Maximum queue capacity (reduce this if you have out-of-memory issue)
                data_kwargs: Additional keyword arguments for data augmentation (geometric transforms, rotation, ...)
            Returns:
                a DataGenerator instance
        """
        self.n_classes = n_classes
        self.label_prefix = label_prefix

        self.list_training = None 
        self.size_training = 0 
        self.q_training = None

        self.list_validation = None 
        self.size_validation = 0
        self.q_validation = None 

        """ 
            Augmentation parameters 
        """

        # Batch size for training & validation 
        self.batch_size_training = data_kwargs.get('batch_size_training', 1)
        self.batch_size_validation = data_kwargs.get('batch_size_validation', 1)

        # Use affine transformation 
        self.affine_training = data_kwargs.get('affine_training', False)
        self.affine_validation = data_kwargs.get('affine_validation', False)
        self.affine_value = data_kwargs.get('affine_value', 0.025)

        # Use elastic transformation 
        self.elastic_training = data_kwargs.get('elastic_training', False)
        self.elastic_validation = data_kwargs.get('elastic_validation', False)
        self.elastic_value_x = data_kwargs.get('elastic_value_x', 0.0002)
        self.elastic_value_y = data_kwargs.get('elastic_value_y', 0.0002)

        # Use random rotation
        self.rotate_training = data_kwargs.get('rotate_training', False)
        self.rotate_valiation = data_kwargs.get('rotate_validation', False)
        self.rotate_90_training = data_kwargs.get('rotate_90_training', False)
        self.rotate_90_validation = data_kwargs.get('rotate_90_validation', False)

        #
        self.dilated_num = data_kwargs.get('dilated_num', 1)

        #
        self.scale_min = data_kwargs.get('scale_min', 1.0)
        self.scale_max = data_kwargs.get('scale_max', 1.0)
        self.scale_val = data_kwargs.get('scale_val', 1.0)

        # Ensure one-hot encoding consistency after geometric distortions
        self.one_hot_encoding = data_kwargs.get('one_hot_encoding', True)
        self.dominating_channel = data_kwargs.get('dominating_channel', 0)
        self.dominating_channel = min(self.dominating_channel, n_classes-1)

        # shuffle dataset after each epoch 
        self.shuffle = data_kwargs.get('shuffle', True)

        self.thread_num = thread_num
        self. queue_capacity = queue_capacity
        self.stop_training = threading.Event()
        self.stop_validation = threading.Event() 

        # Start data generator thread(s) to fill the training queue 
        if path_list_train != None:
            self.list_training = read_image_list(path_list_train, prefix=None)
            self.size_training = len(self.list_training)
            self.q_training, self.thread_training = self._get_list_queue(self.list_training, self.thread_num, 
                                                                         self.queue_capacity, self.stop_training, 
                                                                         self.batch_size_training, self.scale_min, 
                                                                         self.scale_max, self.affine_training, 
                                                                         self.elastic_training, self.rotate_training, 
                                                                         self.rotate_90_training)

        # Start data generator thread(s) to fill the validation queue 
        if path_list_eval != None:
            self.list_validation = read_image_list(path_list_eval, prefix=None)
            self.size_validation = len(self.list_validation)
            self.q_validation, self.thread_validation = self._get_list_queue(self.list_validation, self.thread_num, 
                                                                             self.queue_capacity, self.stop_validation, 
                                                                             self.batch_size_validation, self.scale_val, 
                                                                             self.scale_val, self.affine_validation, 
                                                                             self.elastic_validation, self.rotate_valiation, 
                                                                             self.rotate_90_validation)


    def next_data(self, name):
        """
            Return next data from the queue 
        """
        if name is 'validation':
            q = self.q_validation

        elif name is 'training':
            q = self.q_training
        
        if q is None:
            return None, None 
        
        return q.get() 

    def stop_all(self):
        """
            Stop all data generator threads
        """
        self.stop_training.set()
        self.stop_validation.set()

    def restart_val_runner(self):
        """
            Restart validation runner
        """
        if self.list_validation != None:
            self.stop_validation.set()
            self.stop_validation = threading.Event()
            self.q_validation, self.thread_validation = self._get_list_queue(self.list_validation, 1, 100, self.stop_validation,
                                                                             self.batch_size_validation, self.scale_val, 
                                                                             self.scale_val, self.affine_validation, 
                                                                             self.elastic_validation, self.rotate_valiation, 
                                                                             self.rotate_90_validation)

    def _get_list_queue(self, input_list, thread_num, queue_capacity, 
                        stop_event, batch_size, min_scale, max_scale, 
                        affine, elastic, rotate, rotate_90):
        """
            Create a queue and add dedicated generator thread(s) to fill it
        """

        q = Queue(maxsize = queue_capacity)
        threads = []

        for t in range(thread_num):
            threads.append(threading.Thread(target=self._fillQueue, args=(
            q, input_list[:], stop_event, batch_size, min_scale, max_scale, 
            affine, elastic, rotate, rotate_90)))

        for t in threads:
            t.start()
        return q, threads
    
    def _get_mask_image(self, path, num_mask_per_sample=1):
        """
            Create a list of mask images with respect to each image sample 
            :param path: the image path 
            :param num_mask_per_sample: number of masks for that image sample (mask channel) 
            :param prefix: name of folder containing mask images 
            :return: a list of mask images 
        """
        if num_mask_per_sample < 1:
            raise ValueError('{} should not be less than 1!'.format(num_mask_per_sample))
        else:
            list_mask = [] 
            file_path, file_extension = os.path.splitext(path)
            dir_file, file_name = os.path.split(file_path)
            print(dir_file)
            print(file_name)
            check_dir = os.path.join(os.path.dirname(dir_file), self.label_prefix)
            if not os.path.exists(check_dir):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), check_dir)

            if num_mask_per_sample == 1:
                mask_idx_path = os.path.join(os.path.dirname(dir_file), \
                    self.label_prefix, '{}{}'.format(file_name, file_extension))
                list_mask.append(misc.imread(mask_idx_path))
            else:
                for idx in range(0, num_mask_per_sample):
                    mask_idx_path = os.path.join(os.path.dirname(dir_file), \
                        self.label_prefix, '{}_{}{}'.format(file_name, idx, file_extension))
                    list_mask.append(misc.imread(mask_idx_path))

            return list_mask

    def _fillQueue(self, q, input_list, stop_event, batch_size, 
                min_scale, max_scale, affine, elastic, rotate, rotate_90):
        """
            Main function to generate new input-output pair an put it into the queue
            Args:
                q: Output queue
                input_list: List of input JSON(s)
                stop_event: Thread stop event
                batch_size: Batch-size
                affine: Use affine transform
                elastic: Use elastic transform
                rotate: Use random rotation
                rotate_90: Use random rotation (constrained to multiple of 90 degree)
            Returns:
                None
        """

        if self.shuffle:
            shuffle(input_list)
        index = 0
        batch_pair = None

        while (not stop_event.is_set()):
            if batch_pair is None:
                imgs = []
                new_imgs = []

                masks = []
                new_masks = []

                max_height = 0
                max_width = 0

                while len(imgs) < batch_size:
                    if index == len(input_list):
                        if self.shuffle:
                            shuffle(input_list)
                        index = 0
                    try:
                        image_path = input_list[index]
                    except IndexError:
                        print(index, len(input_list))

                    scale = uniform(min_scale, max_scale)
                    pair_maps = []
                    img_channels = 0
                    mask_channels = self.n_classes

                    # filename, file_extension = os.path.splitext(image_path)
                    input_img = misc.imread(image_path)

                    if len(input_img.shape) == 2:
                        pair_maps.append(input_img)
                        img_channels += 1
                    elif len(input_img.shape) == 3:
                        for channel in range(0, input_img.shape[2]):
                            pair_maps.append(input_img[:, :, channel])
                            img_channels += 1
                    
                    list_mask = self._get_mask_image(image_path, mask_channels)
                    pair_maps.extend(list_mask)

                    res = np.dstack([np.expand_dims(misc.imresize(pair_maps[i], scale, interp='bicubic'), 2) \
                                for i in range(0, img_channels + mask_channels)])

                    if affine:
                        res = affine_transform(res, self.affine_value)
                    if elastic:
                        res = elastic_transform(res, self.elastic_value_x, self.elastic_value_y)
                    if rotate or rotate_90:
                        angle = uniform(-20, 20)
                        if rotate_90:
                            if angle < 0:
                                angle = -45.0
                            elif angle < 45:
                                angle = -45.0
                            elif angle < 90.0:
                                angle = 45.0
                            else:
                                angle = 90.0
                        res = ndimage.interpolation.rotate(res, angle)

                    input_img = res[:, :, 0: img_channels]
                    input_mask = res[:, :, img_channels: ]
                    input_mask = np.where(input_mask > 64, 1.0, 0.0)

                    if self.one_hot_encoding:
                        aMap = input_mask[:, :, self.dominating_channel]
                        for aM in range(0, mask_channels - 1):
                            if aM == self.dominating_channel:
                                continue
                            else:
                                tMap = np.logical_and(input_mask[:, :, aM], np.logical_not(aMap))
                                aMap = np.logical_or(aMap, tMap)
                                input_mask[:, :, aM] = tMap

                        # Add+Calculate the clutter map
                        # print(input_mask.shape)
                        # input_mask = np.pad(input_mask, ((0,0),(0,0),(0,1)), mode='constant')
                        # print(aMap.shape)
                        # input_mask[:, :, mask_channels - 1] = np.logical_not(aMap)
                        # print('*** ', input_mask.shape)
                        # plt.imshow(input_mask[:, :, mask_channels - 1])
                        # plt.show()
                    
                    # input_img = (255.0 - input_img) / 255.0 
                    input_img = input_img / 255.0
                    imgs.append(input_img)
                    masks.append(input_mask)

                    max_width = max(input_img.shape[1], max_width)
                    max_height = max(input_img.shape[0], max_height)

                for img in imgs:
                    height = img.shape[0]
                    pad_h = max_height - height

                    width = img.shape[1]
                    pad_w = max_width - width

                    if pad_h + pad_w > 0:
                        npad = ((0, pad_h), (0, pad_w))
                        img = np.pad(img, npad, mode='constant', constant_values=0)
                    new_imgs.append(np.expand_dims(img, 0))
                
                for mask in masks:
                    height = mask.shape[0]
                    pad_h = max_height - height

                    width = mask.shape[1] 
                    pad_w = max_width - width
                    
                    mask_content = mask[:, :, 0: mask.shape[2] - 1]
                    mask_back_ground = mask[:, :, mask.shape[2] -1]

                    if pad_h + pad_w > 0:
                        npad = ((0, pad_h), (0, pad_w), (0, 0))
                        mask_content = np.pad(mask_content, npad, mode='constant', constant_values=0)
                        mask_back_ground = np.pad(mask_back_ground, npad, mode='constant', constant_values=1)
                    new_masks.append(np.expand_dims(np.dstack([mask_content, mask_back_ground]), 0))

                batch_x = np.concatenate(new_imgs)
                batch_y = np.concatenate(new_masks)
                batch_pair = [batch_x, batch_y]
                # print(batch_x.shape)

            try:
                q.put(batch_pair, timeout=1)
                batch_pair = None
                index += 1
            except:
                continue
