from __future__ import print_function, division

import os
import time

import tensorflow as tf
import numpy as np

from .loss import get_loss, get_mean_iou
from .optimizer import get_optimizer
from utils.eval_segm import mean_IU


class Trainer(object):
    """
    Trains a CU-Net instance
    :param net: the CU-Net-net instance to train
    :param opt_kwargs: (optional) kwargs passed to the optimizer
    :param loss_kwargs: (optional) kwargs passed to the loss function
    """

    def __init__(self, net, opt_kwargs={}, loss_kwargs={}):
        self.net = net
        self.label = tf.placeholder("float", shape=[None, None, None, self.net.n_class])
        # self.label_class = tf.argmax(self.label, axis=-1)
        self.label_class = self.label

        self.global_step = tf.placeholder(tf.int64)
        self.opt_kwargs = opt_kwargs
        self.loss_kwargs = loss_kwargs

        self.loss_type = loss_kwargs.get("loss_name", "cross_entropy")

    def _initialize(self, batch_steps_per_epoch, output_path):
        self.loss, self.final_loss = get_loss(self.net.logits, self.label, self.loss_kwargs)
        self.acc, self.acc_update = get_mean_iou(self.net.predictor_class, self.label_class, num_class=self.net.n_class, ignore_class_id=0)

        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="m_metrics")
        self.reset_metrics_op = tf.variables_initializer(var_list=running_vars)

        self.optimizer, self.ema, self.learning_rate_node = get_optimizer(self.loss, self.global_step,
                                                                batch_steps_per_epoch, self.opt_kwargs)
        init = tf.global_variables_initializer()
        if not output_path is None:
            output_path = os.path.abspath(output_path)
            if not os.path.exists(output_path):
                print("Allocating '{:}'".format(output_path))
                os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, restore_file=None, batch_steps_per_epoch=1024, epochs=250,
              gpu_device='0', max_spat_dim=5000000):
        """
        Launches the training process
        :param data_provider:
        :param output_path:
        :param restore_path:
        :param batch_size:
        :param batch_steps_per_epoch:
        :param epochs:
        :param keep_prob:
        :param gpu_device:
        :param max_spat_dim:
        :return:
        """
        print("Epochs: " + str(epochs))
        print("Batch Size Train: " + str(data_provider.batch_size_training))
        print("Batchsteps per Epoch: " + str(batch_steps_per_epoch))
        if not output_path is None:
            save_path = os.path.join(output_path, "model")
        if epochs == 0:
            return save_path

        init = self._initialize(batch_steps_per_epoch, output_path)

        val_size = data_provider.size_validation

        # gpu_options = tf.GPUOptions(visible_device_list=gpu_device)
        # session_conf = tf.ConfigProto()
        # session_conf.gpu_options.visible_device_list=gpu_device
        # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4

        gpu_options = tf.GPUOptions(visible_device_list=str(gpu_device), per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # with tf.Session(config=session_conf) as sess:
            sess.run(init)
            sess.run(tf.local_variables_initializer())

            if restore_file != None:
                print("Loading Checkpoint.")
                self.net.restore(sess, restore_file)
            else:
                print("Starting from scratch.")

            print("Start optimization")

            bestAcc = 1111110.0
            shown_samples = 0
            for epoch in range(epochs):
                total_loss = 0
                total_loss_final = 0
                total_acc = 0
                lr = 0
                time_step_train = time.time()
                for step in range((epoch * batch_steps_per_epoch), ((epoch + 1) * batch_steps_per_epoch)):
                    sess.run(self.reset_metrics_op)
                    batch_img, batch_mask = data_provider.next_data('training')
                    skipped = 0
                    if batch_img is None:
                        print("No Training Data available. Skip Training Path.")
                        break
                    while batch_img.shape[1] * batch_img.shape[2] > max_spat_dim:
                        batch_img, batch_mask = data_provider.next_data('training')
                        skipped = skipped + 1
                        if skipped > 100:
                            print("Spatial Dimension of Training Data to high. Aborting.")
                            return save_path

                    # Run training
                    if self.final_loss is not None:
                        _, loss, final_loss, acc, lr = sess.run \
                            ([self.optimizer, self.loss, self.final_loss, self.acc_update, self.learning_rate_node],
                             feed_dict={self.net.input_tensor: batch_img,
                                        self.label: batch_mask,
                                        self.global_step: step})
                        total_loss_final += final_loss
                    else:
                        _, loss, acc, lr = sess.run \
                            ([self.optimizer, self.loss, self.acc_update, self.learning_rate_node],
                             feed_dict={self.net.input_tensor: batch_img,
                                        self.label: batch_mask,
                                        self.global_step: step})

                    acc = sess.run(self.acc)

                    shown_samples = shown_samples + batch_img.shape[0]
                    if self.loss_type is "cross_entropy_sum":
                        shape = batch_img.shape
                        loss /= shape[1] * shape[2] * shape[0]
                    total_loss += loss
                    total_acc += acc

                total_loss = total_loss / batch_steps_per_epoch
                total_loss_final = total_loss_final / batch_steps_per_epoch
                total_acc = total_acc / batch_steps_per_epoch

                time_used = time.time() - time_step_train
                train_total_loss = total_loss
                self.output_epoch_stats_train(epoch + 1, total_loss, total_loss_final, total_acc, shown_samples, lr, time_used)

                ### VALIDATION
                total_loss = 0
                total_loss_final = 0
                total_acc = 0
                total_m_iou = 0

                time_step_val = time.time()
                for step in range(0, val_size):
                    sess.run(self.reset_metrics_op)
                    batch_img, batch_mask = data_provider.next_data('validation')
                    if batch_img is None:
                        print("No Validation Data available. Skip Validation Path.")
                        break
                    # Run validation
                    if self.final_loss is not None:
                        loss, final_loss, acc, batch_pred = sess.run([self.loss, self.final_loss, self.acc_update, self.net.predictor],
                                                                     feed_dict={self.net.input_tensor: batch_img, self.label: batch_mask})
                        total_loss_final += final_loss
                    else:
                        loss, acc, batch_pred = sess.run([self.loss, self.acc_update, self.net.predictor],
                                                         feed_dict={self.net.input_tensor: batch_img, self.label: batch_mask})

                    acc = sess.run(self.acc)
                    iou_list = []
                    for pred, label in zip(batch_pred, batch_mask):
                        pred = np.argmax(pred, axis=-1)
                        mask = np.argmax(label, axis=-1)
                        iou = mean_IU(pred, mask)
                        iou_list.append(iou)
                    m_iou = np.mean(iou_list)
                    total_m_iou += m_iou

                    if self.loss_type is "cross_entropy_sum":
                        shape = batch_img.shape
                        loss /= shape[1] * shape[2] * shape[0]
                    total_loss += loss
                    total_acc += acc

                if val_size != 0:
                    total_loss = total_loss / val_size
                    total_loss_final = total_loss_final / val_size
                    total_acc = total_acc / val_size
                    total_m_iou /= val_size

                    time_used = time.time() - time_step_val
                    self.output_epoch_stats_val(epoch + 1, total_loss, total_loss_final, total_acc, total_m_iou, time_used)
                    data_provider.restart_val_runner()

                if not output_path is None:
                    if total_loss <= bestAcc: #or (epoch + 1) % 8 == 0:
                        # if total_acc > bestAcc:
                        bestAcc = total_loss
                        save_pathAct = save_path + str(epoch + 1)
                        print("Saving checkpoint")
                        self.net.save(sess, save_pathAct)

            data_provider.stop_all()
            print("Optimization Finished!")
            print("Best Val Loss: " + str(bestAcc))
            return save_path

    def output_epoch_stats_train(self, epoch, total_loss, total_loss_final, acc, shown_sample, lr, time_used):
        print(
            "TRAIN: Epoch {:}, Average loss: {:.6f}  final: {:.6f}  acc: {:.4f}, training samples shown: {:}, learning rate: {:.6f}, time used: {:.2f}".format(
                epoch, total_loss, total_loss_final, acc, shown_sample, lr, time_used))

    def output_epoch_stats_val(self, epoch, total_loss, total_loss_final, acc, m_iou, time_used):
        print(
            "VAL: Epoch {:}, Average loss: {:.6f}  final: {:.6f}  acc: {:.4f}  mIoU: {:.4f}, time used: {:.2f}".format(epoch, total_loss,
                                                                                            total_loss_final, acc, m_iou, time_used))