from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import click

from model.model import CUNet
from model.training.trainer import Trainer
from data_loader.data_generator import DataGenerator

@click.command()
@click.option('--path_list_train', default="./train.lst")
@click.option('--path_list_val', default="./val.lst")
@click.option('--output_folder', default="./weights/")
@click.option('--restore_path', default=None)

def main(path_list_train, path_list_val, output_folder, restore_path):
    # Since the input images are of arbitrarily size, the autotune will significantly slow down training!
    # (it is calculated for each image)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    img_channels = 1  # Number of image channels (gray scale)
    n_class = 1   # Number of output classes

    ### data generator parameters
    data_kwargs = dict(batch_size_training=1,\
                        scale_min=0.7,\
                        scale_max=0.7,
                        scale_val=0.7,\
                        affine_training=False,\
                        one_hot_encoding=True)

    data_provider = DataGenerator(path_list_train,\
                                path_list_val,\
                                n_class,
                                thread_num=1,\
                                queue_capacity=1,\
                                label_prefix='labels',\
                                data_kwargs=data_kwargs)

    ### model hyper-parameters
    model_kwargs = dict(activation='relu',\
                        num_unets=3,\
                        number_scale=3,\
                        feature_root=16,\
                        res_depth=3,\
                        attention=True,\
                        residual=True,\
                        dilated=False,\
                        final_activation='relu')

    model = CUNet(img_channels, n_class, model_kwargs=model_kwargs)
    opt_kwargs = dict(optimizer='adam', learning_rate=1e-4)
    loss_kwargs = dict(loss_name='mse', act_name='identity')

    # start training
    trainer = Trainer(model, opt_kwargs=opt_kwargs, loss_kwargs=loss_kwargs)
    trainer.train(data_provider, output_folder, restore_path, batch_steps_per_epoch=256, epochs=100, gpu_device=0)


if __name__ == "__main__":
    main()
