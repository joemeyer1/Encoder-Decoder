#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

import os
# os.sys.path.append('/Users/joemeyer/Documents/Encoder-Decoder')

from math import log, floor

import torch

from dataclasses import dataclass
from typing import Tuple, Optional

# print(os.sys.path)
from src.encoder_decoder import EncoderDecoder
from src.train import train_net, save_net, load_net

from src.data_utils import ImageSpec, EncoderDecoderSpec, TrainingSpec

from src.image_functions import show_image, show_images, get_image_data


def generate_trained_net():

    net_to_load: Optional[str] = None  # e.g. "nets/net180.pickle"
    i: int = 1000  # i indicates minimum number ID to use for file naming

    image_spec = ImageSpec(dir_name='img_data', n_images=64, img_dim=512)

    encoder_decoder_spec = EncoderDecoderSpec(
        cnn_shape=(5, 3, 3, 1),
        activation='relu',
        compression_factor=2,
        res_weight=0,
        dropout=.05,
        embedding_size=16,
        n_linear_embedding_layers=0,
        n_linear_final_layers=0,
    )

    cnn_str = str(encoder_decoder_spec.cnn_shape).replace('(', '').replace(')', '').replace(', ', '_')
    param_filename = f"{cnn_str}_cnn_shape-{encoder_decoder_spec.activation}_activation-" \
               f"{encoder_decoder_spec.res_weight}_res_weight-{encoder_decoder_spec.embedding_size}_embedding_size-" \
               f"{encoder_decoder_spec.n_linear_embedding_layers}_n_linear_embedding_layers"

    # checks if image directories for storing generated images exist - if not, makes them
    make_directories((
        'trained-nets',
        'trained-net-losses',
    ))

    image_data = get_image_data(image_spec)

    training_spec = TrainingSpec(
        epochs=8000,
        batch_size=8,
        learning_rate=1e-3,
        test_proportion=.125,
        save_best_net='min_test_loss',
        max_n_epochs_unimproved_loss=10,
        train_until_loss_margin_falls_to=.1,
        save_loss_as=f'trained-net-losses/{param_filename}',
        delete_data=False,
    )
    training_spec.check_params(image_spec.n_images)

    def get_param_filename() -> str:
        cnn_str = str(encoder_decoder_spec.cnn_shape).replace('(', '').replace(')', '').replace(', ', '_')
        lr_first_digit = str(training_spec.learning_rate).replace('0', '').replace('.', '')[0]
        learning_rate_str = f"{lr_first_digit}e{floor(log(training_spec.learning_rate, 10))}"
        dropout_first_digit = str(encoder_decoder_spec.dropout).replace('0', '').replace('.', '')[0]
        dropout_str = f"{dropout_first_digit}e{floor(log(encoder_decoder_spec.dropout, 10))}"
        param_filename = f"{cnn_str}_cnn_shape--{encoder_decoder_spec.activation}_activation--" \
            f"{encoder_decoder_spec.res_weight}_res_weight--{dropout_str}_dropout--{encoder_decoder_spec.embedding_size}_embedding_size--" \
            f"{encoder_decoder_spec.n_linear_embedding_layers}_n_linear_embedding_layers--" \
            f"{training_spec.batch_size}_batch_size--{learning_rate_str}_lr"
        return param_filename

    param_filename = get_param_filename()
    training_spec.save_loss_as = f'trained-net-losses/{param_filename}'

    encoder_decoder = load_net(net_to_load) if net_to_load else EncoderDecoder(
        encoder_decoder_spec=encoder_decoder_spec,
        img_dim=image_spec.img_dim,
    )

    train_net(net=encoder_decoder, data=image_data, training_spec=training_spec)
    save_net(encoder_decoder, f'trained-nets/{param_filename}-net.pickle', i=i)


def make_directories(image_directories=(
            'trained-nets',
            'trained-net-losses',
)):

    for dir_name in image_directories:
        if not os.path.isdir(dir_name):
            # makedirs allows for more complex paths to be created, versus mkdir
            os.makedirs(dir_name)


if __name__ == '__main__':
    generate_trained_net()

