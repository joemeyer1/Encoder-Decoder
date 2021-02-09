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

    delete_data: bool = False  # indicates to delete generated images
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

    training_spec = TrainingSpec(
        epochs=8000,
        batch_size=8,
        learning_rate=1e-3,
        test_proportion=.125,
        save_best_net='min_test_loss',
        max_n_epochs_unimproved_loss=10,
        train_until_loss_margin_falls_to=.1,
        save_loss_as=f'losses-optimization/{param_filename}',
        delete_data=delete_data,
    )
    training_spec.check_params(image_spec.n_images)

    # get param filename
    cnn_str = str(encoder_decoder_spec.cnn_shape).replace('(', '').replace(')', '').replace(', ', '_')
    lr_first_digit = str(training_spec.learning_rate).replace('0', '').replace('.', '')[0]
    learning_rate_str = f"{lr_first_digit}e{floor(log(training_spec.learning_rate, 10))}"
    dropout_first_digit = str(encoder_decoder_spec.dropout).replace('0', '').replace('.', '')[0]
    dropout_str = f"{dropout_first_digit}e{floor(log(encoder_decoder_spec.dropout, 10))}"
    param_filename = f"{cnn_str}_cnn_shape--{encoder_decoder_spec.activation}_activation--" \
        f"{encoder_decoder_spec.res_weight}_res_weight--{dropout_str}_dropout--{encoder_decoder_spec.embedding_size}_embedding_size--" \
        f"{encoder_decoder_spec.n_linear_embedding_layers}_n_linear_embedding_layers--" \
        f"{training_spec.batch_size}_batch_size--{learning_rate_str}_lr"
    training_spec.save_loss_as = f'losses-optimization/{param_filename}'

    # checks if image directories for storing generated images exist - if not, makes them
    make_image_directories((
        'before/random',
        'before/original_images',
        'after/random',
        'after/reconstructed_images',
        'nets',
        'losses-optimization',
    ))

    image_data = get_image_data(image_spec)

    encoder_decoder = load_net(net_to_load) if net_to_load else EncoderDecoder(
        encoder_decoder_spec=encoder_decoder_spec,
        img_dim=image_spec.img_dim,
    )

    # get original image
    before_img_name = show_image(image_data[0], "before/original_images/" + param_filename + ".jpg", delete_after=delete_data, i=i)

    # show untrained encoder-decoder's interpretation of random img
    encoded_decoded_random_img_before_training = encoder_decoder.forward(abs(torch.randn(image_data[:1].shape)) * 128)
    before_rand_img_name = show_image(encoded_decoded_random_img_before_training[0], "before/random/"+param_filename+'.jpg', delete_after=delete_data, i=i)

    train_net(net=encoder_decoder, data=image_data, training_spec=training_spec)
    if not delete_data:
        save_net(encoder_decoder, f'nets/{param_filename}-net.pickle', i=i)

    # get trained encoder-decoder's interpretation of random image
    encoded_decoded_random_img = encoder_decoder.forward(abs(torch.randn(1, 3, image_spec.img_dim, image_spec.img_dim))*128)
    after_rand_img_name = show_image(encoded_decoded_random_img[0], "after/random/"+param_filename+'.jpg', delete_after=delete_data, i=i)

    # get trained encoder-decoder's interpretation of original image
    encoded_decoded_img = encoder_decoder.forward(image_data[:1])
    after_img_name = show_image(encoded_decoded_img[0], "after/reconstructed_images/" + param_filename + '.jpg', delete_after=delete_data, i=i)

    def ensure_img_filenames_match():
        img_name_len = len(param_filename)
        before_rand_name_i = before_rand_img_name[len("before/random/")+img_name_len:]
        after_rand_name_i = after_rand_img_name[len("after/random/")+img_name_len:]
        assert before_rand_name_i == after_rand_name_i, "i does not match before / after"
        before_img_name_i = before_img_name[len("before/original_images/")+img_name_len:]
        after_img_name_i = after_img_name[len("after/reconstructed_images/")+img_name_len:]
        assert before_img_name_i == after_img_name_i
    ensure_img_filenames_match()


def make_image_directories(image_directories=(
        'original_images',
        'encoded_decoded_images_before_training',
        'encoded_decoded_random_imgs_before_training',
        'encoded_decoded_images_after_training',
        'encoded_decoded_random_imgs_after_training',
        'double_encoded_decoded_random_imgs_after_training',
        'losses',
)):

    for dir_name in image_directories:
        if not os.path.isdir(dir_name):
            # makedirs allows for more complex paths to be created, versus mkdir
            os.makedirs(dir_name)


if __name__ == '__main__':
    generate_trained_net()

