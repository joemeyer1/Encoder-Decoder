#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


from src.train import load_net
from src.data_utils import get_rand_img_embedding


def generate_image(encoder_decoder_filename):
    encoder_decoder = load_net(encoder_decoder_filename)
    rand_img_embedding = get_rand_img_embedding( encoder_decoder.embedding_size)
    decoder = encoder_decoder.net[1:]
    generated_img = decoder(rand_img_embedding)[0]
    return generated_img


if __name__ == '__main__':
    generate_image("test")

