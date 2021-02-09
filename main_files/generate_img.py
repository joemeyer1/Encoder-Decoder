#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


import torch

from src.train import train_net, save_net, load_net


def generate_image(encoder_decoder_filename):
    encoder_decoder = load_net(encoder_decoder_filename)
    rand_img_embedding = get_rand_img_embedding( encoder_decoder.embedding_size)
    decoder = encoder_decoder.net[1:]
    generated_img = decoder(rand_img_embedding)[0]
    return generated_img

def get_rand_img_embedding(embedding_size, rand_embedding_type='both', brightness=200):
    randn_input_embedding = lambda: abs(torch.randn(size=(1, 3, embedding_size, embedding_size))) * brightness
    randu_input_embedding = lambda: torch.randint(low=-brightness, high=brightness, size=(1, 3, embedding_size, embedding_size), dtype=torch.float32)
    if rand_embedding_type == "randn":
        input_embedding = randn_input_embedding()
    elif rand_embedding_type == 'randu':
        input_embedding = randu_input_embedding()
    elif rand_embedding_type == 'both':
        input_embedding = randn_input_embedding() + randu_input_embedding()
    else:
        raise Exception("Must specify rand_embedding_type='randn' or 'randu' or 'both'")
    return input_embedding


if __name__ == '__main__':
    generate_image("test")

