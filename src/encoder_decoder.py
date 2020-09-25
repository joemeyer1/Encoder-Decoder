#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


from math import log2
from typing import Tuple

from torch import nn, Tensor

from src.cnn import CNN


class EncoderDecoder(nn.Module):
    def __init__(self, img_size=(256, 256), embedding_size=4, cnn_shape=None):
        super(EncoderDecoder, self).__init__()

        self.net = nn.Sequential()

        # # round down to next highest power of 2
        # self.img_size = (2**int(log2(img_size[0])), 2**int(log2(img_size[1])))
        assert img_size[0]%2 == 0
        assert img_size[1]%2 == 0
        self.img_size = img_size
        self.embedding_size = 2**int(log2(embedding_size))
        print(f"embedding size: {self.embedding_size}")

        if cnn_shape == None:
            cnn_shape = self.get_net_shape('encoder')
        self.encoder_shape = cnn_shape
        print(f"encoder shape: {self.encoder_shape}")
        encoder = CNN(shape=self.encoder_shape, stride=2)
        self.net.add_module('encoder', encoder)


        # linear1 = nn.Linear(img_size[-1], img_size[-1] // compression_factor)
        # self.net.add_module('linear1', linear1)
        # # linear2 = nn.Linear(img_size[-1] // 2, img_size[-1] // 4)
        # # self.net.add_module('linear2', linear2)
        # #
        # # linear3 = nn.Linear(img_size[-1] // 4, img_size[-1] // 2)
        # # self.net.add_module('linear3', linear3)
        # linear4 = nn.Linear(img_size[-1] // compression_factor, img_size[-1])
        # self.net.add_module('linear4', linear4)

        self.decoder_shape = self.get_net_shape('decoder')
        print(f"decoder shape: {self.decoder_shape}")
        decoder = CNN(shape=self.decoder_shape, stride=2)
        self.net.add_module('decoder', decoder)

    def forward(self, x: Tensor):
        return self.net(x)

    # Helper for __init__
    def get_net_shape(self, net_type: str) -> Tuple:

        net_shape = []
        next_layer_size = self.img_size[-1]
        assert next_layer_size >= self.embedding_size
        while next_layer_size >= self.embedding_size:
            net_shape.append(next_layer_size)
            next_layer_size //= 2
        if net_type == 'decoder':
            net_shape.reverse()
        return tuple(net_shape)