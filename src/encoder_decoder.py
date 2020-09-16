#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


from math import log2
from typing import Tuple

from torch import nn, Tensor

from src.cnn import CNN


class EncoderDecoder(nn.Module):
    def __init__(self, img_size=(256, 256), embedding_size=64):
        super(EncoderDecoder, self).__init__()

        self.net = nn.Sequential()

        # round down to next highest power of 2
        self.img_size = (2**int(log2(img_size[0])), 2**int(log2(img_size[1])))
        self.embedding_size = 2 ** int(log2(embedding_size))

        self.encoder_shape = self.get_net_shape(net_type='encoder')
        print(f"encoder shape: {self.encoder_shape}")
        encoder = CNN(shape=self.encoder_shape)
        self.net.add_module('encoder', encoder)

        self.decoder_shape = self.get_net_shape(net_type='decoder')
        print(f"decoder shape: {self.decoder_shape}")
        decoder = CNN(shape=self.decoder_shape)
        self.net.add_module('decoder', decoder)

    def forward(self, x: Tensor):
        return self.net(x)

    # Helper for __init__
    def get_net_shape(self, net_type: str) -> Tuple:
        net_shape = []
        layer_size = self.img_size[-1]
        while layer_size >= self.embedding_size:
            net_shape.append(layer_size)
            layer_size //= 2
        if net_type == 'decoder':
            net_shape.reverse()
        return tuple(net_shape)