#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

from copy import deepcopy

from math import log2
from typing import Tuple

from math import log

from torch import nn, Tensor

from src.cnn import CNN, ConvBlock


class EncoderDecoder(nn.Module):
    def __init__(self, img_size=(256, 256), embedding_size=4, cnn_shape=None, compression_factor=2):
        super(EncoderDecoder, self).__init__()

        self.net = nn.Sequential()

        # # round down to next highest power of 2
        # self.img_size = (2**int(log2(img_size[0])), 2**int(log2(img_size[1])))
        assert img_size[0]%2 == 0
        assert img_size[1]%2 == 0
        self.img_size = img_size
        self.embedding_size = compression_factor**int(log(embedding_size, compression_factor))
        print(f"embedding size: {self.embedding_size}")

        # if cnn_shape == None:
        #     cnn_shape = self.get_net_shape('encoder')

        self.encoder_shape = []
        encoder = nn.Sequential()
        output_size = img_size[-1]
        encoder_length = max(len(cnn_shape), int(log(output_size / self.embedding_size, compression_factor)))
        for i in range(encoder_length):
            stride = compression_factor if output_size // compression_factor >= embedding_size else 1
            kernel_size = cnn_shape[i] if i < len(cnn_shape) else cnn_shape[-1]
            self.encoder_shape.append((kernel_size, stride))
            conv_block = ConvBlock(kernel_size, stride)
            # linear_layer = nn.Linear(output_size, output_size // compression_factor)
            encoder.add_module(f'conv_block{i}', conv_block)
            output_size //= stride
            i += 1

        print(f"encoder shape (maps kernel size to stride): {self.encoder_shape}")

        # encoder = CNN(shape=self.encoder_shape, stride=compression_factor)
        self.net.add_module('encoder', encoder)



        # add decoder
        self.decoder_shape = self.encoder_shape
        self.decoder_shape.reverse()
        decoder = nn.Sequential()
        i = 0
        for kernel_size, stride in self.decoder_shape:
            conv_block = ConvBlock(kernel_size, stride=1, scale_factor=stride)
            # linear_layer = nn.Linear(output_size, output_size // compression_factor)
            decoder.add_module(f'deconv_block{i}', conv_block)
            i += 1

        print(f"decoder shape: {self.decoder_shape}")
        # decoder = CNN(shape=self.decoder_shape, stride=compression_factor)
        self.net.add_module('decoder', decoder)



        # TRASH:

        # linear1 = nn.Linear(img_size[-1], img_size[-1] // compression_factor)
        # self.net.add_module('linear1', linear1)
        # linear2 = nn.Linear(img_size[-1] // 2, img_size[-1] // 4)
        # self.net.add_module('linear2', linear2)
        #
        # linear3 = nn.Linear(img_size[-1] // 4, img_size[-1] // 2)
        # self.net.add_module('linear3', linear3)
        # linear4 = nn.Linear(img_size[-1] // compression_factor, img_size[-1])
        # self.net.add_module('linear4', linear4)

        # self.decoder_shape = list(self.encoder_shape)
        # self.decoder_shape.reverse()
        # print(f"decoder shape: {self.decoder_shape}")
        # decoder = CNN(shape=self.decoder_shape, stride=1)
        # self.net.add_module('decoder', decoder)
        print(self.net)

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
