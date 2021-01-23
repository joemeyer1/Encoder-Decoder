#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

from copy import deepcopy

from math import log2
from typing import Tuple

from math import log

from torch import nn, Tensor

from src.cnn import ConvBlock  # CNN, ConvBlock


class EncoderDecoder(nn.Module):
    def __init__(
            self,
            img_size=(256, 256),
            embedding_size=4,
            cnn_shape=None,
            compression_factor=2,
            n_linear_embedding_layers=1,
            n_linear_final_layers=1,
    ):
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
            kernel_size = cnn_shape[i] if i < len(cnn_shape) else max(cnn_shape[-1], stride)
            self.encoder_shape.append((kernel_size, stride))
            conv_block = ConvBlock(kernel_size, stride)
            # linear_layer = nn.Linear(output_size, output_size // compression_factor)
            encoder.add_module(f'conv_block{i}', conv_block)
            output_size //= stride
            i += 1

        print(f"encoder shape (maps kernel size to stride): {self.encoder_shape}")

        # encoder = CNN(shape=self.encoder_shape, stride=compression_factor)
        self.net.add_module('encoder', encoder)

        for i in range(n_linear_embedding_layers):
            linear_layer = nn.Sequential(
                nn.Dropout(),
                Linear2d(res_weight=.2, in_features=output_size**2, out_features=output_size**2),
                nn.ReLU(),
                nn.Dropout(),
                Linear2d(res_weight=.2, in_features=output_size**2, out_features=output_size**2),
            )
            self.net.add_module(f'linear_embedding_layer{i}', linear_layer)


        # add decoder
        self.decoder_shape = self.encoder_shape
        self.decoder_shape.reverse()
        decoder = nn.Sequential()
        i = 0
        for kernel_size, stride in self.decoder_shape:
            conv_block = ConvBlock(kernel_size, stride=1, scale_factor=stride)
            # linear_layer = nn.Linear(output_size, output_size // compression_factor)
            decoder.add_module(f'deconv_block{i}', conv_block)
            output_size *= stride
            i += 1

        print(f"decoder shape: {self.decoder_shape}")
        # decoder = CNN(shape=self.decoder_shape, stride=compression_factor)
        self.net.add_module('decoder', decoder)

        for i in range(n_linear_final_layers):
            linear_layer = nn.Sequential(
                nn.Dropout(),
                nn.Linear(output_size, output_size),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(output_size, output_size),
            )
            self.net.add_module(f'linear_final_layer{i}', linear_layer)

        # linear_output_layer = nn.Sequential(
        #         Flattener(),
        #         nn.Linear(output_size**2, output_size**2),
        #         Deflattener(shape=(output_size, output_size))
        # )

        # below yields  stripes only
        # linear_output_layer = Linear2d(output_size**2, output_size**2)
        # linear_output_layer = LinearBiAxis(output_size, output_size)
        #
        # self.net.add_module('linear_output_layer', linear_output_layer)

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

class Linear2d(nn.Linear):
    def __init__(self, **kwargs):
        self.res_weight = kwargs.pop("res_weight", 0.)
        super().__init__(**kwargs)

    def forward(self, x: Tensor):
        y = super().forward(x.flatten(-2, -1)).reshape(x.shape)
        y += x * self.res_weight
        return y

class LinearBiAxis(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.linear0 = nn.Linear(in_dim, out_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
    def forward(self, x: Tensor):
        return self.linear0(x) + self.linear1(x.transpose(-2, -1))

# class Flattener(nn.Module):
#     def forward(self, x: Tensor):
#         return x.flatten(-2, -1)
#
# class Deflattener(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape
#
#     def forward(self, x: Tensor):
#         return x.reshape(self.shape)