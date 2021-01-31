#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

from copy import deepcopy

from math import log2
from typing import Tuple

from math import log

from torch import nn, Tensor

from src.cnn import ConvBlock, scaled_tanh  # CNN, ConvBlock


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_decoder_spec, img_dim):


        super(EncoderDecoder, self).__init__()

        self.img_dim = img_dim

        self.embedding_size = encoder_decoder_spec.embedding_size
        self.compression_factor = encoder_decoder_spec.compression_factor
        print(f"image dim: {self.img_dim}")
        print(f"embedding size: {self.embedding_size}")
        print(f"compression factor: {self.compression_factor}")

        self.res_weight = encoder_decoder_spec.res_weight
        self.build_activation_fn(encoder_decoder_spec.activation)

        self.net = nn.Sequential()
        self.build_encoder(encoder_decoder_spec.cnn_shape)
        self.build_linear_block(n_layers=encoder_decoder_spec.n_linear_embedding_layers, linear_block_type='embedding')
        self.build_decoder()
        self.build_linear_block(n_layers=encoder_decoder_spec.n_linear_final_layers, linear_block_type='final')
        print(self.net)

    def forward(self, x: Tensor):
        return self.net(x)

    # Helpers for __init__
    
    def build_activation_fn(self, activation):
        if activation.lower() == 'relu':
            self.conv_activation_fn = nn.ReLU
        elif activation.lower() == 'tanh':
            self.conv_activation_fn = scaled_tanh
        else:
            assert isinstance(activation, nn.Module), \
                "You must pass 'relu', 'tanh', or a valid activation fn for arg 'activation'."
            self.conv_activation_fn = activation
    
    def build_encoder(self, cnn_shape):
        self.encoder_shape = []
        self.current_output_size = self.img_dim
        # encoder compresses image down to embedding size, adding additional layers after if specified by cnn shape
        encoder_length = max(len(cnn_shape), int(log(self.current_output_size / self.embedding_size, self.compression_factor)))
        encoder = nn.Sequential()
        for i in range(encoder_length):
            stride = self.compression_factor if self.current_output_size // self.compression_factor >= self.embedding_size else 1
            kernel_size = cnn_shape[i] if i < len(cnn_shape) else max(cnn_shape[-1], stride)
            self.encoder_shape.append((kernel_size, stride))
            if i + 1 < encoder_length:
                conv_activation = self.conv_activation_fn
            else:
                conv_activation = nn.Identity
            conv_block = ConvBlock(kernel_size=kernel_size, stride=stride, activation_fn=conv_activation)
            # linear_layer = nn.Linear(self.current_output_size, self.current_output_size // self.compression_factor)
            encoder.add_module(f'conv_block{i}', conv_block)
            self.current_output_size //= stride

        print(f"encoder shape (maps kernel size to stride): {self.encoder_shape}")
        # encoder = CNN(shape=self.encoder_shape, stride=self.compression_factor)
        self.net.add_module('encoder', encoder)
    
    def build_decoder(self):
        # build decoder shape to mirror encoder, but w final 1-kernel layer
        self.decoder_shape = self.encoder_shape
        self.decoder_shape.reverse()
        self.decoder_shape += [(1, 1)]
        # build decoder
        decoder = nn.Sequential()
        for i in range(len(self.decoder_shape)):
            kernel_size, stride = self.decoder_shape[i]
            if i + 1 < len(self.decoder_shape):
                conv_activation = self.conv_activation_fn
            else:
                conv_activation = nn.Identity
            conv_block = ConvBlock(kernel_size, stride=1, scale_factor=stride, activation_fn=conv_activation)
            # linear_layer = nn.Linear(self.current_output_size, self.current_output_size // self.compression_factor)
            decoder.add_module(f'deconv_block{i}', conv_block)
            self.current_output_size *= stride

        print(f"decoder shape: {self.decoder_shape}")
        # decoder = CNN(shape=self.decoder_shape, stride=self.compression_factor)
        self.net.add_module('decoder', decoder)
            
    def build_linear_block(self, n_layers, linear_block_type=''):
        for i in range(n_layers):
            linear_layer = nn.Sequential(
                nn.Dropout(),
                Linear2d(res_weight=self.res_weight, in_features=self.current_output_size**2, out_features=self.current_output_size**2),
                nn.ReLU(),
                nn.Dropout(),
                Linear2d(res_weight=self.res_weight, in_features=self.current_output_size**2, out_features=self.current_output_size**2),
            )
            self.net.add_module(f'linear_{linear_block_type}_layer{i}', linear_layer)
    
    # def get_net_shape(self, net_type: str) -> Tuple:
    #
    #     net_shape = []
    #     next_layer_size = self.img_dim
    #     assert next_layer_size >= self.embedding_size
    #     while next_layer_size >= self.embedding_size:
    #         net_shape.append(next_layer_size)
    #         next_layer_size //= 2
    #     if net_type == 'decoder':
    #         net_shape.reverse()
    #     return tuple(net_shape)

class Linear2d(nn.Linear):
    def __init__(self, **kwargs):
        self.res_weight = kwargs.pop("res_weight", 0.)
        super().__init__(**kwargs)

    def forward(self, x: Tensor):
        y = super().forward(x.flatten(-2, -1)).reshape(x.shape)
        y += x * self.res_weight
        return y

class LinearBiAxis(nn.Module):
    def __init__(self, in_dim, out_dim, res_weight=0):
        self.res_weight = res_weight
        super().__init__()
        self.linear0 = nn.Linear(in_dim, out_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
    def forward(self, x: Tensor):
        y = self.linear0(x) + self.linear1(x.transpose(-2, -1))
        y += self.res_weight * x
        return y
