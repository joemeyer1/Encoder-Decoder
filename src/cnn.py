#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

import torch
from torch import nn


# class CNN(nn.Module):
# 	def __init__(self, shape, stride):
# 		super(CNN, self).__init__()
#
# 		self.net = nn.Sequential()
#
# 		for i in range(len(shape)):
# 			kernel_size = shape[i]
# 			layer = ConvBlock(kernel_size, stride)
# 			layer_name = "block"+str(i)
# 			self.net.add_module(layer_name, layer)
#
# 	def forward(self, x):
# 		return self.net(x)


class ConvBlock(nn.Module):
	def __init__(self, kernel_size=3, stride=1, scale_factor=None, activation_fn=nn.ReLU(), pool=True, dropout=.05):
		# make kernel odd
		if kernel_size % 2 == 0:
			kernel_size += 1
		# calculate padding to retain dim
		padding = (kernel_size - 1) // 2

		self.dropout = dropout

		super(ConvBlock, self).__init__()

		conv_layer = nn.Conv2d(
			in_channels=3,
			out_channels=3,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
		)


		if pool:
			if scale_factor:
				pool_layer = RandomUpsample(scale_factor=scale_factor, weight_rand=1)
			else:
				pool_layer = nn.MaxPool2d(
					kernel_size=kernel_size,
					stride=1,
					padding=padding,
				)
		else:
			pool_layer = nn.Identity()

		self.block = nn.Sequential(
			nn.Dropout(self.dropout),
			conv_layer,
			activation_fn,
			pool_layer,
		)

	def forward(self, x):
		return self.block(x)


class RandomUpsample(nn.UpsamplingNearest2d):
	def __init__(self, **kwargs):
		self.weight_rand = kwargs.pop("weight_rand", 1.)
		super().__init__(**kwargs)

	def forward(self, x):
		y = super().forward(x)
		return y + (torch.randn(y.shape) * self.weight_rand)

# UPSAMPLE: add a sample from normal distribution to upsample to avoid uniformity

class scaled_tanh(nn.Tanh):
	def __init__(self, scale_factor=1):
		self.scale_factor = scale_factor / 2  # to yield range [0, scale_factor]
		super().__init__()

	def forward(self, x):
		return (self.scale_factor * super().forward(x)) + self.scale_factor
