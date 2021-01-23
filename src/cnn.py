#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

from torch import nn


class CNN(nn.Module):
	def __init__(self, shape = (3, 3, 3, 3, 1), stride=1):
		super(CNN, self).__init__()

		self.net = nn.Sequential()

		for i in range(len(shape)):
			kernel_size = shape[i]
			layer = ConvBlock(kernel_size, stride)
			layer_name = "block"+str(i)
			self.net.add_module(layer_name, layer)

	def forward(self, x):
		return self.net(x)


class ResConvBlock(nn.Module):
	def __init__(self, kernel_size=3, stride=1):
		super().__init__()
		self.block = nn.Sequential(
			ConvBlock(kernel_size, stride),
			ConvBlock(kernel_size, stride),
		)

	def forward(self, x):
		return self.block(x) + x


class ConvBlock(nn.Module):
	def __init__(self, kernel_size=3, stride=1, scale_factor=None):
		# make kernel odd
		if kernel_size % 2 == 0:
			kernel_size += 1
		# calculate padding to retain dim
		padding = (kernel_size - 1) // 2

		super(ConvBlock, self).__init__()

		conv_layer = nn.Conv2d(
			in_channels=3,
			out_channels=3,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
		)

		if scale_factor:
			pool_layer = nn.UpsamplingNearest2d(scale_factor=scale_factor)
		else:
			pool_layer = nn.MaxPool2d(
				kernel_size=kernel_size,
				stride=1,
				padding=padding,
			)

		self.block = nn.Sequential(
			nn.Dropout(),
			conv_layer,
			nn.ReLU(),
			pool_layer,
		)

	def forward(self, x):
		return self.block(x)



# UPSAMPLE: add a sample from normal distrubtion to upsample to avoid uniformity