#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


# data [(tensor(image/non-image), tensor(P(image)), ... ]
import torch
from numpy.random import shuffle
from PIL import Image
import os
import sys
import random
sys.path.append('/Users/joe/img_gen/src')


@dataclass
class EncoderDecoderSpec:
    cnn_shape: Tuple[int, ...]
    activation: str
    compression_factor: int
    res_weight: float
    embedding_size: int
    n_linear_embedding_layers: int
    n_linear_final_layers: int

@dataclass
class TrainingSpec:
    epochs: int
    batch_size: int
    learning_rate: float
    max_n_epochs_rising_loss: int
    save_best_net: str

@dataclass
class ImageSpec:
    dir_name: str
    n_images: int
    img_dim: int


def batch(data, batch_size):
    n_batches = data.shape[0] // batch_size
    batched_shape = [n_batches, batch_size] + list(data.shape[1:])
    batched_data = data[:n_batches*batch_size].reshape(batched_shape)
    return batched_data
