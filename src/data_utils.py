#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


from dataclasses import dataclass
from typing import Tuple

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
    test_proportion: float
    max_n_epochs_rising_loss: int
    save_best_net: str

    def check_params(self, n_images: int):
        for param in (self.epochs, self.batch_size, self.learning_rate):
            assert param > 0, f"param {param} must be > 0"
        for param in (self.test_proportion, self.max_n_epochs_rising_loss):
            assert param >= 0, f"param {param} must be >= 0"
        assert self.batch_size <= n_images, "batch can't be larger than dataset"
        if self.save_best_net not in ("min_test_loss", "min_train_loss"):
            print("Not saving best net - "
                  "to save best net pass 'save_best_net' equal to 'min_test_loss' or 'min_train_loss',"
                  f" not '{self.save_best_net}'")

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
