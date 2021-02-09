#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch


@dataclass
class EncoderDecoderSpec:
    cnn_shape: Tuple[int, ...]
    activation: str
    compression_factor: int
    res_weight: float
    dropout: float
    embedding_size: int
    n_linear_embedding_layers: int
    n_linear_final_layers: int


@dataclass
class TrainingSpec:
    epochs: int
    batch_size: int
    learning_rate: float
    test_proportion: float
    save_best_net: str  # 'min_test_loss' or 'min_train_loss' or best net not saved
    show_image_every_n_epochs: Optional[int] = None
    show_image_every_n_loss_drop: Optional[int] = None
    max_n_epochs_unimproved_loss: Optional[int] = None
    train_until_loss_margin_falls_to: Optional[float] = None
    save_loss_as: str = 'losses/loss'
    delete_data: bool = False

    def check_params(self, n_images: int):
        for param in (self.epochs, self.batch_size, self.learning_rate):
            assert param > 0, f"param {param} must be > 0"
        for param in (self.test_proportion, self.max_n_epochs_unimproved_loss, self.train_until_loss_margin_falls_to):
            assert param >= 0, f"param {param} must be >= 0"
        assert self.test_proportion >= 0, f"param {self.test_proportion} must be >= 0"
        if self.save_best_net:
            for param in (self.max_n_epochs_unimproved_loss, self.train_until_loss_margin_falls_to):
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

def finalize_filename(filename, i=0):
    name, ext = filename.split('.')
    filename = name + str(i) + '.' + ext
    while os.path.exists(filename):
        i += 1
        filename = name + str(i) + '.' + ext
    return filename

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
