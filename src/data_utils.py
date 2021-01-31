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

def get_image_data(image_spec):
    dir_name = image_spec.get("dir_name")
    n_images = image_spec.get("n_images")
    img_dim = image_spec.get("img_dim")
    img_size = (img_dim, img_dim)

    fnames = os.listdir(dir_name)
    img_vecs = []
    while n_images and fnames:
        i = random.randint(0, len(fnames)-1)
        fname = fnames.pop(i)
        fpath = os.path.join(dir_name, fname)
        try:
            img_vec = get_image_vec(fpath, img_size)
            img_vecs.append(img_vec)
            n_images -= 1
        except:
            print("{} invalid.".format(fpath))
            # image file invalid
            pass
    # return data w pos labels
    return torch.stack(img_vecs)
    # return [(img_vec, torch.tensor([label], dtype=torch.float)) for img_vec in img_vecs]


# helpers for get_pos_images()

def get_image_vec(fname, img_size=(256,256)):
    im = Image.open(fname).resize(img_size)
    r, g, b = get_band_lists(im)
    img_vec = get_tensor(r, g, b, img_size)
    return img_vec

# helpers for get_image_vec()
def get_band_lists(im):
    data = im.split()
    r, g, b = (list(d.getdata()) for d in data)
    return r, g, b

def get_tensor(r, g, b, img_size):
    w, h = img_size
    r, g, b = torch.tensor(r, dtype=torch.float), torch.tensor(g, dtype=torch.float), torch.tensor(b, dtype=torch.float)
    rgb = torch.cat((r, g, b)).reshape(3, w, h)
    return rgb




