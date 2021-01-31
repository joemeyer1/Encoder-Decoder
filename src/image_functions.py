#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

from PIL import Image
import torch
import os
import time


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
    return torch.stack(img_vecs)

def show_image(img_vec, img_filename, delete_after=True, i=0):
    img_filename = finalize_filename(img_filename, i)
    save_img(img_vec, img_filename)
    os.system(f"open {img_filename}")
    if delete_after:
        time.sleep(-(-img_vec.shape[0]//256))
        os.system(f"rm {img_filename}")

def save_img(img_vec, img_filename):
    im = get_image(img_vec)
    im.save(img_filename)


def finalize_filename(filename, i=0):
    name, ext = filename.split('.')
    filename = name + str(i) + '.' + ext
    while os.path.exists(filename):
        i += 1
        filename = name + str(i) + '.' + ext
    return filename

def get_image(img_vec):
    img_size = img_vec.shape[-2:]
    img_vec = format_img(img_vec, img_size)
    im = Image.new('RGB', img_size)
    im.putdata(img_vec)
    return im


# helpers for get_image_data

# convert img from network format to Image format
def format_img(img, img_size):
    r, g, b = (ch.int().flatten().tolist() for ch in img)
    return [(r[i], g[i], b[i]) for i in range(img_size[0] * img_size[1])]

# get image vector from .jpg file
def get_image_vec(fname, size=None):

    def get_band_lists(im):
        data = im.split()
        r, g, b = (list(d.getdata()) for d in data)
        return r, g, b

    def get_tensor(r, g, b, size):
        w, h = size
        def convert_to_tensor(rgb):
            r, g, b = (torch.tensor(band, dtype=torch.float) for band in rgb)
            # return torch.stack((r, g, b)
            return torch.cat((r, g, b)).reshape(3, w, h)
        r, g, b = convert_to_tensor((r, g, b))
        rgb = torch.cat((r, g, b)).reshape(3, w, h)
        return rgb

    im = Image.open(fname)#.resize((256,256))
    if not size:
        size = im.size
    im = im.resize(size)
    r, g, b = get_band_lists(im)
    img_vec = get_tensor(r, g, b, size)
    return img_vec



