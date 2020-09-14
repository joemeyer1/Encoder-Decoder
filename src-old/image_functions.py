
from PIL import Image
import torch


def show_images(n=float('inf'), dir_name='recent'):
    img_vecs = get_image_vecs(n, dir_name)
    for img_vec in img_vecs:
        show_image(img_vec,tuple(img_vec.shape[-2:]))

def save_img(img_vec, img_filename):
    im = show_image(img_vec)
    im.save(img_filename)

def show_image(img_vec, img_size = (256, 256)):
    if img_size == 'natural':
        pass
    img_vec = format_img(img_vec, img_size)
    im = Image.new('RGB', img_size)
    im.putdata(img_vec)
    im.show()
    return im



# get image vectors
def uniform(n=1, val=127):
    return torch.ones(n, 3, 256, 256)*val

def random_im(n=1):
    return torch.rand(n,3,256,256)*255



# get img vectors from directory of '.jpg's
def get_image_vecs(n=float('inf'), dir_name='recent'):
    import os
    fnames = os.listdir(dir_name)
    img_vecs = []
    i = 0
    while n and i < len(fnames):
        fname = fnames[i]
        fpath = os.path.join(dir_name, fname)
        try:
            img_vec = get_image_vec(fpath)
            img_vecs.append(img_vec)
            n -= 1
        except:
            print("{} invalid.".format(fpath))
            # image file invalid
            pass
        i += 1
    print([img_vec.shape for img_vec in img_vecs])
    return torch.stack(img_vecs)


# HELPERS

# convert img from network format to Image format
def format_img(img, img_size=(256, 256)):
    n = img_size[0]*img_size[1]
    r,g,b=(ch.int().flatten().tolist() for ch in img)
    return [(r[i], g[i], b[i]) for i in range(n**2)]



# helpers for get_image_vecs()

# get image vector from .jpg file
def get_image_vec(fname, size=None):
    im = Image.open(fname)#.resize((256,256))
    if not size:
        size = im.size
    im = im.resize(size)
    r, g, b = get_band_lists(im)
    img_vec = get_tensor(r, g, b, size)
    return img_vec

# helpers for get_image_vec()

def get_band_lists(im):
    data = im.split()
    r, g, b = (list(d.getdata()) for d in data)
    return r, g, b

def get_tensor(r, g, b, size):
    w, h = size
    r, g, b = torch.tensor(r, dtype=torch.float), torch.tensor(g, dtype=torch.float), torch.tensor(b, dtype=torch.float)
    rgb = torch.cat((r, g, b)).reshape(3, w, h)
    return rgb


