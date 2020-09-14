
# data [(tensor(image/non-image), tensor(P(image)), ... ]
import torch
from numpy.random import shuffle
from PIL import Image
import os
import sys
import random
sys.path.append('/Users/joe/img_gen/src')

def get_data(n=6000, img_size=(256,256)):
	print("\tgetting neg data...")
	neg_images = get_neg_images(n//2, img_size)
	print("\tgetting pos data...")
	pos_images = get_pos_images(len(neg_images), img_size=img_size)
	del neg_images[len(pos_images):]
	return torch.stack(pos_images), torch.stack(neg_images)

	images = neg_images+pos_images
	# mix up images
	indices = [i for i in range(len(images))]
	shuffle(indices)
	features, labels = [], []
	for j in range(len(indices)):
		i = indices[j]
		feature, label = images[i]
		features.append(feature)
		labels.append(label)

	return torch.stack(features), torch.stack(labels)

# HELPERS for get_data()


def get_neg_images(n, img_size=(256,256)):
	# return gen'd images w neg labels
	# images = get_image_data(n//3, 'generated_images', 0, img_size) + get_neg_images_rand(n//3, img_size) + get_neg_images_uniform(n//3, size=img_size)
	images = get_neg_images_rand(n//2, img_size) + get_neg_images_uniform(n//2, size=img_size)
	return images*max(1, n//len(images))


def get_pos_images(n, dir_name='img_data', img_size=(256,256)):
	return get_image_data(n, dir_name, 1, img_size)



def get_image_data(n, dir_name='img_data', label=1, img_size=(256,256)):
	fnames = os.listdir(dir_name)
	img_vecs = []
	while n and fnames:
		i = random.randint(0, len(fnames)-1)
		fname = fnames.pop(i)
		fpath = os.path.join(dir_name, fname)
		try:
			img_vec = get_image_vec(fpath, img_size)
			img_vecs.append(img_vec)
			n -= 1
		except:
			print("{} invalid.".format(fpath))
			# image file invalid
			pass
	# return data w pos labels
	return img_vecs
	# return [(img_vec, torch.tensor([label], dtype=torch.float)) for img_vec in img_vecs]

def get_neg_images_rand(n, size):
	# return rand imgs w neg labels
	w, h = size
	return [(torch.rand(3, w, h)*255).int().float() for _ in range(n)]
	# return [((torch.rand(3, w, h)*255).int().float(), torch.tensor([0], dtype=torch.float)) for _ in range(n)]

def get_neg_images_uniform(n, val=127, size=(256,256)):
	w, h = size
	return [(torch.ones(3, w, h)*val).int().float() for _ in range(n)]
	# return [((torch.ones(3, w, h)*val).int().float(), torch.tensor([0], dtype=torch.float)) for _ in range(n)]

# def get_neg_images(n):
# 	# return gen'd images w neg labels
# 	images = get_image_data(n, 'generated_images', 0)
# 	return images*max(1, n//len(images))

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




