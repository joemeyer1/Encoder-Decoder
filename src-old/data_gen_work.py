
# data [(tensor(image/non-image), tensor(P(image)), ... ]
import torch
from numpy.random import shuffle
from PIL import Image
import os
import sys
import random
sys.path.append('/Users/joe/img_gen/src')

def get_data(n=6000):
	print("\tgetting neg data...")
	neg_images = get_neg_images(n)
	print("\tgetting pos data...")
	pos_images = get_pos_images(len(neg_images))
	del neg_images[len(pos_images):]
	shuffle(pos_images)
	shuffle(neg_images)
	images = []
	for i in zip(pos_images, neg_images):
		images += list(i)
	features, labels = [], []
	for im in images:
		feature, label = im
		features.append(feature)
		labels.append(label)

	return torch.stack(features), torch.stack(labels)

# HELPERS for get_data()

def get_image_data(n, dir_name='src/sunsets', label=1):
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
	# return data w pos labels
	return [(img_vec, torch.tensor([label], dtype=torch.float)) for img_vec in img_vecs]

def get_neg_images_rand(n):
	# return rand imgs w neg labels
	return [((torch.rand(3, 256, 256)*255).int().float(), torch.tensor([0], dtype=torch.float)) for _ in range(n)]

def get_neg_images_uniform(n, val=127):
	return [((torch.ones(3, 256, 256)*val).int().float(), torch.tensor([0], dtype=torch.float)) for _ in range(n)]

# def get_neg_images(n):
# 	# return gen'd images w neg labels
# 	images = get_image_data(n, 'generated_images', 0)
# 	return images*max(1, n//len(images))

def get_neg_images(n):
	# return gen'd images w neg labels
	images = get_image_data(n//3, 'generated_images', 0) + get_neg_images_rand(n//3) + get_neg_images_uniform(n//3)
	return images*max(1, n//len(images))


def get_pos_images(n, dir_name='src/sunsets'):
	return get_image_data(n, 'src/sunsets', 1)



# helpers for get_pos_images()

def get_image_vec(fname):
	im = Image.open(fname).resize((256,256))
	r, g, b = get_band_lists(im)
	img_vec = get_tensor(r, g, b)
	return img_vec

# helpers for get_image_vec()
def get_band_lists(im):
	data = im.split()
	r, g, b = (list(d.getdata()) for d in data)
	return r, g, b

def get_tensor(r, g, b):
	r, g, b = torch.tensor(r, dtype=torch.float), torch.tensor(g, dtype=torch.float), torch.tensor(b, dtype=torch.float)
	rgb = torch.cat((r, g, b)).reshape(3, 256, 256)
	return rgb




