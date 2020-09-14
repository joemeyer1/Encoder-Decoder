
# data [(tensor(image/non-image), tensor(P(image)), ... ]
import torch
from numpy.random import shuffle
import random
from PIL import Image
import os
import sys
sys.path.append('/Users/joe/img_gen/src')

def get_data(n=1000):
	pos_images = get_pos_images(n)
	neg_images = get_neg_images(n)#//4, 'g') + get_neg_images(n//4, 'b') + get_neg_images(n//4, 'r', 100) + [(torch.randn(3, 256, 256), torch.tensor([0], dtype=torch.float))]*(n//4)
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

def get_pos_images(n, color='r'):
	return get_images(n, color, True)

def get_neg_images(n, color='r'):
	return get_images(n, color, False)

def get_images(n, color='r', pos=True):
	images = []
	for i in range(n):
		hot, cold = get_fns(pos)
		color_map = get_color_map(color, hot, cold)
		bands = []
		for c in color_map:
			bands.append(torch.ones((256, 256), dtype=torch.float)*color_map[c])
		image = (torch.stack(bands), torch.tensor([1], dtype=torch.float)*pos)
		images.append(image)
	return images


# helpers for get_images()

def get_color_map(color, hot=lambda : random.randint(100, 255), cold = lambda hot_val : random.randint(0, hot_val//4)):
	color_map = {'r':0, 'g':0, 'b':0}
	hot_val = hot()
	for c in color_map:
		if c == color:
			color_map[c] = hot_val
		else:
			color_map[c] = cold(hot_val)
	return color_map

def get_fns(pos):
	if pos:
		hot = lambda : random.randint(100, 255)
		cold = lambda hot_val : random.randint(0, hot_val//4)
	else: # neg
		hot=lambda : random.randint(0,255)
		def cold(hot_val):
			if hot_val < 100:
				return random.randint(0,255)
			else:
				return random.randint(hot_val//4+1,255)
	return hot, cold



    

# def img(r,g,b, x=256, y=256):
#     data = img_data(r,g,b, x*y)
#     i = Image.new('RGB', (x, y))
#     i.putdata(data)
# #    i.show()
#     return i

# # img helpers

# def img_data(r,g,b, n):
#     return img_data_helper({'r':r, 'g':g, 'b':b}, n)

# def img_data_helper(color_map={'r':0, 'g':0, 'b':0}, n=256**2, low=0,high=255):
#     colors = ('r', 'g', 'b')
#     unit = tuple([torch.clamp(torch.tensor(color_map[col]), low, high).item() for col in colors])
#     return [unit]*n











