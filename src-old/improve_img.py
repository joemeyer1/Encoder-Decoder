
import torch
import os
from PIL import Image
from tqdm import tqdm

def improve(image,
			net,
			epochs,
			lr=10,
			verbose=True,
			show_every=20,
			save_intermediate=False,
			img_fname='temp'):
	# improve image

	image = torch.nn.Parameter(image)
	optimizer = torch.optim.Adam({image}, lr=lr, amsgrad=True)
	loss_fn = torch.nn.MSELoss()
	pos_label = torch.tensor([[1*10000]]*len(image), dtype=torch.float)
	im = None
	with tqdm(range(epochs)) as t:
		for i in t:
			try:
				# prepare for backprop
				optimizer.zero_grad()
				# compute prediction
				output = net(image)
				# compute loss
				loss = loss_fn(output, pos_label)
				# compute loss gradient
				loss.backward()
				# update image
				optimizer.step()
				# report loss
				if verbose:
					t.desc = "Epoch {} Loss: {}".format(i, loss)
					t.refresh()
					if i%show_every == 0:
						if save_intermediate:
							img_fname += str(i)
						im = save_image(image[0], im, img_fname)
			except:
				print("Interrupted.")
				return image
	print('\n')
	return image

def save_image(img_vec, im=None, fname='temp', reopen_file=False):
	fname += '.jpg'
	size = tuple(img_vec[0].shape)
	img_vec = format(img_vec, size = size)
	if not im:
		im = Image.new('RGB', size)
	im.putdata(img_vec)
	# im.show()
	im.save(fname)
	if reopen_file:
		os.system("killall Preview")
		os.system("open {}".format(fname))
	return im

def format(img, size=(256, 256)):
	n = size[0]*size[1]
	r,g,b=(ch.int().flatten().tolist() for ch in img)
	return [(r[i], g[i], b[i]) for i in range(n)]


def test(epochs=20, verbose=False):
	global net
	global image
	image = torch.randn(8)
	print("image: {}".format(image))
	net = lambda x:torch.sigmoid(torch.sum(x))
	# net = torch.nn.Linear(8,1)
	print("net(image): {}".format(net(image)))
	image = improve(image, net, epochs, verbose=verbose)
	print("image: {}".format(image))
	print("net(image): {}".format(net(image)))


def test2(epochs=100, verbose=False):
	import pickle
	# global net
	# global image
	image = torch.randn(1,3,256,256)
	print("image: {}".format(image))
	with open('net.pickle','rb') as f:
		net = pickle.load(f)
	# net = torch.nn.Linear(8,1)
	print("net(image): {}".format(net(image)))
	image = improve(image, net, epochs, verbose=verbose)
	print("image: {}".format(image))
	print("net(image): {}".format(net(image)))
	return image