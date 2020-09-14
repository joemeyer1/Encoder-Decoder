

from torch import nn, flatten

class Classifier(nn.Module):
	def __init__(self, size=256):
		super(Classifier, self).__init__()

		self.net = nn.Sequential(
			LinearBlock(3*size**2, 256),
			# LinearBlock(256, 64),
			LinearBlock(256, 256),
			LinearBlock(256, 256),
			# LinearBlock(256, 64),
			# LinearBlock(128, 64),
			# LinearBlock(64,1),
			# LinearBlock(256, 1),
			nn.Linear(256, 1),
			# nn.Sigmoid()
		)

	def forward(self, x):
		x = flatten(x, 1)
		return self.net(x)


class LinearBlock(nn.Module):
	def __init__(self, D_in = 3*256*256, D_out = 256):
		super(LinearBlock, self).__init__()

		self.block = nn.Sequential(
			nn.Dropout(),
			nn.Linear(D_in, D_out),
			nn.ReLU()
		)
		
	def forward(self, x):
		return self.block(x)




