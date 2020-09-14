

from cnn import CNN

class Encoder(CNN):
	def __init__(self, shape = [3]*3, stride=1, conv_length=3):
		super().__init__(shape=shape, stride=stride, conv_length=conv_length)
