# Copyright (c) 2020 Joseph Meyer


import unittest
from src.encoder_decoder import EncoderDecoder
from src.train import train_net

class TestEncoderDecoder(unittest.TestCase):

    def test_encoder_decoder(self):
        import torch
        x = torch.randn(2, 3, 16, 16)*100
        c = EncoderDecoder(img_size=x.shape[-2:], embedding_size=2)
        y1 = c.forward(x)
        train_net(net=c, data=x, epochs=2000, batch_size=5, verbose=True, lr=.0001, save_best_net=False)
        y2 = c.forward(x)
        y2_beats_y1 = torch.sum(abs(y2 - x)) < torch.sum(abs(y1 - x))
        self.assertTrue(y2_beats_y1)

if __name__ == '__main__':
    unittest.main()