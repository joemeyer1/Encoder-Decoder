# Copyright (c) 2020 Joseph Meyer


import unittest

import torch

from src.encoder_decoder import EncoderDecoder
from src.train import train_net


class TestEncoderDecoder(unittest.TestCase):

    # def test_train_encoder_decoder_sunsets(self):
    #     from src.data_utils import get_image_data
    #     from src.image_functions import show_image
    #     image_data = get_image_data(n=10)
    #     c = EncoderDecoder(img_size=image_data.shape[-2:], embedding_size=128)
    #     y1 = c.forward(image_data[:1])
    #     y2 = c.forward(image_data[:2])
    #     y3 = c.forward(image_data)
    #     train_net(net=c, data=image_data, epochs=100, batch_size=2, verbose=True, lr=.0001, save_best_net=False)
    #     y = c.forward(image_data[0])
    #     show_image(y)
    #     self.assertTrue(image_data is not None)


    def test_train_encoder_decoder(self):
        x = torch.randn(4, 3, 16, 16) * 100
        c = EncoderDecoder(img_size=x.shape[-2:], embedding_size=2)
        y1 = c.forward(x)
        train_net(net=c, data=x, epochs=10, batch_size=2, verbose=True, lr=.0001, save_best_net=False)
        # from src.batch_data import batch
        y2 = c.forward(x)
        y2_beats_y1 = torch.sum(abs(y2 - x)) < torch.sum(abs(y1 - x))
        self.assertTrue(y2_beats_y1)

    def test_batch_data(self):
        from src.data_utils import batch
        data = torch.randn(6, 3, 16, 16) * 100
        batch_size = 2
        n_batches = data.shape[0] // batch_size
        batched_data_shape_expected = torch.Size([n_batches, batch_size] + list(data.shape[1:]))
        batched_data = batch(data, batch_size=2)
        self.assertTrue(batched_data.shape, batched_data_shape_expected)

    def test_get_and_show_image(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_image
        image_data = get_image_data(n=10)
        show_image(image_data[0])
        self.assertTrue(image_data is not None)

    def test_get_and_show_images(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_images_from_vec
        image_data = get_image_data(n=10)
        show_images_from_vec(image_data[:4])
        self.assertTrue(image_data is not None)


if __name__ == '__main__':
    unittest.main()