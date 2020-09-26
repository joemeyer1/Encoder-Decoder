#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


import unittest

import torch

from src.encoder_decoder import EncoderDecoder
from src.train import train_net


class TestEncoderDecoder(unittest.TestCase):

    # @unittest.skip
    def test_train_encoder_decoder_sunsets(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_image, show_images
        import time
        image_data = get_image_data(n=256, img_size=(256, 256))
        c = EncoderDecoder(img_size=image_data.shape[-2:], embedding_size=128, cnn_shape=(3,3,3,1))
        show_image(image_data[0])
        time.sleep(3)
        y1 = c.forward(image_data[:1])
        show_image(y1[0])
        train_net(net=c, data=image_data, epochs=800, batch_size=4, verbose=True, lr=1e-5, save_best_net=False)
        y = c.forward(image_data[:1])
        show_image(y[0])
        # time.sleep(3)
        # show_images(image_data[:2])
        # embedding1 = c.net[:2](image_data[:1])
        # embedding2 = c.net[:2](image_data[1:2])
        # decoded = c.net[2:](embedding1+embedding2)
        # show_image(decoded[0])
        time.sleep(3)
        rand_y = c.net.encoder.forward(torch.randn(y[:1].shape)*100)
        rand_y = c.net.decoder.forward(rand_y)
        time.sleep(3)
        show_image(rand_y[0])
        self.assertTrue(torch.sum(abs(rand_y1[0] - image_data[0])) > torch.sum(abs(y[0] - image_data[0])))

    @unittest.skip
    def test_train_encoder_decoder_sunset(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_image
        image_data = get_image_data(n=1, img_size=(16, 16))
        c = EncoderDecoder(img_size=image_data.shape[-2:], embedding_size=16)
        show_image(image_data[0])
        y1 = c.forward(image_data[:1])
        show_image(y1[0])
        train_net(net=c, data=image_data, epochs=50, batch_size=1, verbose=True, lr=1e-5, save_best_net=False)
        y = c.forward(image_data[:1])
        show_image(y[0])
        self.assertTrue(image_data is not None)

    @unittest.skip
    def test_train_encoder_decoder(self):
        x = torch.randn(4, 3, 16, 16) * 100
        c = EncoderDecoder(img_size=x.shape[-2:], embedding_size=2)
        y1 = c.forward(x)
        train_net(net=c, data=x, epochs=10, batch_size=2, verbose=True, lr=.0001, save_best_net=False)
        # from src.batch_data import batch
        y2 = c.forward(x)
        y2_beats_y1 = torch.sum(abs(y2 - x)) < torch.sum(abs(y1 - x))
        self.assertTrue(y2_beats_y1)

    @unittest.skip
    def test_batch_data(self):
        from src.data_utils import batch
        data = torch.randn(6, 3, 16, 16) * 100
        batch_size = 2
        n_batches = data.shape[0] // batch_size
        batched_data_shape_expected = torch.Size([n_batches, batch_size] + list(data.shape[1:]))
        batched_data = batch(data, batch_size=2)
        self.assertTrue(batched_data.shape, batched_data_shape_expected)

    @unittest.skip
    def test_get_and_show_image(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_image
        image_data = get_image_data(n=10)
        show_image(image_data[0])
        self.assertTrue(image_data is not None)

    @unittest.skip
    def test_get_and_show_images(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_images
        image_data = get_image_data(n=10)
        show_images(image_data[:4])
        self.assertTrue(image_data is not None)

    @unittest.skip
    def testPass(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
