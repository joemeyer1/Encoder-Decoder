#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.

import os
os.sys.path.append('/Users/joemeyer/Documents/Encoder-Decoder')
import time

from math import log

import unittest

import torch

from dataclasses import dataclass
from typing import Tuple, Optional

print(os.sys.path)
from src.encoder_decoder import EncoderDecoder
from src.train import train_net, save_net, load_net

from src.data_utils import ImageSpec, EncoderDecoderSpec, TrainingSpec

from src.image_functions import show_image, show_images, get_image_data



class TestEncoderDecoder(unittest.TestCase):

    # @unittest.skip
    # def test_train_encoder_decoder_sunsets(self):
    #
    #     delete_images: bool = False  # indicates to delete generated images
    #     net_to_load: Optional[str] = None  # e.g. "nets/net180.pickle"
    #     i: int = 183  # i indicates minimum number ID to use for file naming
    #
    #     image_spec = ImageSpec(dir_name='img_data', n_images=1024, img_dim=512)
    #
    #     encoder_decoder_spec = EncoderDecoderSpec(
    #         cnn_shape=(5, 3, 3, 1),
    #         activation='tanh',
    #         compression_factor=2,
    #         res_weight=0,
    #         embedding_size=32,
    #         n_linear_embedding_layers=0,
    #         n_linear_final_layers=0,
    #     )
    #
    #     training_spec = TrainingSpec(
    #         epochs=8000,
    #         batch_size=8,
    #         learning_rate=1e-3,
    #         test_proportion=.125,
    #         save_best_net='min_test_loss',
    #         max_n_epochs_unimproved_loss=10,
    #         train_until_loss_margin_falls_to=.1,
    #     )
    #     training_spec.check_params(image_spec.n_images)
    #
    #     # checks if image directories for storing generated images exist - if not, makes them
    #     self.make_image_directories()
    #
    #     image_data = get_image_data(image_spec)
    #
    #     encoder_decoder = load_net(net_to_load) if net_to_load else EncoderDecoder(
    #         encoder_decoder_spec=encoder_decoder_spec,
    #         img_dim=image_spec.img_dim,
    #     )
    #
    #     # show original image
    #     show_image(image_data[0], "original_images/image.jpg", delete_after=delete_images, i=i)
    #     time.sleep(3)
    #
    #     # show untrained encoder-decoder's interpretation of original image
    #     encoded_decoded_image_before_training = encoder_decoder.net.forward(image_data[:1])
    #     show_image(encoded_decoded_image_before_training[0], "encoded_decoded_images_before_training/encoded_decoded_image_before_training.jpg", delete_after=delete_images, i=i)
    #     # show untrained encoder-decoder's interpretation of random img
    #     encoded_decoded_random_img_before_training = encoder_decoder.forward(abs(torch.randn(image_data[:1].shape)) * 128)
    #     show_image(encoded_decoded_random_img_before_training[0], "encoded_decoded_random_imgs_before_training/encoded_decoded_random_img_before_training.jpg", delete_after=delete_images, i=i)
    #
    #     train_net(net=encoder_decoder, data=image_data, training_spec=training_spec)
    #     save_net(encoder_decoder, 'nets/net.pickle', i=i)
    #
    #     # # get trained encoder-decoder's interpretation of sum of 2 original image embeddings
    #     # time.sleep(3)
    #     # show_images(image_data[:2])
    #     # embedding1 = c.net[:2](image_data[:1])
    #     # embedding2 = c.net[:2](image_data[1:2])
    #     # decoded = c.net[2:](embedding1+embedding2)
    #     # show_image(decoded[0])
    #
    #     # get trained encoder-decoder's interpretation of original image
    #     encoded_decoded_image_after_training = encoder_decoder.net.forward(image_data[:1])
    #     show_image(
    #         encoded_decoded_image_after_training[0],
    #         "encoded_decoded_images_after_training/encoded_decoded_image_after_training.jpg",
    #         delete_after=delete_images,
    #         i=i,
    #     )
    #     # get trained encoder-decoder's interpretation of random image
    #     encoded_decoded_random_img = encoder_decoder.forward(abs(torch.randn(encoded_decoded_image_after_training[:1].shape))*128)
    #     show_image(encoded_decoded_random_img[0], "encoded_decoded_random_imgs_after_training/encoded_decoded_random_img_after_training.jpg", delete_after=delete_images, i=i)
    #     # see what random image interpretation looks like if run through encoder-decoder again
    #     deep_encoded_decoded_random_img = encoded_decoded_random_img
    #     for _ in range(1):
    #         deep_encoded_decoded_random_img = encoder_decoder.net.forward(deep_encoded_decoded_random_img[:1])
    #     show_image(deep_encoded_decoded_random_img[0], "double_encoded_decoded_random_imgs_after_training/double_encoded_decoded_random_img_after_training.jpg", delete_after=delete_images, i=i)
    #
    #     # assert that encoder-decoder reconstruction from original image is closer to original image than
    #     #   encoder-decoder reconstruction from random image
    #     self.assertTrue(torch.sum(abs(encoded_decoded_random_img[0] - image_data[0])) > torch.sum(abs(encoded_decoded_image_after_training[0] - image_data[0])))

    # @unittest.skip
    def test_train_encoder_decoder_sunsets_for_optimization(self):

        delete_data: bool = False  # indicates to delete generated images
        net_to_load: Optional[str] = None  # e.g. "nets/net180.pickle"
        i: int = 0  # i indicates minimum number ID to use for file naming

        image_spec = ImageSpec(dir_name='img_data', n_images=64, img_dim=512)

        encoder_decoder_spec = EncoderDecoderSpec(
            cnn_shape=(5, 3, 3, 1),
            activation='relu',
            compression_factor=2,
            res_weight=0,
            dropout=.05,
            embedding_size=16,
            n_linear_embedding_layers=0,
            n_linear_final_layers=0,
        )

        cnn_str = str(encoder_decoder_spec.cnn_shape).replace('(', '').replace(')', '').replace(', ', '_')
        param_filename = f"{cnn_str}_cnn_shape-{encoder_decoder_spec.activation}_activation-" \
                   f"{encoder_decoder_spec.res_weight}_res_weight-{encoder_decoder_spec.embedding_size}_embedding_size-" \
                   f"{encoder_decoder_spec.n_linear_embedding_layers}_n_linear_embedding_layers"

        training_spec = TrainingSpec(
            epochs=8000,
            batch_size=8,
            learning_rate=1e-3,
            test_proportion=.125,
            save_best_net='min_test_loss',
            max_n_epochs_unimproved_loss=10,
            train_until_loss_margin_falls_to=.1,
            save_loss_as=f'losses-optimization/{param_filename}',
            delete_data=delete_data,
        )
        training_spec.check_params(image_spec.n_images)

        # get param filename
        cnn_str = str(encoder_decoder_spec.cnn_shape).replace('(', '').replace(')', '').replace(', ', '_')
        lr_first_digit = str(training_spec.learning_rate).replace('0', '').replace('.', '')[0]
        learning_rate_str = f"{lr_first_digit}e{round(log(training_spec.learning_rate, 10))}"
        param_filename = f"{cnn_str}_cnn_shape--{encoder_decoder_spec.activation}_activation--" \
            f"{encoder_decoder_spec.res_weight}_res_weight--{encoder_decoder_spec.embedding_size}_embedding_size--" \
            f"{encoder_decoder_spec.n_linear_embedding_layers}_n_linear_embedding_layers--" \
            f"{training_spec.batch_size}_batch_size--{learning_rate_str}_lr"
        training_spec.save_loss_as = f'losses-optimization/{param_filename}'

        # checks if image directories for storing generated images exist - if not, makes them
        self.make_image_directories((
            'before/random',
            'before/original_images',
            'after/random',
            'after/reconstructed_images',
            'nets',
            'losses-optimization',
        ))

        image_data = get_image_data(image_spec)

        encoder_decoder = load_net(net_to_load) if net_to_load else EncoderDecoder(
            encoder_decoder_spec=encoder_decoder_spec,
            img_dim=image_spec.img_dim,
        )

        # get original image
        before_img_name = show_image(image_data[0], "before/original_images/" + param_filename + ".jpg", delete_after=delete_data, i=i)

        # show untrained encoder-decoder's interpretation of random img
        encoded_decoded_random_img_before_training = encoder_decoder.forward(abs(torch.randn(image_data[:1].shape)) * 128)
        before_rand_img_name = show_image(encoded_decoded_random_img_before_training[0], "before/random/"+param_filename+'.jpg', delete_after=delete_data, i=i)

        train_net(net=encoder_decoder, data=image_data, training_spec=training_spec)
        if not delete_data:
            save_net(encoder_decoder, f'nets/{param_filename}-net.pickle', i=i)

        # get trained encoder-decoder's interpretation of random image
        encoded_decoded_random_img = encoder_decoder.forward(abs(torch.randn(1, 3, image_spec.img_dim, image_spec.img_dim))*128)
        after_rand_img_name = show_image(encoded_decoded_random_img[0], "after/random/"+param_filename+'.jpg', delete_after=delete_data, i=i)
        
        # get trained encoder-decoder's interpretation of original image
        encoded_decoded_img = encoder_decoder.forward(image_data[:1])
        after_img_name = show_image(encoded_decoded_img[0], "after/reconstructed_images/" + param_filename + '.jpg', delete_after=delete_data, i=i)

        def ensure_img_filenames_match():
            img_name_len = len(param_filename)
            before_rand_name_i = before_rand_img_name[len("before/random/")+img_name_len:]
            after_rand_name_i = after_rand_img_name[len("after/random/")+img_name_len:]
            assert before_rand_name_i == after_rand_name_i, "i does not match before / after"
            before_img_name_i = before_img_name[len("before/original_images/")+img_name_len:]
            after_img_name_i = after_img_name[len("after/reconstructed_images/")+img_name_len:]
            assert before_img_name_i == after_img_name_i
        ensure_img_filenames_match()
        self.assertIsNotNone(encoded_decoded_random_img)

    @unittest.skip
    def test_train_encoder_decoder_sunset(self):
        from src.data_utils import get_image_data
        from src.image_functions import show_image
        image_data = get_image_data(n=1, img_size=(16, 16))
        c = EncoderDecoder(img_size=image_data.shape[-2:], embedding_size=16)
        show_image(image_data[0])
        y1 = c.forward(image_data[:1])
        show_image(y1[0])
        train_net(net=c, data=image_data, epochs=50, batch_size=1, verbose=True, lr=1e-5, save_best_net='min_test_loss')
        y = c.forward(image_data[:1])
        show_image(y[0])
        self.assertTrue(image_data is not None)

    @unittest.skip
    def test_train_encoder_decoder(self):
        x = torch.randn(4, 3, 16, 16) * 100
        c = EncoderDecoder(img_size=x.shape[-2:], embedding_size=2)
        y1 = c.forward(x)
        train_net(net=c, data=x, epochs=10, batch_size=2, verbose=True, lr=.0001, save_best_net='min_test_loss')
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

    @unittest.skip("skip")
    def test_pretrained_autoencoder_from_image(
            self,
            generate_from='randu_image',
            double_process=False,  # whether to run input through network twice
            delete_images=True,
            net_to_load='nets/net90.pickle',
            i=1000,
    ):
        from src.data_utils import get_image_data
        from src.image_functions import show_image, show_images
        import time

        self.make_image_directories()

        encoder_decoder = load_net(net_to_load)
        # assert output_method in ("from_image", "from_random_embedding")
        if generate_from == "image":
            input_image = get_image_data(n=1, img_size=encoder_decoder.img_size)
            output_image = encoder_decoder.forward(input_image[:1])
            show_image(input_image[0], "original_images/image.jpg", delete_after=delete_images, i=i)
        elif generate_from == "randn_image":
            input_image = abs(torch.randn(size=(1, 3)+tuple(encoder_decoder.img_size)))*100
            output_image = encoder_decoder.forward(input_image[:1])
            show_image(input_image[0], "original_images/rand_norm_image.jpg", delete_after=delete_images, i=i)
        elif generate_from == "randu_image":
            input_image = torch.randint(low=0, high=256, size=(1, 3)+tuple(encoder_decoder.img_size), dtype=torch.float32)
            output_image = encoder_decoder.forward(input_image[:1])
            show_image(input_image[0], "original_images/rand_uniform_image.jpg", delete_after=delete_images, i=i)
        else:
            raise Exception("Must specify generate_from='image', 'randn' or 'randu'")
        if double_process:
            output_image = encoder_decoder(output_image)
        time.sleep(3)
        show_image(
            output_image[0],
            f"{generate_from}.jpg",
            delete_after=delete_images,
            i=i,
        )

    @unittest.skip("skip")
    def test_pretrained_autoencoder_from_embedding(
            self,
            generate_from='both',
            brightness=200,
            double_process=True,  # whether to run input through network twice
            delete_images=False,
            net_to_load='nets/net161.pickle',
            i=1000,
    ):
        # from src.data_utils import get_image_data
        from src.image_functions import show_image  # , show_images
        import time

        self.make_image_directories()

        encoder_decoder = load_net(net_to_load)
        # assert output_method in ("from_image", "from_random_embedding")
        randn_input_embedding = lambda: abs(torch.randn(size=(1, 3, encoder_decoder.embedding_size, encoder_decoder.embedding_size))) * brightness
        randu_input_embedding = lambda: torch.randint(low=-brightness, high=brightness, size=(1, 3, encoder_decoder.embedding_size, encoder_decoder.embedding_size), dtype=torch.float32)
        if generate_from == "randn":
            input_embedding = randn_input_embedding()
        elif generate_from == 'randu':
            input_embedding = randu_input_embedding()
        elif generate_from == 'both':
            input_embedding = randn_input_embedding() + randu_input_embedding()
        else:
            raise Exception("Must specify generate_from='randn' or 'randu'")
        output_image = encoder_decoder.net[1:](input_embedding)
        if double_process:
            output_image = encoder_decoder(output_image)
        time.sleep(3)
        show_image(
            output_image[0],
            f"{generate_from}.jpg",
            delete_after=delete_images,
            i=i,
        )

    @unittest.skip("skip")
    def test_random_autoencoder(
            self,

            n_images_train=4,
            img_dim=512,
            embedding_size=16,
            compression_factor=4,
            cnn_shape=(5, 3, 1),
            n_linear_embedding_layers=2,
            n_linear_final_layers=1,
            delete_images=True,
            i=1000,
    ):
        from src.data_utils import get_image_data
        from src.image_functions import show_image, show_images
        import time

        self.make_image_directories()

        image_data = get_image_data(n=n_images_train, img_size=(img_dim, img_dim))
        encoder_decoder = EncoderDecoder(
                img_size=image_data.shape[-2:],
                embedding_size=embedding_size,
                cnn_shape=cnn_shape,
                compression_factor=compression_factor,
                n_linear_embedding_layers=n_linear_embedding_layers,
                n_linear_final_layers=n_linear_final_layers,
            )
        show_image(image_data[0], "original_images/image.jpg", delete_after=delete_images, i=i)
        time.sleep(3)
        encoded_decoded_image_after_training = encoder_decoder.forward(image_data[:1])
        show_image(
            encoded_decoded_image_after_training[0],
            "encoded_decoded_images_after_training/encoded_decoded_image_after_training.jpg",
            delete_after=delete_images,
            i=i,
        )

        encoded_decoded_random_img = encoder_decoder.net.forward(
            torch.randn(encoded_decoded_image_after_training[:1].shape) * 128)
        show_image(
            encoded_decoded_random_img[0],
            "encoded_decoded_random_imgs_after_training/encoded_decoded_random_img_after_training.jpg",
            delete_after=delete_images,
            i=i,
        )

        deep_encoded_decoded_random_img = encoded_decoded_random_img
        for _ in range(1):
            deep_encoded_decoded_random_img = encoder_decoder.net.forward(deep_encoded_decoded_random_img[:1])
        show_image(deep_encoded_decoded_random_img[0],
                   "double_encoded_decoded_random_imgs_after_training/double_encoded_decoded_random_img_after_training.jpg",
                   delete_after=delete_images, i=i)


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

    def make_image_directories(self, image_directories=(
            'original_images',
            'encoded_decoded_images_before_training',
            'encoded_decoded_random_imgs_before_training',
            'encoded_decoded_images_after_training',
            'encoded_decoded_random_imgs_after_training',
            'double_encoded_decoded_random_imgs_after_training',
            'losses',
    )):

        for dir_name in image_directories:
            if not os.path.isdir(dir_name):
                # makedirs allows for more complex paths to be created, versus mkdir
                os.makedirs(dir_name)



if __name__ == '__main__':
    unittest.main()
