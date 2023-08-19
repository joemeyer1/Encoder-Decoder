# Encoder-Decoder
 
I experimented with a variety of methods for generating sunset images. I first tried DeepDream’s idea of running gradient descent directly on input images to maximize their classification as sunsets by a Convolutional Neural Network (CNN), which I pre-trained on sunset photos scraped from the web. (Code can be found here: https://github.com/joemeyer1/Img-Gen ).
This produced some cool output, but each CNN yielded only variations on the same pattern. Even across CNNs, the core visual themes were very similar, given similar training data - the models usually learned to look for / generate a round, neon sun glowing brightly against a dark background. 

[ example DeepDream-ish generated image:
https://www.dropbox.com/s/am6xxq4zsqenuu2/deepdreamish_generated_image.jpg?dl=0 ]

(Code for Deep Dream-ish strategy: https://github.com/joemeyer1/Img-Gen)

Next, I trained an Encoder-Decoder network (also composed of CNNs), the Encoder compressing the sunset features into a vector embedding from which the Decoder then reconstructed the original sunsets. Once trained, I isolated the Decoder and fed it random vector embeddings. This enabled generation of a wide range of distinct images from a single model. 

[ example Encoder-Decoder generated images:
https://www.dropbox.com/s/mqyu9a59wak1rc5/encoder_decoder_generated_img.jpg?dl=0 ,
https://www.dropbox.com/s/tl3y5vcxz815xbw/encoder_decoder_generated_image2.jpg?dl=0 ]

The next step in this project is integrating the Discriminator CNN from the first method with the Decoder/Generator from the second method to yield a Generative Adversarial Network (GAN), which could further improve the quality, variety and complexity of sunset output.
