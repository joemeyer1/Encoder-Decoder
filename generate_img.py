# Copyright (c) 2020 Joseph Meyer


def generate_image(decoder_filename):
    decoder = get_net(decoder_filename)
    rand_img_embedding = torch.randn(decoder.input_size)
    generated_img = decoder(rand_img_embedding)
    return generated_img


if __name__ == '__main__':
    generate_image()

