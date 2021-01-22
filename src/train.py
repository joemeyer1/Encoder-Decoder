#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


# from typing import Tuple
# from src.encoder_decoder import EncoderDecoder
# from src.data_utils import fetch_img_data
from src.image_functions import finalize_filename

# def train(img_size: Tuple[int] = (256, 256), embedding_size: int = 64, data_dirname: str = "img_data"):
#     """Train and save an encoder-decoder."""
#
#     # encoder = Encoder(img_size=img_size, embedding_size=embedding_size)
#     # decoder = Decoder(img_size=img_size, embedding_size=embedding_size)
#     encoder_decoder = EncoderDecoder(img_size=img_size, embedding_size=embedding_size)
#     img_data = fetch_img_data(data_dirname)
#     encoder_decoder.train(img_data)
#     save_net(encoder_decoder, "encoder_decoder")

def train_net(net, data, epochs=1000, batch_size=100, verbose=True, lr=.001, save_best_net=True):

    # train net
    import torch
    from tqdm import tqdm
    from copy import deepcopy
    from src.data_utils import batch

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_net, min_loss = None, float('inf')
    with tqdm(range(epochs)) as epoch_counter:
        try:
            tot_loss = 0.
            for epoch in epoch_counter:
                tot_batch_loss = 0
                batches = batch(data, batch_size)
                with tqdm(range(len(batches)), leave=False) as batch_counter:
                    for batch_i in batch_counter:
                        features = batches[batch_i]
                        # prepare for backprop
                        optimizer.zero_grad()
                        # compute prediction
                        # print("getting output")
                        output = net(features)
                        # print("out: {}\n\tlabel: {}\n".format(output, labels))
                        # compute loss
                        # print("Getting loss")
                        loss = loss_fn(output, features)
                        # compute loss gradient
                        # print("getting grad")
                        loss.backward()
                        # print("stepping")
                        # update weights
                        optimizer.step()
                        # report loss
                        # print("reporting")
                        tot_batch_loss += loss.item()
                        if verbose:
                            running_loss = tot_batch_loss / float(batch_i + 1)
                            batch_counter.desc = "Epoch {} Loss: {}".format(epoch,
                                                                            running_loss)  # str(running_loss)
                        # epoch_counter.write("\t Epoch {} Running Loss: {}\n".format(epoch, running_loss))
                    batch_counter.close()
                # report loss
                tot_loss += tot_batch_loss
                avg_loss = tot_loss / ((epoch + 1) * len(batches))
                epoch_loss = tot_batch_loss / float(len(batches))
                # epoch_counter.write("")
                epoch_counter.write(" Epoch {} Avg Loss: {}\n".format(epoch, epoch_loss))
                if save_best_net and epoch_loss < min_loss:
                    best_net, min_loss = deepcopy(net), deepcopy(avg_loss)
                epoch_counter.desc = "Total Loss: " + str(avg_loss)
        except:
            print("Interrupted.")
            if save_best_net:
                return best_net
            else:
                return net
    print('\n')
    if save_best_net:
        return best_net
    else:
        return net


def save_net(net, net_name, i=None):
    import pickle
    net_name = finalize_filename(net_name, i)
    with open(net_name, 'wb') as f:
        pickle.dump(net, f)

def load_net(net_name):
    import pickle
    with open(net_name, 'rb') as f:
        net = pickle.load(f)
    return net
