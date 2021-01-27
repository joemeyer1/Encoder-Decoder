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

def train_net(
        net,
        data,
        epochs=1000,
        batch_size=100,
        test_proportion=.125,
        verbose=True,
        lr=.001,
        save_best_net='min_test_loss',
        max_n_epochs_rising_loss=10,
):

    assert len(data) > 0
    assert test_proportion < 1

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
            n_epochs_rising_loss = 0
            n_test_data = int(len(data) * test_proportion)
            n_train_data = len(data) - n_test_data
            train_data = data[:n_train_data]
            test_data = data[n_train_data:]
            for epoch in epoch_counter:
                tot_epoch_train_loss = 0
                batches = batch(train_data, batch_size)
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
                        tot_epoch_train_loss += loss.item()
                        if verbose:
                            running_loss = tot_epoch_train_loss / float(batch_i + 1)
                            batch_counter.desc = "Epoch {} Loss: {}".format(epoch, running_loss)  # str(running_loss)
                        # epoch_counter.write("\t Epoch {} Running Loss: {}\n".format(epoch, running_loss))
                    batch_counter.close()
                # report loss
                # tot_loss += tot_epoch_train_loss
                epoch_train_loss = tot_epoch_train_loss / len(batches)
                # get test loss
                test_output = net(test_data)
                epoch_test_loss = loss_fn(test_output, test_data).item()
                if save_best_net == 'min_test_loss':
                    if epoch_test_loss < min_loss:
                        best_net, min_loss = deepcopy(net), deepcopy(epoch_test_loss)
                        n_epochs_rising_loss = 0
                    else:
                        n_epochs_rising_loss += 1
                    epoch_counter.write(" Epoch {} Avg Test Loss: {}\n".format(epoch, epoch_test_loss))
                else:
                    if save_best_net == 'min_train_loss' and epoch_train_loss < min_loss:
                        best_net, min_loss = deepcopy(net), deepcopy(epoch_train_loss)
                        n_epochs_rising_loss = 0
                    else:
                        n_epochs_rising_loss += 1
                    epoch_counter.write(" Epoch {} Avg Train Loss: {}\n".format(epoch, epoch_train_loss))
                if save_best_net and n_epochs_rising_loss > max_n_epochs_rising_loss:
                    return best_net
                # epoch_counter.desc = "Total Loss: " + str(tot_loss)
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
