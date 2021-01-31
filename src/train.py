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
        training_spec
):
    verbose = True
    save_best_net = training_spec.save_best_net

    assert len(data) > 0
    assert training_spec.test_proportion < 1

    # train net
    import torch
    from tqdm import tqdm
    from copy import deepcopy
    from src.data_utils import batch

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=training_spec.learning_rate)
    best_net, min_loss = net, float('inf')
    with tqdm(range(training_spec.epochs)) as epoch_counter:
        try:
            n_epochs_unimproved_loss = 0
            n_test_data = int(len(data) * training_spec.test_proportion)
            n_train_data = len(data) - n_test_data
            train_data = data[:n_train_data]
            test_data = data[n_train_data:]
            train_losses = []
            test_losses = []
            for epoch in epoch_counter:
                tot_epoch_train_loss = 0
                batches = batch(train_data, training_spec.batch_size)
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
                train_losses.append(epoch_train_loss)
                # get test loss
                test_output = net(test_data)
                epoch_test_loss = loss_fn(test_output, test_data).item()
                test_losses.append(epoch_test_loss)

                if save_best_net in ('min_test_loss', 'min_train_loss'):
                    if save_best_net == 'min_test_loss':
                        epoch_loss = epoch_test_loss
                        loss_type = 'Test'
                    else:  # save_best_net == 'min_train_loss':
                        epoch_loss = epoch_train_loss
                        loss_type = 'Train'
                    # update best net, min_loss, n_epochs_unimproved_loss
                    loss_improvement = min_loss - epoch_loss
                    if loss_improvement > training_spec.train_until_loss_margin_falls_to:
                        best_net, min_loss = deepcopy(net), deepcopy(epoch_test_loss)
                        n_epochs_unimproved_loss = 0
                    else:
                        n_epochs_unimproved_loss += 1
                    epoch_counter.write(f" Epoch {epoch} Avg {loss_type} Loss: {epoch_loss}\n")
                    # return best net if training's finished
                    if n_epochs_unimproved_loss > training_spec.max_n_epochs_unimproved_loss:
                        if verbose:
                            graph_loss(train_loss=train_losses, test_loss=test_losses, loss_dir=training_spec.save_loss_to_dir)
                        return best_net
                # epoch_counter.desc = "Total Loss: " + str(tot_loss)
        except:
            print("Interrupted.")
            if verbose:
                graph_loss(train_loss=train_losses, test_loss=test_losses, loss_dir=training_spec.save_loss_to_dir)
            if save_best_net:
                return best_net
            else:
                return net
    print('\n')
    if verbose:
        graph_loss(train_loss=train_losses, test_loss=test_losses, loss_dir=training_spec.save_loss_to_dir)
    if save_best_net:
        return best_net
    else:
        return net

def update_best_net(net, epoch_test_loss, min_loss, n_epochs_unimproved_loss):
    if epoch_test_loss < min_loss:
        net, min_loss = deepcopy(net), deepcopy(epoch_test_loss)
        n_epochs_unimproved_loss = 0
    else:
        n_epochs_unimproved_loss += 1
    return {'best_net': net, 'n_epochs_unimproved_loss': n_epochs_unimproved_loss, 'min_loss': min_loss}

def graph_loss(train_loss, test_loss, loss_dir):
    from matplotlib import pyplot as plt
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.legend()
    fig_name = finalize_filename(f'{loss_dir}/loss.png')
    plt.savefig(fig_name)
    import os
    os.system(f"open {fig_name}")

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
