#!/usr/bin/env python3
# Copyright (c) 2020 Joseph Meyer. All Rights Reserved.


# from typing import Tuple
# from src.encoder_decoder import EncoderDecoder
# from src.data_utils import fetch_img_data
from src.data_utils import finalize_filename, get_rand_img_embedding
from src.image_functions import save_img_with_finalized_filename


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

            test_output = net(test_data)
            pretraining_test_loss = loss_fn(test_output, test_data).item()
            epoch_counter.write(f" Pre-Training Avg Test Loss: {pretraining_test_loss}\n")

            for epoch in epoch_counter:
                tot_epoch_train_loss = 0
                batches = batch(train_data, training_spec.batch_size)
                with tqdm(range(len(batches)), leave=False) as batch_counter:
                    for batch_i in batch_counter:
                        features = batches[batch_i]
                        optimizer.zero_grad()
                        output = net(features)
                        loss = loss_fn(output, features)
                        loss.backward()
                        optimizer.step()
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

                if training_spec.show_image_every_n_epochs and epoch % training_spec.show_image_every_n_epochs == 0:
                    rand_img_embedding = get_rand_img_embedding(net.embedding_size)
                    generated_img = net.net[1:](rand_img_embedding)[0]
                    img_epoch_filename = f"net-training-epochs/{epoch}_epoch_net_training.jpg"
                    save_img_with_finalized_filename(generated_img, img_epoch_filename)

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
                            graph_loss(train_loss=train_losses, test_loss=test_losses, training_spec=training_spec)
                        return best_net
                # epoch_counter.desc = "Total Loss: " + str(tot_loss)
        except:
            print("Interrupted.")
            if verbose:
                graph_loss(train_loss=train_losses, test_loss=test_losses, training_spec=training_spec)
            if save_best_net:
                return best_net
            else:
                return net
    print('\n')
    if verbose:
        graph_loss(train_loss=train_losses, test_loss=test_losses, training_spec=training_spec)
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

def graph_loss(train_loss, test_loss, training_spec):
    from matplotlib import pyplot as plt
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.legend()
    fig_name = finalize_filename(f'{training_spec.save_loss_as}.png')
    plt.savefig(fig_name)
    import os
    os.system(f"open {fig_name}")
    if training_spec.delete_data:
        import time
        time.sleep(1)
        os.system(f"rm {fig_name}")

def save_net(net, net_name, i=None) -> str:
    import pickle
    net_name = finalize_filename(net_name, i)
    with open(net_name, 'wb') as f:
        pickle.dump(net, f)
    return net_name

def load_net(net_name):
    import pickle
    with open(net_name, 'rb') as f:
        net = pickle.load(f)
    return net
