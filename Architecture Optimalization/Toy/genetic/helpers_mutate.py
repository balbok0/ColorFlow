import random

import numpy as np
from keras.callbacks import LearningRateScheduler
from typing import *

import helpers
from network import Network
from program_variables import program_params as const


def add_layer(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Creates a copy of given Network, but with added layer, randomly derived from given parameters.

    :param base_net: Network, which copy (with added layer) will be returned.
    :param params: Parameters, defining possible choices of layers.
    :return: Copy of given network, with additional layer inserted in a random position.
    """
    layer_idx = random.randint(0, len(base_net.arch))

    possible_layers = {}

    if helpers.get_number_of_weights(base_net.model) > const.max_n_weights or len(base_net.arch) + 1 > const.max_depth:
        remove_layer(base_net, params)

    if layer_idx == 0:
        possible_layers['conv'] = random.choice(params['kernel_size']), random.choice(params['conv_filters'])
    elif layer_idx == len(base_net.arch):
        possible_layers['dense'] = random.choice(params['dense_size'])
    else:
        prev_layer = base_net.arch[layer_idx - 1]
        next_layer = base_net.arch[layer_idx]
        if hasattr(prev_layer, '__getitem__') and \
                (not isinstance(prev_layer, str) or (isinstance(prev_layer, str) and prev_layer.startswith('max'))):
            possible_layers['conv'] = (random.choice(params['kernel_size']), random.choice(params['conv_filters']))
            if not (isinstance(prev_layer, str) or (isinstance(next_layer, str) and next_layer.startswith('max'))):
                possible_layers['max'] = 'max'

        check_if_flat = lambda x: isinstance(x, int) or (isinstance(x, str) and x.startswith('drop'))

        if check_if_flat(next_layer):
            possible_layers['dense'] = random.choice(params['dense_size'])
            if isinstance(next_layer, int) and check_if_flat(prev_layer) \
                    and not (isinstance(prev_layer, str) and prev_layer.startswith('drop')):
                possible_layers['drop'] = 'drop' + str(random.choice(params['dropout']))

    layer_name = random.choice(possible_layers.values())

    new_arch = base_net.arch[:layer_idx] + [layer_name] + base_net.arch[layer_idx:]

    layer_idx += 1  # difference between net.arch and actual architecture. - First activation layer.
    if isinstance(layer_name, int) or (isinstance(layer_name, str) and layer_name.startswith('drop')):
        layer_idx += 1  # difference between net.arch and actual architecture. - Flatten layer.

    return Network(
        architecture=new_arch,
        copy_model=helpers.insert_layer(base_net.model, helpers.arch_to_layer(layer_name, base_net.act), layer_idx),
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def remove_layer(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Creates a copy of given Network, but with removed layer, at a random index.

    :param base_net: Network, which copy (with removed layer) will be returned.
    :param params: Parameters, defining possible choices.
    :return: Copy of given network, with one less layer.
    """
    if len(base_net.arch) <= 2:
        return add_layer(base_net, params)
    layer_idx = random.randint(1, len(base_net.arch) - 2)  # so that, Conv is always first, and Dense is always last.
    layer_name = base_net.arch[layer_idx]
    new_arch = base_net.arch[:layer_idx] + base_net.arch[layer_idx + 1:]

    layer_idx += 1  # difference between net.arch and actual architecture. - First activation layer.
    if isinstance(layer_name, int) or (isinstance(layer_name, str) and layer_name.startswith('drop')):
        layer_idx += 1  # difference between net.arch and actual architecture. - Flatten layer.

    return Network(
        architecture=new_arch,
        copy_model=helpers.remove_layer(base_net.model, layer_idx),
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def change_opt(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Creates a copy of given Network, but with changed optimizer on which it will be trained.

    :param base_net: Network, which copy will be returned.
    :param params: Parameters, defining possible choices of optimizers (and their learning rates).
    :return: Copy of given network, with changed optimizer.
    """
    return Network(
        architecture=base_net.arch,
        copy_model=base_net.model,
        opt=random.choice(params['optimizer']),
        lr=random.choice(params['optimizer_lr']),
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def change_activation(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Creates a copy of given Network, but with changed activation function on each layer specified in architecture.

    :param base_net: Network, which copy will be returned.
    :param params: Parameters, defining possible choices of activation functions.
    :return: Copy of given network, with changed activation functions.
    """
    return Network(
        architecture=base_net.arch,
        copy_model=base_net.model,
        opt=base_net.opt,
        activation=random.choice(params['activation']),
        callbacks=base_net.callbacks
    )


def change_lr_schedule(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Creates a copy of given Network, but with changed callbacks (Learning Rate Scheduler specifically)
    on which it will be trained.

    :param base_net: Network, which copy will be returned.
    :param params: Parameters, defining possible choices of LRS values used.
    :return: Copy of given network, with changed callbacks.
    """
    if random.choice(params['learning_decay_type']) == 'linear':
        def schedule(x):
            return base_net.opt.get_config()['lr'] - float(x * random.choice(params['learning_decay_rate']))
    else:
        def schedule(x):
            return base_net.opt.get_config()['lr'] - float(np.exp(-x * random.choice(params['learning_decay_rate'])))

    return Network(
        architecture=base_net.arch,
        copy_model=base_net.model,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=Network.default_callbacks + [LearningRateScheduler(schedule)]
    )
