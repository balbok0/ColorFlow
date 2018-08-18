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
        prev_type = helpers.arch_type(prev_layer)
        next_layer = base_net.arch[layer_idx]

        if prev_type in ['conv', 'max']:
            possible_layers['conv'] = (random.choice(params['kernel_size']), random.choice(params['conv_filters']))
            if not prev_type == 'max':
                possible_layers['max'] = 'max'

        check_if_flat = lambda x: helpers.arch_type(x) in ['dense', 'drop']

        if check_if_flat(next_layer):
            possible_layers['dense'] = random.choice(params['dense_size'])
            if check_if_flat(prev_layer) and not prev_type == 'drop':
                possible_layers['drop'] = 'drop' + str(random.choice(params['dropout']))

    layer_name = random.choice(possible_layers.values())

    return _add_layer(base_net, layer_name, layer_idx)


def _add_layer(base_net, layer_name, layer_idx):
    # type: (Network, Union[int, str, Tuple[Tuple[int], int]], int) -> Network

    new_arch = base_net.arch[:layer_idx] + [layer_name] + base_net.arch[layer_idx:]

    layer_idx += 1  # difference between net.arch and actual architecture. - First activation layer.
    if helpers.arch_type(layer_name) in ['dense', 'drop']:
        layer_idx += 1  # difference between net.arch and actual architecture. - Flatten layer.

    if const.debug:
        print('')
        print('_add_layer')
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('')

    return Network(
        architecture=new_arch,
        copy_model=helpers._insert_layer(base_net.model, helpers.arch_to_layer(layer_name, base_net.act), layer_idx),
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
    if helpers.arch_type(layer_name) in ['dense', 'drop']:
        layer_idx += 1  # difference between net.arch and actual architecture. - Flatten layer.

    return Network(
        architecture=new_arch,
        copy_model=helpers._remove_layer(base_net.model, layer_idx),
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


def add_conv_max(base_net, params, conv_num=3):
    # type: (Network, Dict[str, List], int) -> Network
    """
    Adds a sequence of Convolutional layers, followed by MaxPool layer to a copy of a given Network.

    :param base_net: Network, which copy (with added sequence) will be returned.
    :param params: Parameters, defining possible choice for parameters for convolutional layers.
    :param conv_num: Number of convolutional layers in a sequence.
    :return: Copy of given network, with additional sequence inserted in a position of maxpool layer,
                or at the beginning of the model.
    """
    max_idx = [0]
    idx = 1
    for l in base_net.arch:
        if helpers.arch_type(l) == 'max':
            max_idx += [idx]
        idx += 1

    if const.debug:
        print('')
        print('add_conv_max')
        print('max_idx: {}'.format(max_idx))
        print('')

    idx_add = random.choice(max_idx)
    conv_params = (random.choice(params['kernel_size']), random.choice(params['conv_filters']))

    return __add_conv_max(base_net, idx_add, conv_num, conv_params)


def __add_conv_max(base_net, idx, conv_num, conv_params):
    # type: (Network, int, int, Tuple[Tuple[int, int], int]) -> Network

    new_arch = base_net.arch
    new_model = base_net.model

    new_arch = new_arch[:idx] + ['max'] + new_arch[idx:]
    new_model = helpers._insert_layer(new_model, helpers.arch_to_layer('max', activation=base_net.act), idx + 1)

    if const.debug:
        print('')
        print('__add_conv_max: outside for-loop')
        print('Index of adding sequence: %d' % idx)
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('')

    for l in range(conv_num):
        new_arch = new_arch[:idx] + [conv_params] + new_arch[idx:]
        if const.debug:
            print('')
            print('__add_conv_max: inside for-loop')
            print('New arch: {}'.format(new_arch))
            print('')

        new_model = helpers._insert_layer(
            new_model, helpers.arch_to_layer(conv_params, activation=base_net.act), idx + 1
        )

        if const.deep_debug:
            print('')
            print('__add_conv_max: inside for-loop')
            new_model.summary()
            print('')

    return Network(
        architecture=new_arch,
        copy_model=new_model,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def add_dense_drop(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Adds a sequence of Dense layer, followed by Dropout layer to a copy of a given Network.

    :param base_net: Network, which copy (with added sequence) will be returned.
    :param params: Parameters, defining possible choice for parameters for dense and dropout layers.
    :return: Copy of given network, with additional sequence inserted in a position of a random dropout layer,
                or at the beginning of 1D computations in the model.
    """
    drop_idx = [helpers.find_first_dense(base_net.model)[0] - 3]
    idx = 0  # Since Activation layer is always first, and Flatten is before any Dropouts.
    for l in base_net.arch:
        if helpers.arch_type(l) == 'drop':
            drop_idx += [idx]
        idx += 1

    if const.debug:
        print('')
        print('add_drop_dense')
        print('drop_idx: {}'.format(drop_idx))
        print('')

    idx_add = random.choice(drop_idx)
    dense_params = random.choice(params['dense_size'])
    drop_params = 'drop%.2f' % random.choice(params['dropout'])

    return __add_dense_drop(base_net, idx_add, dense_params, drop_params)


def __add_dense_drop(base_net, idx, dense_params, drop_params):
    # type: (Network, int, int, str) -> Network
    new_arch = base_net.arch
    new_model = base_net.model

    if idx >= len(new_arch):
        new_arch = new_arch[:idx] + [drop_params] + new_arch[idx:]
        new_model = helpers._insert_layer(
            new_model,
            helpers.arch_to_layer(drop_params, activation=base_net.act),
            idx + 2
        )
    else:
        new_arch = new_arch[:idx + 1] + [drop_params] + new_arch[idx + 1:]
        new_model = helpers._insert_layer(
            new_model,
            helpers.arch_to_layer(drop_params, activation=base_net.act),
            idx + 3
        )

    if const.debug:
        print('')
        print('add_dense_drop: after adding dropout')
        print('Index of adding sequence: %d' % idx)
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('')

    if idx >= len(base_net.arch):
        new_arch = new_arch[:idx] + [dense_params] + new_arch[idx:]
        new_model = helpers._insert_layer(
            new_model,
            helpers.arch_to_layer(dense_params, activation=base_net.act),
            idx + 2
        )
    else:
        new_arch = new_arch[:idx + 1] + [dense_params] + new_arch[idx + 1:]
        new_model = helpers._insert_layer(
            new_model,
            helpers.arch_to_layer(dense_params, activation=base_net.act),
            idx + 3
        )

    if const.debug:
        print('')
        print('add_dense_drop: at the end')
        print('New arch: {}'.format(new_arch))
        print('')

    return Network(
        architecture=new_arch,
        copy_model=new_model,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def remove_conv_max(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Removes a sequence of Convolution layers, followed by MaxOut layer, in a given Network.\n
    If no such sequence is found, then it adds one, instead of removing it.

    :param base_net: A Network, which copy, with mutations, will be returned.
    :param params: Parameters defining possible choices for add_conv_max, if no MaxOut layers are found.
    :return: A Network, based on base_net, but with a sequence of Conv layers and a MaxOut layer removed.
    """
    max_idx = []
    idx = 0  # Since Activation layer is always first.
    for l in base_net.arch:
        if helpers.arch_type(l) == 'max':
            max_idx += [idx]
        idx += 1

    if not max_idx:
        return add_conv_max(base_net, params)

    if len(max_idx) > 1:
        curr_idx = random.randint(1, len(max_idx))
    else:
        curr_idx = 1

    if const.debug:
        print('')
        print('remove_conv_max')
        print('\tmax_idx: {}'.format(max_idx))
        print('\tcurr_idx: {}'.format(curr_idx - 1))
        print('')

    end = max_idx[curr_idx - 1]

    if curr_idx == 1:
        start = 0
    else:
        start = max_idx[curr_idx - 2]

    return __remove_conv_max(base_net, start, end)


def __remove_conv_max(base_net, idx_start, idx_end):
    # type: (Network, int, int) -> Network

    new_model = base_net.model

    if idx_start == 0:
        new_arch = base_net.arch[:idx_start] + base_net.arch[idx_end + 1:]

    else:
        new_arch = base_net.arch[:idx_start] + base_net.arch[idx_end:]

    if const.debug:
        print('')
        print('__remove_conv_max')
        print('\told arch: {}'.format(base_net.arch))
        print('\tnew arch: {}'.format(new_arch))
        print('\tidx_start: {}'.format(idx_start))
        print('\tidx_end: {}'.format(idx_end))
        print('')

    if idx_start == 0:
        idx_start = -1  # Edge case

    for i in range(idx_start, idx_end):
        if const.debug:
            print('')
            print('__remove_conv_max')
            print('\t layer tb removed: {}'.format(base_net.arch[i + 1]))
            print('')
        new_model = helpers._remove_layer(new_model, idx_start + 1)

    return Network(
        architecture=new_arch,
        copy_model=new_model,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def remove_dense_drop(base_net, params):
    # type: (Network, Dict[str, List]) -> Network
    """
    Removes a sequence of Dense layer, followed by Dropout layer/layers, in a given Network.\n
    If no such sequence is found, then it adds one (Dropout + Dense), instead of removing it.

    :param base_net: A Network, which copy, with mutations, will be returned.
    :param params: Parameters defining possible choices for add_dense_drop, if no Dropout layers are found.
    :return: A Network, based on base_net, but with a sequence of Dense layer and a Dropout layers removed.
    """
    drop_idx = []
    idx = 0  # Since Activation layer is always first, and Flatten is before any Dropouts.
    for l in base_net.arch:
        if helpers.arch_type(l) == 'drop':
            drop_idx += [idx]
        idx += 1

    if not drop_idx:
        return add_dense_drop(base_net, params)

    if len(drop_idx) > 1:
        curr_idx = random.randint(1, len(drop_idx))
    else:
        curr_idx = 1

    drop_arch_idx = drop_idx[curr_idx - 1]

    return __remove_dense_drop(base_net, drop_arch_idx)


def __remove_dense_drop(base_net, drop_idx):
    # type: (Network, int) -> Network

    new_model = base_net.model
    new_arch = base_net.arch

    if const.debug:
        print('')
        print('__remove_dense_drop')
        print('Index of drop layer in arch: {}'.format(drop_idx))
        print('Drop layer: {}'.format(new_arch[drop_idx]))
        print('Layer before: {}'.format(new_arch[drop_idx - 1]))
        print('')

    if helpers.arch_type(base_net.arch[drop_idx - 1]) == 'dense':  # Previous layer is dense.
        if const.debug:
            print('')
            print('remove_dense_drop - 1st path (layer before is dense)')
            print('')
        new_model = helpers._remove_layer(new_model, drop_idx + 2)
        new_model = helpers._remove_layer(new_model, drop_idx + 1)
        new_arch = new_arch[:drop_idx-1] + new_arch[drop_idx+1:]

    elif helpers.arch_type(base_net.arch[drop_idx - 1]) == 'drop':  # Previous layer is dropout.
        if const.debug:
            print('')
            print('remove_dense_drop - 2nd path (layer before is drop)')
            print('')
        new_model = helpers._remove_layer(new_model, drop_idx + 2)
        new_model = helpers._remove_layer(new_model, drop_idx + 1)
        lay_before = 2
        while helpers.arch_type(base_net.arch[drop_idx - lay_before]) in ['dense', 'drop']:
            new_model = helpers._remove_layer(new_model, drop_idx + 2 - lay_before)
            lay_before += 1
            if helpers.arch_type(base_net.arch[drop_idx - lay_before + 1]) == 'dense':
                break
        new_arch = new_arch[:drop_idx - lay_before + 1] + new_arch[drop_idx + 1:]
    else:
        if const.debug:
            print('')
            print('remove_dense_drop - 3rd path (layer before is something else)')
            print('')
        new_model = helpers._remove_layer(new_model, drop_idx + 2)
        new_arch = new_arch[:drop_idx] + new_arch[drop_idx + 1:]

    return Network(
        architecture=new_arch,
        copy_model=new_model,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )
