import random
import warnings

import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, Dense, Dropout, Flatten, Layer, MaxPool2D
from keras.models import Sequential, Model
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
        copy_model=_insert_layer(base_net.model, arch_to_layer(layer_name, base_net.act), layer_idx),
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
        copy_model=_remove_layer(base_net.model, layer_idx),
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


def find_first_dense(model):
    # type: (Model) -> (int, int)
    """
    Finds an index of first dense layer in a given model.

    :param model: A model in which index of the first dense layer is to be found.
    :return: Tuple containing two ints:
                    (index of first dense layer in layer count, index of first dense layer in weights count).
                    Returns (None, None) if no dense layer exisits in the model.
    """
    layer_idx = 0
    weight_idx = 0
    for l in model.layers:
        if isinstance(l, Dense):
            break
        layer_idx += 1
        weight_idx += len(l.get_weights())
    else:
        return None, None
    return layer_idx, weight_idx


def _insert_layer(model, layer, index):
    # type: (Model, Layer, int) -> Model
    """
    Immutable. Creates copy of a model, adds a layer at a given index, and returns it.

    :param model: A model, which copy will be modified and returned.
    :param layer: A layer to be inserted.
    :param index: An index at which layer is supposed to be inserted at. > 0 and < len(layers) - 1
    :return: A new model, with specified layer inserted at specified index. Not complied yet.
    """
    # Copy of the whole model. Deep copy, due to safety.
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    result = Sequential()

    # Such a deep copy is needed, so that input_shape is not specified, and layers are not shared.
    for l in model_copy.layers[:index]:
        result.add(__clone_layer(l))
    result.add(layer)
    for l in model_copy.layers[index:]:
        result.add(__clone_layer(l))

    # Needed to clone weights.
    weight_number_before = 0
    for l in model_copy.layers[:index]:
        weight_number_before += len(l.get_weights())
    weight_number_after = weight_number_before + len(layer.get_weights()) + len(model_copy.layers[index].get_weights())

    if isinstance(layer, MaxPool2D):
        # MaxPool changes shape of the output, thus weights will not have the same shape.
        new_weights = model_copy.get_weights()[:weight_number_before]

        _, first_dense = find_first_dense(result)
        new_weights += result.get_weights()[weight_number_before:first_dense + 1]  # New weights, shape changed.
        new_weights += model_copy.get_weights()[first_dense + 1:]  # Back to old shape, since Dense resets it.

    else:
        new_weights = model.get_weights()[:weight_number_before]
        if (index < len(model_copy.layers) and isinstance(model_copy.layers[index], Flatten)) or \
                (index < len(model_copy.layers) - 1 and isinstance(model_copy.layers[index], MaxPool2D) and
                    isinstance(model_copy.layers[index+1], Flatten)):
            _, first_dense = find_first_dense(result)
            new_weights += result.get_weights()[weight_number_before:first_dense + 1]  # New weights, shape changed.
            new_weights += model_copy.get_weights()[first_dense - 1:]  # Back to old shape, since Dense resets it.
        else:
            new_weights += result.get_weights()[weight_number_before:weight_number_after]
            new_weights += model_copy.get_weights()[weight_number_before + len(model_copy.layers[index].get_weights()):]
    result.set_weights(new_weights)
    return result


def _remove_layer(model, index):
    # type: (Model, int) -> Model
    """
    Immutable. Creates copy of a model, removes a layer at a given index, and returns it.

    :param model: A model, which copy will be modified and returned.
    :param index: index of layer to be removed. > 0 < len(layers) - 1
    :return: A new model, with a layer at specified index removed. Not compiled yet.
    """
    # Copy of the whole model. Deep copy, due to safety.
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    layer = model_copy.layers[index]  # type: Layer

    warnings.warn("Since a layer at a chosen index is changing shape of data, model doesn't have to compile.") \
        if isinstance(layer, Flatten) else None

    result = Sequential()

    # Such a deep copy is needed, so that input_shape is not specified, and layers are not shared.
    for l in model_copy.layers[:index]:
        result.add(__clone_layer(l))

    for l in model_copy.layers[(index+1):]:
        result.add(__clone_layer(l))

    # Needed to clone weights.
    weight_number_before = 0
    for l in model_copy.layers[:index]:
        weight_number_before += len(l.get_weights())
    layer_weight_width = len(model_copy.layers[index].get_weights())
    weight_number_after = weight_number_before + layer_weight_width

    if isinstance(layer, MaxPool2D):
        # MaxPool changes shape of the output, thus weights will not have the same shape.
        new_weights = model_copy.get_weights()[:weight_number_before]
        _, first_dense = find_first_dense(result)
        new_weights += result.get_weights()[weight_number_before:first_dense + 1]
        new_weights += model_copy.get_weights()[first_dense + 1:]
    else:
        new_weights = model_copy.get_weights()[:weight_number_before]
        new_weights += result.get_weights()[weight_number_before:weight_number_after - (layer_weight_width / 2)]
        new_weights += model_copy.get_weights()[weight_number_after + (layer_weight_width / 2):]

    result.set_weights(new_weights)
    return result


def __clone_layer(layer):
    # type: (Layer) -> Layer
    """
    Clones a layer, with the exact same configuration as a given one.
        It removes variables not mentioned in configuration. In example:
        input shape, which is given due to adding layer to model.

    :param layer: Layer to be cloned.
    :return: An exact copy of a given layer, only based on configuration (no additional information is used).
    """
    return type(layer).from_config(layer.get_config())


def arch_to_layer(layer, activation):
    # type: (str, str) -> Layer
    """
    Given an architecture layer description, and an activation function, returns a new layer based on them.

    :param layer: Layer description. Specific to THIS Network implementation.
    :param activation: Activation function of a layer to be created.
    :return: A new layer based on given description and activation.
    """
    if hasattr(layer, '__getitem__') and not isinstance(layer, str):
        return Conv2D(filters=layer[1], kernel_size=layer[0], activation=activation, kernel_initializer='he_uniform',
                      padding='same')

    elif isinstance(layer, int):
        return Dense(units=layer, activation=activation)

    elif isinstance(layer, str) and layer.lower() in ['m', 'max', 'maxout', 'maxpool']:
        return MaxPool2D()

    elif isinstance(layer, str) and layer.lower().startswith(('drop', 'dropout')):
        if layer.lower().startswith('dropout'):
            return Dropout(rate=float(layer[7:]))
        else:
            return Dropout(rate=float(layer[4:]))
    else:
        raise BaseException('Illegal form of argument layer. '
                            'Please modify fix the argument, or modify this file, to allow different args.')


def layer_to_arch(layer):
    """
    Given a keras layer returns an architecture layer decription. Specific to THIS Network implementation.

    :param layer: Keras layer, which is to be translated to architecture layer description.
    :return: An architecture layer description based on given layer.
    """
    # type: Layer -> list
    if isinstance(layer, Conv2D):
        return [(layer.get_config()['kernel_size'], layer.get_config()['filters'])]
    elif isinstance(layer, MaxPool2D):
        return ['max']
    elif isinstance(layer, Dropout):
        return ['drop%.2f' % layer.get_config()['rate'], 'dropout%.2f' % layer.get_config()['rate']]
    elif isinstance(layer, Dense):
        return [layer.get_config()['units']]
    else:
        return [None]
