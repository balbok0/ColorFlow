import random
import warnings
from collections import defaultdict

import keras
import numpy as np
from keras.activations import softmax, softplus, softsign, elu, relu, selu, linear, tanh, sigmoid, hard_sigmoid
from keras.layers import MaxPool2D, Layer, Flatten, Dense, Dropout, Conv2D, Activation
from keras.models import Sequential, Model
from keras.utils.io_utils import HDF5Matrix
from scipy import interp
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Union

Array_Type = Union[HDF5Matrix, np.ndarray]

activations_function_calls = {
    'softmax': softmax,
    'elu': elu,
    'selu': selu,
    'softplus': softplus,
    'softsign': softsign,
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'hard_sigmoid': hard_sigmoid,
    'linear': linear
}


def multi_roc_score(y_true, y_score):
    # type: (np.ndarray, np.ndarray) -> float

    n_classes = len(y_true[0])
    gen_fpr = {}
    gen_tpr = {}
    gen_roc_auc = {}
    for i in range(n_classes):
        gen_fpr[i], gen_tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        gen_roc_auc[i] = auc(gen_fpr[i], gen_tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([gen_fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, gen_fpr[i], gen_tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr = all_fpr
    tpr = mean_tpr
    auc_score = auc(fpr, tpr)

    return auc_score


# Helper method for prepare_data.
# Approximates memory cost of hdf5 dataset.
def get_memory_size(hdf5_data_set, n_samples=None):
    # type: (np.ndarray, int) -> int
    if n_samples is None:
        n_samples = len(hdf5_data_set)
    first = hdf5_data_set[0][()]
    return n_samples * first.size * first.itemsize


def get_masks(x_shape, y, n_train):
    # type: (Tuple[int], np.ndarray, int) -> (np.ndarray, np.ndarray)

    all_indexes = defaultdict(list)  # type: Dict[int, List[int]]
    for i in range(len(y)):
        curr = int(y[i])
        all_indexes[curr].append(i)

    ratios = defaultdict()  # type: Dict[int, float]

    for i, j in all_indexes.items():
        ratios[i] = (len(j) * 1. / len(all_indexes[0]))

    # Ratios split the whole dataset to ratios given class and first class.
    # Part scales these ratios up, so that, 'part' corresponds to size of first class.
    part = n_train * 1. / sum(ratios)

    # Masks of what to keep.
    indexes_x = np.full(shape=x_shape, fill_value=False, dtype=bool)
    indexes_y = np.full(shape=y.shape, fill_value=False, dtype=bool)

    for i in all_indexes.keys():
        for j in random.sample(all_indexes[i], int(part * ratios[i])):
            indexes_y[j] = True
            indexes_x[j, ...] = True

    return indexes_x, indexes_y


def prepare_data(dataset_name='colorflow', first_time=True):
    # type: (str, bool) -> Tuple[Tuple[Array_Type, Array_Type], Tuple[Array_Type, Array_Type]]
    """
    Prepares a dataset of a choice, and returns it in form of pair of tuples, containg training and validation datasets.

    :param dataset_name: Name of the dataset, valid arguments are:
            - cifar10   - 'cifar' or 'cifar10'
            - mnist     - 'mnist'
            - colorflow - 'colorflow', 'this', 'project'
            - testing   - 'testing' - a smaller colorflow dataset, for debug purposes.
    :param first_time: Whether a validation dataset should be returned too, or not.
                        If called for the first time, should be returned. If not, can be avoided for better performence.
    :return: (x_train, y_train), (x_val, y_val),
                each being of type np.ndarray, or HDF5Matrix, depending on memory space.
                x_train - is a input to nn, on which neural network can be trained.
                y_train - are actual results, which compared to output of nn, allow it to learn information about data.
                x_val   - is a input to nn, on which nn can be checked how well it performs.
                y_val   - are actual results, agaist which nn can be checked how well it performs.
    """

    name = dataset_name.lower()

    # Needed for typing.
    (_, _), (x_val, y_val) = (None, None), (None, None)  # type: Array_Type

    if name in ['cifar', 'cifar10']:
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical

        (x_train, y_train), (x_val, y_val) = cifar10.load_data()
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

    elif name == 'mnist':
        from keras.datasets import mnist
        from keras.utils.np_utils import to_categorical

        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = np.reshape(x_train, list(np.array(x_train).shape) + [1])
        x_val = np.reshape(x_val, list(np.array(x_val).shape) + [1])
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

    elif name == 'testing':
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical

        (x_train, y_train), (x_val, y_val) = cifar10.load_data()
        y_train = to_categorical(y_train[:2500])
        y_val = to_categorical(y_val[:2500])
        x_train = x_train[:2500, ...]
        x_val = x_val[:2500, ...]

    elif name in ['project', 'this', 'colorflow']:
        from get_file_names import get_ready_path

        import psutil
        import h5py as h5

        from keras.utils.io_utils import HDF5Matrix
        from keras.utils.np_utils import to_categorical

        fname = get_ready_path('Herwig Dipole')

        # Data loading
        with h5.File(fname) as hf:
            # Cap of training images (approximately).
            n_train = 500

            memory_cost = 122 * 4  # Buffer for creating np array
            memory_cost += get_memory_size(hf['train/x'], n_train)
            memory_cost += get_memory_size(hf['val/x'])
            memory_cost += 2 * get_memory_size(hf['val/y'])
            memory_cost += 2 * get_memory_size(hf['train/y'], n_train)

            indexes_x, indexes_y = get_masks(hf['train/x'].shape, hf['train/y'][()], n_train)

        if memory_cost < psutil.virtual_memory()[1]:  # available memory
            with h5.File(fname) as hf:
                x_sing_shape = list(hf['train/x'][0].shape)
                x_train = hf['train/x'][indexes_x]
                x_train = np.reshape(x_train, [int(len(x_train) / np.prod(x_sing_shape))] + x_sing_shape)

                y_train = to_categorical(hf['train/y'][indexes_y], len(np.unique(hf['train/y'])))
                if first_time:
                    x_val = hf['val/x'][()]
                    y_val = to_categorical(hf['val/y'], len(np.unique(hf['val/y'])))[()]

        else:  # data too big for memory.
            x_sing_shape = list(hf['train/x'][0].shape)
            x_train = HDF5Matrix(fname, 'train/x')[indexes_x]
            x_train = np.reshape(x_train, [int(len(x_train) / np.prod(x_sing_shape))] + x_sing_shape)
            y_train = to_categorical(HDF5Matrix(fname, 'train/y')[indexes_y], len(np.unique(hf['train/y'])))
            if first_time:
                x_val = HDF5Matrix(fname, 'val/x')
                y_val = to_categorical(HDF5Matrix(fname, 'val/y'), len(np.unique(hf['val/y'])))

    else:
        raise AttributeError('Invalid name of dataset.')

    return (x_train, y_train), (x_val, y_val)


def arch_to_layer(layer, activation):
    # type: (str, str) -> Layer
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
    # type: (Layer) -> list
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


def assert_model_arch_match(model, arch):
    # type: (Model, List) -> bool
    arch_idx = 0
    for l in model.layers[1:-1]:  # type: Layer
        if isinstance(l, (Activation, Flatten)):
            arch_idx -= 1
        else:
            if not arch[arch_idx] in layer_to_arch(l):
                from local_vars import debug

                if debug:
                    print(arch)
                    print(arch_idx)
                    print(model.layers)
                    print(l.get_config())
                return False
        arch_idx += 1
    return True


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


def insert_layer(model, layer, index):
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
            new_weights += model_copy.get_weights()[first_dense - 1:] # Back to old shape, since Dense resets it.
        else:
            new_weights += result.get_weights()[weight_number_before:weight_number_after]
            new_weights += model_copy.get_weights()[weight_number_before + len(model_copy.layers[index].get_weights()):]
    result.set_weights(new_weights)
    return result


def remove_layer(model, index):
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
