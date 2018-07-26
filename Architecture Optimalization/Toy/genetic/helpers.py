import random
from collections import defaultdict

import numpy as np
from keras.activations import softmax, softplus, softsign, elu, relu, selu, linear, tanh, sigmoid, hard_sigmoid
from keras.layers import Layer, Flatten, Activation
from keras.models import Model
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


def get_number_of_weights(model):
    # type: (Model) -> int
    """
    :param model: A model which number of weights is to be determined.
    :return: int - Number of weights in a given model.
    """
    num_weights = 0
    for l in model.get_weights():
        _l = np.ravel(l)
        num_weights += len(_l)
    return num_weights


def multi_roc_score(y_true, y_score):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Returns area under ROC curve value for a multi-class dataset. It's a 1-D scalar value.

    :param y_true: True classes of a dataset. (In one-hot vector form)
    :param y_score: Predicted classes for the same dataset. (Also in one-hot vector form)
    :return: Area under curve, based on interpolated averaged true positive rates and actual false positive rates.
    """

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


def get_memory_size(hdf5_data_set, n_samples=None):
    # type: (HDF5Matrix, int) -> int
    """
    Approximates memory cost of a given number of images in a hdf5 dataset.

    :param hdf5_data_set: Dataset, from which images are used.
    :param n_samples: Number of images in a given dataset which memory cost is to be approximated.
                        If None, number of images equals size of hdf5_data_set.
    :return: Approximated size of whole array in RAM memory.
    """
    # type: (np.ndarray, int) -> int
    if n_samples is None:
        n_samples = len(hdf5_data_set)
    first = hdf5_data_set[0][()]
    return n_samples * first.size * first.itemsize


def __get_masks(x_shape, y, n_train):
    # type: (Tuple[int], np.ndarray, int) -> (np.ndarray, np.ndarray)
    """
    Creates masks, which choose n_train random images after applying a mask.

    :param x_shape: Shape of x dataset (images).
    :param y: True classes corresponding to images of dataset, which shape is given in x_shape.
    :param n_train: Size which dataset should have after applying a mask.
    :return: Two masks, first one for x part of dataset (images), another for y part of dataset (classes)
    """

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

            indexes_x, indexes_y = __get_masks(hf['train/x'].shape, hf['train/y'][()], n_train)

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


def assert_model_arch_match(model, arch):
    # type: (Model, List) -> bool
    """
    Asserts that given architecture and given model match. Specific for THIS Network implementation!

    :param model: A keras neural network model, which architecture is to be compared.
    :param arch: A list containing descriptions of layers to be comapered.
    :return: True, if both arguments match, False otherwise.
    """
    arch_idx = 0
    for l in model.layers[1:-1]:  # type: Layer
        if isinstance(l, (Activation, Flatten)):
            arch_idx -= 1
        else:
            if not arch[arch_idx] in layer_to_arch(l):
                from program_variables.program_params import debug

                if debug:
                    print(arch)
                    print(arch_idx)
                    print(model.layers)
                    print(l.get_config())
                return False
        arch_idx += 1
    return True
