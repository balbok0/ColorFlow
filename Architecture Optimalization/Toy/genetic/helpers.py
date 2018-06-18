import random
from collections import defaultdict

import numpy as np
from keras.utils.io_utils import HDF5Matrix
from scipy import interp
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Union

Array_Type = Union[HDF5Matrix, np.ndarray]


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

    name = dataset_name.lower()

    # Needed for typing.
    (x_train, y_train), (x_val, y_val) = (None, None), (None, None)  # type: Array_Type

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
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

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
                    x_val = hf['val/x'][()][:500] + hf['val/x'][()][150500:151000]
                    y_val = to_categorical(hf['val/y'][:500] + hf['val/y'][150500:151000], len(np.unique(hf['val/y'])))

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
