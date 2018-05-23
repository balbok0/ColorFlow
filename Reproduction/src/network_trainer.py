import os

# Data loaders
import h5py as h5
import psutil
# Model loaders, training specific
from keras.backend import clear_session
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical

import local_vars
from get_file_names import get_ready_path
from network_trainer_helpers import net, save_history, get_memory_size

# based on: https://arxiv.org/abs/1609.00607

model_path = local_vars.models
# generator = 'Pythia Standard'
# generator = 'Pythia Vincia'
# generator = 'Sherpa'
generator = 'Herwig Angular'
# generator = 'Herwig Dipole'


def network_trainer(gen):
    fname = get_ready_path(gen)

    # Data loading
    with h5.File(fname) as hf:
        memory_cost = 122 * 4  # Buffer for creating np array
        memory_cost += get_memory_size(hf['train/x'])
        memory_cost += get_memory_size(hf['val/x'])
        memory_cost += 2 * get_memory_size(hf['train/y'])
        memory_cost += 2 * get_memory_size(hf['val/y'])

    if memory_cost < psutil.virtual_memory()[1]:  # available memory
        with h5.File(fname) as hf:
            x_train = hf['train/x'][()]
            y_train = to_categorical(hf['train/y'], 2)
            x_val = hf['val/x'][()]
            y_val = to_categorical(hf['val/y'], 2)
        print "Using data loaded into memory"
    else:
        x_train = HDF5Matrix(fname, 'train/x')
        y_train = to_categorical(HDF5Matrix(fname, 'train/y'), 2)
        x_val = HDF5Matrix(fname, 'val/x')
        y_val = to_categorical(HDF5Matrix(fname, 'val/y'), 2)
        print "Using data from HDF5Matrix"

    # Model loading
    model = net()
    model.summary()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    calls = [LearningRateScheduler(lambda i: float(0.001*(0.98**i))),
             EarlyStopping(monitor='val_loss', min_delta=0., patience=10, verbose=2, mode='auto'),
             ModelCheckpoint('{0}{1}.h5'.format(model_path, gen), monitor='val_loss', verbose=2,
                             save_best_only=True, mode='auto')]

    hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                     batch_size=100, epochs=100, shuffle='batch', verbose=2, callbacks=calls)

    save_history(hist.history, gen)
    clear_session()


network_trainer(generator)
network_trainer('Herwig Dipole')
