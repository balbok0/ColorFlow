import os

from keras.backend import clear_session
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical

from get_file_names import get_ready_names as data
from network_trainer_helpers import net, save_history

# based on: https://arxiv.org/abs/1609.00607

model_path = '../models/'

# generator = 'Pythia Standard'
# generator = 'Pythia Vincia'
# generator = 'Sherpa'
# generator = 'Herwig Angular'
generator = 'Herwig Dipole'


def network_trainer(gen):
    fname = data()[gen]
    x_train = HDF5Matrix(fname, 'train/x')
    y_train = to_categorical(HDF5Matrix(fname, 'train/y'), 2)
    x_val = HDF5Matrix(fname, 'val/x')
    y_val = to_categorical(HDF5Matrix(fname, 'val/y'), 2)

    model = net()
    model.summary()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    calls = [LearningRateScheduler(lambda i: float(0.001*(0.98**i))),
             EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=2, mode='auto'),
             ModelCheckpoint('../models/{0}.h5'.format(generator), monitor='val_loss', verbose=2,
                             save_best_only=True, mode='auto')]

    hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                     batch_size=100, epochs=100, shuffle='batch', verbose=2, callbacks=calls)

    save_history(hist.history, gen)
    clear_session()


network_trainer(generator)
