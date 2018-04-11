from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from keras.backend import clear_session
from network_trainer_helpers import net, lr_schedule, save_history
from get_file_names import get_ready_names as data
import os

# based on: https://arxiv.org/abs/1609.00607

# generator = 'Pythia Standard'
# generator = 'Pythia Vincia'
# generator = 'Sherpa'
# generator = 'Herwig Angular'
generator = 'Herwig Dipole'


def network_trainer(gen):
    fname = data()[gen]
    x_train = HDF5Matrix(fname, 'train/x')
    y_train = HDF5Matrix(fname, 'train/y')
    x_val = HDF5Matrix(fname, 'val/x')
    y_val = HDF5Matrix(fname, 'val/y')

    model = net()
    model.summary()

    calls = [LearningRateScheduler(lr_schedule), EarlyStopping(patience=10)]

    hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                     batch_size=100, epochs=20, shuffle='batch', callbacks=calls)

    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save('{path}{g}'.format(path=model_path, g=gen))

    save_history(hist.history, gen)
    clear_session()


network_trainer(generator)
