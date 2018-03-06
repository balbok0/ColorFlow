import networks as net
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
import os
from methods import *
from keras.backend import clear_session

drop = 0.5
kernel = (3, 3)
gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
# gen_used = "Herwig Dipole"

# model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"


def model_trainer(model_name, generator, dropout=0.5, kernel_size=(3, 3), dense_size=128,
                  saving=True):
    # Figures out which path to use, whether it's from usb or 'data/' sub-folder.
    # Creates path to data.h5 file for a generator chosen above.
    file_path = get_toy_names()[generator]

    # Data loading.
    xtr = HDF5Matrix(file_path, 'train/x')
    ytr = HDF5Matrix(file_path, 'train/y')
    xval = HDF5Matrix(file_path, 'val/x')
    yval = HDF5Matrix(file_path, 'val/y')

    # Model loading.
    # First line option: Create new model. Overwrite last one, if exists.
    # Second line option: Load model trained before.
    model = net.get_model(model_name, dropout, kernel_size, dense_size=dense_size)
    # model = load_model("models/validated " + model_name + " " + generator)
    model.summary()

    # training
    callback = []
    if saving:
        if not os.path.exists('toy_models'):
            os.makedirs('toy_models')
        callback = [ModelCheckpoint(filepath="toy_models/validated " + model_name + " " +
                                             generator + str(dense_size), save_best_only=True)]
    history = model.fit(x=xtr, y=ytr, epochs=20, verbose=2, callbacks=callback, validation_data=(xval, yval),
                        shuffle='batch')

    if saving:
        model.save("toy_models/" + model_name + " " + generator + str(dense_size))

    if os.path.exists('toy_models_data/' + model_name + "_history_" + generator + ".p"):
        with open('toy_models_data/' + model_name + "_history_" + generator + str(dense_size) + ".p", 'r') as file_pi:
            previous = pickle.load(file_pi)
            current = combine_dict(previous, history.history)
        with open('toy_models_data/' + model_name + "_history_" + generator + str(dense_size) + ".p", 'wb') as file_pi:
            pickle.dump(current, file_pi)
    else:
        if not os.path.exists('toy_models_data/'):
            os.makedirs('toy_models_data')
        with open('toy_models_data/' + model_name + "_history_" + generator + str(dense_size) + ".p", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    clear_session()


for model_name in ['SM', 'lanet']:
    for d in [128, 1024]:
        model_trainer(model_name, gen_used, drop, kernel, dense_size=d)
