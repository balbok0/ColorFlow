import os
import pickle

from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint

import networks as net
from methods import *

drop = 0.5
kernel = (3, 3)
# gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
gen_used = "Herwig Dipole"

model_type = "SM"
# model_type = "lanet"
# model_type = "lanet2"
# model_type = "lanet3"


def model_trainer(model_name, generator, dropout=0.5, kernel_size=(3, 3), dense_size=128,
                  saving=True):
    # Figures out which path to use, whether it's from usb or 'data/' sub-folder.
    # Creates path to data.h5 file for a generator chosen above.
    file_path = get_ready_names()[generator]

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
        if not os.path.exists('../toy_models'):
            os.makedirs('../toy_models')
        callback = [ModelCheckpoint(filepath="../toy_models/validated " + model_name + " " +
                                             generator + str(dropout), save_best_only=True)]
    history = model.fit(x=xtr, y=ytr, epochs=20, verbose=2, callbacks=callback, validation_data=(xval, yval),
                        shuffle='batch')

    if saving:
        model.save("../toy_models/" + model_name + " " + generator + str(dropout))

    if os.path.exists('../toy_models_data/' + model_name + "_history_" + generator + str(dropout) + ".p"):
        with open('../toy_models_data/' + model_name + "_history_" + generator + str(dropout) + ".p", 'r') as file_pi:
            previous = pickle.load(file_pi)
            current = combine_dict(previous, history.history)
        with open('../toy_models_data/' + model_name + "_history_" + generator + str(dropout) + ".p", 'wb') as file_pi:
            pickle.dump(current, file_pi)
    else:
        if not os.path.exists('../toy_models_data/'):
            os.makedirs('../toy_models_data')
        with open('../toy_models_data/' + model_name + "_history_" + generator + str(dropout) + ".p", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    clear_session()


for d in [0.3, 0.4, 0.6, 0.7]:
    model_trainer(model_type, gen_used, dropout=d)
