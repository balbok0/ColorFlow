import networks as net
import h5py
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from keras.backend import clear_session
import os
from methods import *

'''
- WZ is 1, QCD/JZ is 0
- TO DO:
    - Analysis of kernel size, including using vertical kernel instead of square one.
    - Consider LaNET for Herwig Dipole. See whether that wouldn't underfit.
- Done:
    -   Dropout analysis, set to 0.5. Kernel analysis, set to (3, 3).
    -   Statistics for all generators are the same.
    -   Herwig Dipole
    -   Pythia Standard
    -   Model Visualization
'''

drop = 0.5
kernel = (3, 3)
gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
# gen_used = "Herwig Dipole"

model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"


def model_trainer(model_name, generator, dropout=0.5, kernel_size=(3, 3), saving=True):
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
    model = net.get_model(model_name, dropout, kernel_size)
    # model = load_model("models/validated " + model_name + " " + generator)
    model.summary()

    # training
    callback = []
    if saving:
        callback = [ModelCheckpoint(filepath="models/validated " + model_name + " " +
                                             generator,save_best_only=True)]
    history = model.fit(x=xtr, y=ytr, epochs=20,
                        callbacks=callback, validation_data=(xval, yval), shuffle='batch')

    if saving:
        model.save("models/" + model_name + " " + generator)


    if os.path.exists('models_data/' + model_name + "_history_" + generator + ".p"):
        with open('models_data/' + model_name + "_history_" + generator + ".p", 'r') as file_pi:
            previous = pickle.load(file_pi)
            current = combine_dict(previous, history.history)
        with open('models_data/' + model_name + "_history_" + generator + ".p", 'wb') as file_pi:
            pickle.dump(current, file_pi)
    else:
        with open('models_data/' + model_name + "_history_" + generator + ".p", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    clear_session()


model_trainer(model_name, gen_used, drop, kernel)
