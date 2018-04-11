import networks as net
import h5py
import gc
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from keras.optimizers import Nadam, Adam
from keras.backend import clear_session
import os
import datetime
from methods import *

'''
- WZ is 1, QCD/JZ is 0

- TO DO:
    - TRY:
        - All networks
        - dropout: 0.4 - 0.6
        - kernel: (3, 3), (11, 11), (20, 5)
        - Different ADAM learning rates, decays.
    - np.log scale for images.
    
    
    - Analysis of kernel size, including using vertical kernel instead of square one.
        for all generators.
    - Consider LaNET for Herwig Dipole. See whether that wouldn't underfit.
- Done:
    -   Dropout analysis, set to 0.5. Kernel analysis, set to (3, 3).
    -   Statistics for all generators are the same.
    -   Herwig Dipole
    -   Pythia Standard
    -   Model Visualization
'''
slowPC = True
models = ['lanet', 'lanet2', 'lanet3']
drops = [0.4, 0.5, 0.6]
kernels = [(3, 3), (11, 11), (20, 5)]
optimizers = []
for lr in [0.0003, 0.003]:
    optimizers.append(Nadam(lr=lr))
    for decay in [0.0005, 0.005, 0.05]:
        optimizers.append(Adam(lr=lr, decay=decay))
if slowPC:
    f_paths = get_toy_names()
else:
    f_paths = get_ready_names()


def model_trainer(model_name, generator, dropout, kernel_size, xtr, xval, ytr, yval, opt='adam',
                  saving=True):
    # Model loading.
    # First line option: Create new model. Overwrite last one, if exists.
    # Second line option: Load model trained before.

    model = net.get_model(model_name, dropout, kernel_size)
    # model = load_model("models/validated " + model_name + " " + generator)
    if opt == 'adam':
        op = Adam(lr=0.00025, decay=0.0004)
    else:
        op = Nadam(lr=0.00025)
    model.compile(optimizer=op, loss='binary_crossentropy', metrics=['accuracy'])

    # Callback settings.
    callback = []
    if saving:
        callback = [ModelCheckpoint(filepath="models/validated " + model_name + " " +
                                             generator, save_best_only=True)]

    # training
    history = model.fit(x=xtr, y=ytr, epochs=1, verbose=0,
                        callbacks=callback, validation_data=(xval, yval), shuffle='batch')

    # Saving model. Depends on option in method call.
    if saving:
        model.save("models/" + model_name + " " + generator)

    # Saving history of files.
    history_path = 'toy_models_data/' + model_name + "_history_" + generator + \
                   ' ' + str(kernel_size) + ' ' + str(dropout) + ' ' + opt + ".p"

    with open(history_path, 'w') as file_pi:
        pickle.dump(history.history, file_pi)
        file_pi.close()

    # Free RAM up
    del model, history, callback, opt
    clear_session()
    gc.collect()


for gen in generators:
    # Figures out which path to use, whether it's from usb or 'data/' sub-folder.
    # Creates path to data.h5 file for a generator chosen above.

    file_path = f_paths[gen]
    # Data loading.
    xtr = HDF5Matrix(file_path, 'train/x')
    ytr = HDF5Matrix(file_path, 'train/y')
    xval = HDF5Matrix(file_path, 'val/x')
    yval = HDF5Matrix(file_path, 'val/y')
    for mod in models:
        for drop in drops:
            for kernel in kernels:
                print mod, drop, kernel, gen
                print "NAdam", str(datetime.datetime.now())
                model_trainer(mod, gen, drop, kernel, xtr, xval, ytr, yval, opt='nadam', saving=False)
                print "Adam", str(datetime.datetime.now())
                model_trainer(mod, gen, drop, kernel, xtr, xval, ytr, yval, opt='adam', saving=False)
