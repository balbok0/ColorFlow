import networks as net
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Nadam, Adam
from keras.backend import clear_session
import os
import datetime
from methods import *
from get_fnames import get_toy_names, generators

'''
- WZ is 1, QCD/JZ is 0
'''

models = ['lanet', 'lanet2', 'lanet3']
drops = [0.4, 0.5, 0.6]
kernels = [(3, 3), (11, 11), (20, 5)]
optimizers = []
for lr in [0.0003, 0.003]:
    optimizers.append(Nadam(lr=lr))
    for decay in [0.0005, 0.005, 0.05]:
        optimizers.append(Adam(lr=lr, decay=decay))

f_paths = get_toy_names()


def model_trainer(model_name, generator, dropout, kernel_size, x_tr, x_val, y_tr, y_val, opt='adam',
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
    history = model.fit(x=x_tr, y=y_tr, epochs=1, verbose=0,
                        callbacks=callback, validation_data=(x_val, y_val), shuffle='batch')

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
    clear_session()


for gen in generators:
    # Figures out which path to use, whether it's from usb or 'data/' sub-folder.
    file_path = f_paths[gen]

    # Data loading.
    with h5py.File(file_path, 'r') as h:
        x_tr = h['train/x']
        y_tr = h['train/y']
        x_val = h['val/x']
        y_val = h['val/y']

    # Tests various hyper-parameters.
    for mod in models:
        for drop in drops:
            for kernel in kernels:
                print mod, drop, kernel, gen
                print "NAdam", str(datetime.datetime.now())
                model_trainer(mod, gen, drop, kernel, x_tr, x_val, y_tr, y_val, opt='nadam', saving=False)
                print "Adam", str(datetime.datetime.now())
                model_trainer(mod, gen, drop, kernel, x_tr, x_val, y_tr, y_val, opt='adam', saving=False)
