from keras.models import Sequential
from keras.layers import MaxoutDense, Dense
import numpy as np
import pickle
import os

dir_path = '../models_data/'


# Returns a model described by: https://arxiv.org/pdf/1609.00607.pdf
def net():
    model = Sequential()
    model.add(MaxoutDense(256, input_dim=625, nb_feature=5, init='he_uniform'))
    model.add(MaxoutDense(128, nb_feature=5))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


# Takes two dictionaries, with THE SAME keys (Throws error otherwise)
# Returns a dictionary with same keys, and combined values in np array.
def combine_dict(dict1, dict2):
    assert type(dict1) is dict and type(dict2) is dict
    assert dict1.keys() == dict2.keys()
    dict_result = {}
    for k in dict1:
        dict_result[k] = np.concatenate((dict1[k], dict2[k]))
    return dict_result


# Saves history h from model trained on generator gen.
# Overwrites old history, if such exists.
def save_history(h, gen):
    assert type(h) is dict
    assert 'acc' in h
    assert 'val_acc' in h
    assert 'val_loss' in h
    assert 'loss' in h

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_path = "{path}history_{g}.p".format(path=dir_path, g=gen)

    with open(f_path, 'wb') as file_pi:
        pickle.dump(h, file_pi)
