import h5py
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics
from matplotlib.ticker import MaxNLocator
from keras.utils.io_utils import HDF5Matrix
from keras.models import load_model
from get_fnames import *


# Zero pads images from 25x25 to 33x33x1.
def zero_pad(array):
    assert array.shape == (len(array), 25, 25)
    array = np.pad(array, pad_width=([0, 0], [4, 4], [4, 4]),
                   mode='constant', constant_values=0)
    array = array[..., np.newaxis]
    return array


# Creates roc_curves for all generators, for given model type
def roc_curves(model_name):
    for gen in get_generators():
        roc_curve(model_name, gen)


# Creates roc_curves for given generator for given model type
def roc_curve(model_name, model_generator):
    colors = get_colors()
    files = get_ready_names()
    print model_generator
    model = "models/validated {} {}".format(model_name, model_generator)
    print model
    model = load_model(model)
    for gen, gen_path in files.iteritems():
        print 'Creating curve for {}'.format(gen)
        y_predicted = model.predict(HDF5Matrix(gen_path, 'test/x'), verbose=1)
        y_true = HDF5Matrix(gen_path, 'test/y')
        _single_roc_curve(y_true, y_predicted, colors[gen], gen)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], 'r--', label='Luck')
    plt.title(model_name + " " + model_generator + " ROC Curve")
    plt.legend(loc=4)
    plt.savefig("images/ROC Curve " + model_name + " " + model_generator)
    plt.clf()


# Helper for roc_curves, roc_curve. Allows to put a curve for one model, one data set.
def _single_roc_curve(y_actual, y_pred, color, name):
    y_actual = np.array(y_actual, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    auc = sklearn.metrics.roc_auc_score(y_actual, y_pred)
    tpr, fpr, thr = sklearn.metrics.roc_curve(y_true=y_actual, y_score=y_pred, pos_label=1)
    plt.plot(tpr, fpr, color=color, label=name + ' (AUC = %0.4f)' % auc)


# Takes two dictionaries, with THE SAME keys (Throws error otherwise)
# Returns a dictionary with same keys, and combined values in np array
def combine_dict(dict1, dict2):
    assert type(dict1) is dict and type(dict2) is dict
    assert dict1.keys() == dict2.keys()
    dict_result = {}
    for k in dict1:
        dict_result[k] = np.concatenate((dict1[k], dict2[k]))
    return dict_result


# Creates both loss and accuracy learning curves for validation and training data.
# Requires dict_data to has keys: 'acc', 'loss', 'val_acc', 'val_loss'
def graph_learning_curves(dict_data, name):
    assert dict_data.keys() == ['acc', 'loss', 'val_acc', 'val_loss']
    ax = plt.figure().gca()
    plt.title(name + " Accuracy")
    plt.plot(np.arange(1, len(dict_data['acc']) + 1, dtype=int), dict_data['acc'], color='darkorange',
             label='Training Accuracy')
    plt.plot(np.arange(1, len(dict_data['val_acc']) + 1, dtype=int), dict_data['val_acc'], color='darkgreen',
             label='Validation Accuracy')
    plt.legend(loc=4)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy percentage')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(dict_data['acc']), integer=True))
    plt.savefig('images/learning curves/Learning Curve ' + name + ' - Accuracy.png')
    plt.show()

    ax = plt.figure().gca()
    plt.title(name + " Loss")
    plt.plot(np.arange(1, len(dict_data['loss']) + 1, dtype=int), dict_data['loss'], color='darkorange',
             label='Training Loss')
    plt.plot(np.arange(1, len(dict_data['val_loss']) + 1, dtype=int), dict_data['val_loss'], color='darkgreen',
             label='Validation Loss')
    plt.legend(loc=1)
    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(dict_data['acc']), integer=True))
    plt.savefig('images/learning curves/Learning Curve ' + name + ' - Loss.png')
    plt.show()


# Creates histograms from 2 datasets. Requires both files to have 'auxvars' dictionary,
# and following meta-vars: "pT", "pT Trimmed", "Mass", "Mass Trimmed", "subJet dR", "Tau 1", "Tau 2", "Tau 3"
# order matters.
def make_histograms(fname0, fname1):
    names = ["pT", "pT Trimmed", "Mass", "Mass Trimmed", "subJet dR", "Tau 1", "Tau 2", "Tau 3"]
    with h5py.File(fname0, 'r') as f:
        x = np.array(f['auxvars'])
        x0 = [i for lst in x for i in lst]
        x0 = np.array(x0)
        x0 = np.reshape(x0, [len(x0)/8, 8])

    with h5py.File(fname1, 'r') as f:
        x = np.array(f['auxvars'])
        x1 = [i for lst in x for i in lst]
        x1 = np.array(x1)
        x1 = np.reshape(x1, [len(x1)/8, 8])

    for j in range(8):
        plt.hist(x=x0[:, j], log=False, histtype='step', color='r', bins=50, label="QCD/JZ")
        plt.hist(x=x1[:, j], log=False, histtype='step', color='g', bins=50, label="WZ")
        plt.title(names[j])
        plt.legend(loc='upper right')
        plt.show()
