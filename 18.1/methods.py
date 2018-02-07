import h5py
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics


# Show an average image of array from file fname.
# Shows an image with output_name title, and saves it as output_name.
def avg_img(fname, output_name):
    with h5py.File(fname, 'r') as f:
        data = np.array(f['images'][()])
    fig = plt.imshow(np.log(np.mean(data, axis=0)))
    plt.xlabel("Prop. to pseudorapidity")
    plt.ylabel("Prop. to translated azimuthal angle")
    plt.title(output_name)
    plt.colorbar(fig)
    plt.savefig("images/average " + output_name)
    plt.show()


# Zero pads images from 25x25x1 to 33x33x1
def zero_pad(array):
    assert array.shape[1:] == (25, 25, 1)
    array = np.concatenate((np.zeros(([len(array), 25, 4, 1])), array), axis=2)
    array = np.concatenate((array, np.zeros(([len(array), 25, 4, 1]))), axis=2)

    array = np.concatenate((np.zeros(([len(array), 4, 33, 1])), array), axis=1)
    array = np.concatenate((array, np.zeros(([len(array), 4, 33, 1]))), axis=1)
    return array


def roc_curve(y_pred, y_actual, generator_name, model_name, color):
    auc = sklearn.metrics.roc_auc_score(y_actual, y_pred)
    print auc
    tpr, fpr, thr = sklearn.metrics.roc_curve(y_true=y_actual, y_score=y_pred, pos_label=1)
    plt.plot(tpr, fpr, color=color, label=generator_name + ' (AUC = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title(model_name + " ROC Curve")
    # plt.savefig("images/" + model_name + " " + generator_name + " ROC Curve")
    # plt.show()


# Takes two dictionaries, with THE SAME keys (Throws error otherwise)
# Returns a dictionary with same keys, and combined values in np array
def combine_dict(dict1, dict2):
    assert type(dict1) is dict and type(dict2) is dict
    assert dict1.keys() == dict2.keys()
    dict_result = {}
    for k in dict1:
        dict_result[k] = np.concatenate((dict1[k],dict2[k]))
    return dict_result
