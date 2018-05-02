import pickle

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from scipy import interp
from sklearn.metrics import roc_curve, roc_auc_score, auc

from get_file_names import *

learn_curve_data_dir = '../models_data/'
learn_curve_img_dir = '../learning_curves/'
roc_img_dir = '../ROC/'
models_dir = '../models/'


# Produces learning curves of given generator
def create_learning_curve(gen):
    assert gen in generators
    assert os.path.exists("{d}history_{g}.p".format(d=learn_curve_data_dir, g=gen))

    data = pickle.load(open("{path}history_{g}.p".format(path=learn_curve_data_dir, g=gen)))

    assert data.keys() == ['acc', 'loss', 'val_acc', 'val_loss']

    print 'Making image for {}'.format(gen)

    if not os.path.exists(learn_curve_img_dir):
        os.makedirs(learn_curve_img_dir)

    # Accuracy
    ax = plt.figure().gca()
    plt.title(gen + " Accuracy")
    plt.plot(np.arange(1, len(data['acc']) + 1, dtype=int), data['acc'], color='darkorange',
             label='Training Accuracy')
    plt.plot(np.arange(1, len(data['val_acc']) + 1, dtype=int), data['val_acc'], color='darkgreen',
             label='Validation Accuracy')

    plt.legend(loc=4)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy percentage')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(data['acc']), integer=True))

    plt.savefig('{d}Learning Curve {g} - Accuracy.png'.format(d=learn_curve_img_dir, g=gen))
    plt.show()

    # Loss
    ax = plt.figure().gca()
    plt.title(gen + " Loss")
    plt.plot(np.arange(1, len(data['loss']) + 1, dtype=int), data['loss'], color='darkorange',
             label='Training Loss')
    plt.plot(np.arange(1, len(data['val_loss']) + 1, dtype=int), data['val_loss'], color='darkgreen',
             label='Validation Loss')

    plt.legend(loc=1)
    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(data['acc']), integer=True))

    plt.savefig('{d}Learning Curve {g} - Loss.png'.format(d=learn_curve_img_dir, g=gen))
    plt.show()


# Helper for roc_curve.
# Given two arrays, returns indexes from array1 with values
# closest to values from array2.
def find_index_nearest(array1, array2):
    res = []
    for i in array2:
        res.append(np.abs(array1-i).argmin())
    return res


# Produces ROC Curves, as defined in paper: https://arxiv.org/abs/1609.00607
def create_roc_curve(gen):
    assert gen in generators
    assert os.path.exists('{path}{g}.h5'.format(path=models_dir, g=gen))

    if not os.path.exists(roc_img_dir):
        os.makedirs(roc_img_dir)

    if load_model('{path}{g}.h5'.format(path=models_dir, g=gen)).output_shape[1] == 1:
        _create_binary_roc(gen)
    else:
        _create_multi_roc(gen)


# ROC helper for binary network
def _create_binary_roc(gen):
    colors = get_colors()
    files = get_ready_names()

    # Needed to create two subplots with different sizes.
    # If other ratios are needed change height_ratios.
    plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    main = plt.subplot(gs[0])
    main.set_yscale('log')
    main.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ratio = plt.subplot(gs[1])
    ratio.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    model = load_model('{path}{g}.h5'.format(path=models_dir, g=gen))

    # Contain true positive rate (signal efficiency) and false positive rate (background efficiency)
    # for each generator.
    tprs = {}
    fprs = {}

    for gen_i, gen_i_path in files.iteritems():
        print 'Creating curve for {}'.format(gen_i)
        with h5.File(gen_i_path) as h:
            y_actual = h['test/y'][()]
        y_pred = np.array(model.predict(HDF5Matrix(gen_i_path, 'test/x'), verbose=1),
                          dtype=np.float64)
        auc_score = roc_auc_score(y_actual, y_pred)
        tpr, fpr, thr = roc_curve(y_true=y_actual, y_score=y_pred, pos_label=1)
        fpr = np.divide(1., fpr)
        fprs[gen_i] = fpr
        tprs[gen_i] = tpr

        main.plot(tpr, fpr, color=colors[gen_i], label='%s (AUC = %0.4f)' % (gen_i, auc_score))

    # fpr of a generator ROC curve is made of.
    div = fprs[gen]
    for gen_i in tprs.keys():
        curr_fpr = fprs[gen_i]
        curr_tpr = tprs[gen_i]
        # find_index_nearest is needed, because roc_curve
        # returns fprs of different length for different generators.
        np.divide(curr_fpr, div[find_index_nearest(fprs[gen], curr_tpr)])
        ratio.plot(tprs[gen_i], fprs[gen_i], color=colors[gen_i])

    ratio.set_xlabel("Signal Positive Rate")
    ratio.set_ylabel("Model / %s" % gen)
    main.set_ylabel("1 / [Background Efficiency]")
    main.set_title("ROC Curve for model trained on {}".format(gen))
    main.legend(loc=1, frameon=False)
    plt.tight_layout()
    plt.savefig("%sROC Curve %s" % (roc_img_dir, gen))
    plt.clf()
    print 'ROC Curve for {} successfully created.'.format(gen)


# ROC helper for multi-class network
def _create_multi_roc(gen):
    model = load_model('{path}{g}.h5'.format(path=models_dir, g=gen))

    colors = get_colors()
    files = get_ready_names()

    # Needed to create two subplots with different sizes.
    # If other ratios are needed change height_ratios.
    plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    main = plt.subplot(gs[0])
    main.set_yscale('log')
    main.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ratio = plt.subplot(gs[1])
    ratio.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Contain true positive rate (signal efficiency), false positive rate (background efficiency) and
    # area under curve (auc) score for each generator.
    tprs = {}
    fprs = {}
    roc_auc = {}

    for gen_i, gen_i_path in files.iteritems():
        print 'Creating curve for {}'.format(gen_i)
        with h5.File(gen_i_path) as h:
            y_actual = to_categorical(h['test/y'][()])
        y_pred = np.array(model.predict(HDF5Matrix(gen_i_path, 'test/x'), verbose=1),
                          dtype=np.float64)

        n_classes = len(y_actual[0])
        gen_fpr = {}
        gen_tpr = {}
        gen_roc_auc = {}
        for i in range(n_classes):
            gen_fpr[i], gen_tpr[i], _ = roc_curve(y_actual[:, i], y_pred[:, i])
            gen_roc_auc[i] = auc(gen_fpr[i], gen_tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([gen_fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, gen_fpr[i], gen_tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fprs[gen_i] = all_fpr
        fprs[gen_i] = np.divide(1., fprs[gen_i])
        tprs[gen_i] = mean_tpr
        roc_auc[gen_i] = auc(fprs[gen_i], tprs[gen_i])
        main.plot(tprs[gen_i], fprs[gen_i], color=colors[gen_i], label='%s (AUC = %0.4f)' % (gen_i, roc_auc[gen_i]))

    # fpr of a generator ROC curve is made of.
    div = fprs[gen]

    for gen_i in tprs.keys():
        curr_fpr = fprs[gen_i]
        curr_tpr = tprs[gen_i]
        # find_index_nearest is needed, because roc_curve
        # returns fprs of different length for different generators.
        np.divide(curr_fpr, div[find_index_nearest(fprs[gen], curr_tpr)])
        ratio.plot(tprs[gen_i], fprs[gen_i], color=colors[gen_i])

    main.plot([0, 1], [0, 1], 'k--', label='Luck')
    ratio.set_xlabel("Signal Positive Rate")
    ratio.set_ylabel("Model / %s" % gen)
    main.set_ylabel("1 / [Background Efficiency]")
    main.set_title("ROC Curve for model trained on {}".format(gen))
    main.legend(loc=1, frameon=False)
    main.xlim([0.0, 1.0])
    main.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig("%sROC Curve %s" % (roc_img_dir, gen))
    plt.clf()
    print 'ROC Curve for {} successfully created.'.format(gen)


create_roc_curve('Sherpa')
