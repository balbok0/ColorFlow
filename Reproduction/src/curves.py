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
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score, auc

from get_file_names import *

learn_curve_data_dir = local_vars.models_data
learn_curve_img_dir = local_vars.learning_curves
roc_img_dir = local_vars.roc
models_dir = local_vars.models
roc_data_dir = local_vars.roc_data


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Produces learning curves of given generator
def create_learning_curve(gen):
    assert gen in generators
    assert os.path.exists("{d}history_{g}.p".format(d=learn_curve_data_dir, g=gen))

    data = pickle.load(open("{path}history_{g}.p".format(path=learn_curve_data_dir, g=gen)))

    assert data.keys() == ['acc', 'loss', 'val_acc', 'val_loss']

    print 'Making image for {}'.format(gen)

    check_dir(learn_curve_img_dir)

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
def create_roc_curve(gen, verbose=2):
    assert gen in generators
    assert os.path.exists('{path}{g}.h5'.format(path=models_dir, g=gen))

    check_dir(roc_img_dir)

    if not os.path.exists('{directory}{g}.h5'.format(directory=roc_data_dir, g=gen)):
        save_tpr_fpr_auc(gen, verbose=verbose)

    __draw_roc(gen)


# ROC curve drawer, given generator.
def __draw_roc(gen):
    line_styles = get_line_style()
    colors = get_colors()

    fprs = {}
    tprs = {}
    aucs = {}
    ratios = {}

    with h5.File('{directory}{g}.h5'.format(directory=roc_data_dir, g=gen)) as h:
        # Contain true positive rate (signal efficiency), false positive rate (background efficiency) and
        # area under curve (auc) score for each generator.
        for g in generators:
            fprs[g] = h['%s/fpr' % g][:]
            tprs[g] = h['%s/tpr' % g][:]
            aucs[g] = h['%s/auc' % g][:]
            ratios[g] = h['%s/ratio' % g][:]

    # Needed to create two subplots with different sizes.
    # If other ratios are needed change height_ratios.
    plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    main = plt.subplot(gs[0])
    main.set_yscale('log')
    main.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlim([0.1, 1.])
    plt.ylim([1., 10.**3])

    ratio = plt.subplot(gs[1])
    ratio.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlim([0.1, 1.])
    plt.ylim([0.1, 1.1])

    main.plot(np.arange(0.1, 1.0, 0.001), np.divide(1., np.arange(0.1, 1.0, 0.001)), 'k--', label='Luck (AUC = 0.5000)')

    for gen_i in generators:
        print 'Creating curve for {}'.format(gen_i)
        main.plot(tprs[gen_i], fprs[gen_i], color=colors[gen_i], linestyle=line_styles[gen_i],
                  label='%s (AUC = %0.4f)' % (gen_i, aucs[gen_i]))

        ratio.plot(tprs[gen_i], ratios[gen_i], color=colors[gen_i], linestyle=line_styles[gen_i])

    ratio.set_xlabel("Signal Positive Rate")
    ratio.set_ylabel("Model / %s" % gen)
    main.set_ylabel("1 / [Background Efficiency]")
    main.set_title("ROC Curve for model trained on {}".format(gen))
    main.legend(loc=1, frameon=False)
    plt.tight_layout()
    plt.savefig("%sROC Curve %s" % (roc_img_dir, gen))
    plt.clf()
    print 'ROC Curve for {} successfully created.'.format(gen)


# Given generator, saves data (tpr, fpr, auc, ratio of various fpr to fpr of given gen) to 'gen'.h5 file.
def save_tpr_fpr_auc(gen, verbose=2):
    model = load_model('{path}{g}.h5'.format(path=models_dir, g=gen))

    if model.output_shape[1] == 1:
        tprs, fprs, aucs, ratios = __binary_roc_data(model, gen, verbose=verbose)
    else:
        tprs, fprs, aucs, ratios = __multi_roc_data(model, gen, verbose=verbose)

    # Saves file
    check_dir(roc_data_dir)

    with h5.File('{directory}{name}.h5'.format(directory=roc_data_dir, name=gen), 'w') as h:
        for gen_i in tprs.keys():
            t = h.create_group(gen_i)
            t.create_dataset('tpr', data=tprs[gen_i][()])
            t.create_dataset('fpr', data=fprs[gen_i][()])
            t.create_dataset('auc', data=[aucs[gen_i]])
            t.create_dataset('ratio', data=ratios[gen_i][()])


# Calculates and return true positive rate, false positive rate, area under curve,
# and ratios of false positive rate with respect to false positive rate of given generator gen.
# For binary-class model.
def __binary_roc_data(model, gen, verbose=2):
    # Contain true positive rate (signal efficiency) and false positive rate (background efficiency)
    # for each generator.
    tprs = {}
    fprs = {}
    aucs = {}

    for gen_i, gen_i_path in get_ready_names().iteritems():
        print 'Creating curve for {}'.format(gen_i)
        with h5.File(gen_i_path) as h:
            y_actual = h['test/y'][()]
        y_pred = np.array(model.predict(HDF5Matrix(gen_i_path, 'test/x'), verbose=verbose),
                          dtype=np.float64)
        aucs[gen_i] = roc_auc_score(y_actual, y_pred)
        tpr, fpr, _ = roc_curve(y_true=y_actual, y_score=y_pred, pos_label=1)

        fprs[gen_i] = np.divide(1., fpr)
        tprs[gen_i] = tpr

    return tprs, fprs, aucs, __calc_ratios(tprs, fprs, gen)


# Calculates and return true positive rate, false positive rate, area under curve,
# and ratios of false positive rate with respect to false positive rate of given generator gen.
# For multi-class model.
def __multi_roc_data(model, gen, verbose=2):
    # Contain true positive rate (signal efficiency), false positive rate (background efficiency) and
    # area under curve (auc) score for each generator.
    tprs = {}
    fprs = {}
    aucs = {}

    for gen_i, gen_i_path in get_ready_names().iteritems():
        print 'Creating data from model {}'.format(gen_i)
        with h5.File(gen_i_path) as h:
            y_actual = to_categorical(h['test/y'])
        y_pred = np.array(model.predict(HDF5Matrix(gen_i_path, 'test/x'), verbose=verbose),
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
        tprs[gen_i] = mean_tpr
        aucs[gen_i] = auc(fprs[gen_i], tprs[gen_i])
        fprs[gen_i] = np.divide(1., fprs[gen_i])

    return tprs, fprs, aucs, __calc_ratios(tprs, fprs, gen)


# Calculates ratios between fpr values of different generators
# with respect to given generator gen.
def __calc_ratios(tprs, fprs, gen):
    assert tprs.keys() == fprs.keys()
    assert gen in tprs

    ratios = {}
    f_interpolate = interp1d(tprs[gen], fprs[gen], bounds_error=False)
    for gen_i in generators:
        curr_fpr = fprs[gen_i]
        curr_tpr = tprs[gen_i]
        ratios[gen_i] = np.divide(curr_fpr, f_interpolate(curr_tpr))
    return ratios


create_roc_curve('Herwig Dipole')
create_roc_curve('Pythia Vincia')
create_roc_curve('Herwig Angular')
create_roc_curve('Sherpa')
create_roc_curve('Pythia Standard')
