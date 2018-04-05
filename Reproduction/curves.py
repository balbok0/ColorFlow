import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
from get_file_names import *

'''
TO DO:
    - gen / gen_i subplot under main ROC.
'''

learning_curve_data_dir = 'models_data/'
learning_curve_images_dir = 'learning_curves/'

models_dir = 'models/'


# Produces learning curves of given generator
def create_learning_curve(gen):
    assert gen in generators
    assert os.path.exists("{d}history_{g}.p".format(d=learning_curve_data_dir, g=gen))

    data = pickle.load(open("models_data/history_{g}.p".format(g=gen)))

    assert data.keys() == ['acc', 'loss', 'val_acc', 'val_loss']

    print 'Making image for {}'.format(gen)

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

    plt.savefig('{d}Learning Curve {g} - Accuracy.png'.format(d=learning_curve_images_dir, g=gen))
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

    plt.savefig('{d}Learning Curve {g} - Loss.png'.format(d=learning_curve_images_dir, g=gen))
    plt.show()


# Produces ROC Curves, as defined in paper: https://arxiv.org/abs/1609.00607
def create_roc_curve(gen):
    assert gen in generators
    assert os.path.exists('{path}{g}'.format(path=models_dir, g=gen))

    colors = get_colors()
    files = get_ready_names()

    model = load_model('{path}{g}'.format(path=models_dir, g=gen))

    for gen_i, gen_i_path in files.iteritems():
        print 'Creating curve for {}'.format(gen_i)
        y_actual = np.array(HDF5Matrix(gen_i_path, 'test/y'), dtype=np.float64)
        y_pred = np.array(model.predict(HDF5Matrix(gen_i_path, 'test/x'), verbose=1),
                          dtype=np.float64)
        auc = roc_auc_score(y_actual, y_pred)
        tpr, fpr, thr = roc_curve(y_true=y_actual, y_score=y_pred, pos_label=1)
        fpr = np.divide(1.,  fpr)
        plt.plot(tpr, fpr, color=colors[gen_i], label='%s (AUC = %0.4f)' % (gen_i, auc))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for model trained on {}".format(gen))
    plt.legend(loc=1, frameon=False)
    plt.savefig("images/ROC Curve %s" % gen)
    plt.clf()
