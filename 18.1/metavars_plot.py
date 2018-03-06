from matplotlib import pyplot as plt
from keras.utils.io_utils import HDF5Matrix
import numpy as np
from get_fnames import *

meta_names = ['transverse momentum', 'transverse momentum Trimmed', 'Mass', 'Mass Trimmed',
              'subjet dR', 'tau 1', 'tau 2', 'tau 3']


def plot_meta_vars(gen):
    f0, f1 = get_raw_names()[gen]
    f0 = HDF5Matrix(f0, 'auxvars')
    f1 = HDF5Matrix(f1, 'auxvars')
    tmp0 = []
    tmp1 = []
    for i in range(len(meta_names)):
        tmp0.append([])
        tmp1.append([])
    for line in f0:
        for i in range(len(meta_names)):
            tmp0[i].append(line[i])
    for line in f1:
        for i in range(len(meta_names)):
            tmp1[i].append(line[i])
    for i in range(len(meta_names)):
        plt.hist(x=tmp0[i], log=True, histtype='step', color='r', bins=50, label="Singlet")
        plt.hist(x=tmp1[i], log=True, histtype='step', color='g', bins=50, label="Octet")
        plt.legend(loc='upper right')
        plt.title(gen + " " + meta_names[i])
        plt.savefig('images/' + gen + " " + meta_names[i] + ".png")
        plt.show()
        plt.close()


for generator in generators:
    plot_meta_vars(generator)
