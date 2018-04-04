from numpy import ma
import numpy as np
import sys
from get_fnames import *
from keras.utils.io_utils import HDF5Matrix
import h5py


def distributions(x):
    c0 = 0
    c1 = 0
    for i in x:
        if i == 0:
            c0+=1
        elif i == 1:
            c1+=1
    print 'Singlets: ' + str(c0)
    print 'Octets: ' + str(c1)


for g in generators:
    fname = get_ready_names()[g]
    p = ['train', 'val', 'test']
    for i in p:
        print g + " " + i
        with h5py.File(fname) as h:
            distributions(h[i + '/y'])
