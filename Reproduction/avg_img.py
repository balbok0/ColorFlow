from get_file_names import get_raw_names, generators
import numpy as np
from keras.utils.io_utils import HDF5Matrix
import time
import os
import matplotlib.pyplot as plt

avg_npy_dir = 'images/avg_img/npy/'


def _mean(data):
    split_n = int(np.ceil(len(data) / 1000.0))
    sum = []
    for i in range(25):
        sum.append([])
        for j in range(25):
            sum[i].append(0.0)
    sum = np.array(sum, dtype=np.float64)
    for i in range(split_n):
        temp = data[i*1000:(i+1)*1000]
        sum = np.add(sum, np.sum(temp, axis=0))
    return np.divide(sum, np.sum(sum))


def avg_img_npy(gen):
    print 'Saving npy average image for {}\n\n'.format(gen)

    if not os.path.exists(avg_npy_dir):
        os.makedirs(avg_npy_dir)

    fname = get_raw_names()[gen]

    data0 = HDF5Matrix(fname[0], 'images')
    ts = time.time()
    np.save(avg_npy_dir + gen + " Singlet", _mean(data0))
    print "Time it took for Singlet of {} was {:.3f}s.".format(gen, time.time()-ts) + "s"

    data1 = HDF5Matrix(fname[1], 'images')
    ts = time.time()
    np.save(avg_npy_dir + gen + " Octet", _mean(data1))
    print "Time it took for Octet of {} was {:.3f}s.".format(gen, time.time()-ts) + "s"


for g in generators[3:]:
    avg_img_npy(g)
