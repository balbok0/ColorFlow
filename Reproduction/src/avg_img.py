import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.io_utils import HDF5Matrix

from get_file_names import get_raw_names, generators

path_to_avg = '/home/balbok/Documents/Research/ColorFlow/Reproduction/images/avg_img/'
path_to_npy = path_to_avg + 'npy/'
path_to_single = path_to_avg + 'single/'
path_to_diff = path_to_avg + 'differences/'


def _mean(data):
    split_n = int(np.ceil(len(data) / 1000.0))
    img_sum = []
    for i in range(25):
        img_sum.append([])
        for j in range(25):
            img_sum[i].append(0.0)
    img_sum = np.array(img_sum, dtype=np.float64)
    for i in range(split_n):
        temp = data[i*1000:(i+1)*1000]
        img_sum = np.add(img_sum, np.sum(temp, axis=0))
    return np.divide(img_sum, np.sum(img_sum))


def avg_img_npy(gen):
    print 'Saving npy average image for {}\n\n'.format(gen)

    if not os.path.exists(path_to_npy):
        os.makedirs(path_to_npy)

    fname = get_raw_names()[gen]

    data0 = HDF5Matrix(fname[0], 'images')
    ts = time.time()
    np.save(path_to_npy + gen + " Singlet", _mean(data0))
    print "Time it took for Singlet of {} was {:.3f}s.".format(gen, time.time()-ts) + "s"

    data1 = HDF5Matrix(fname[1], 'images')
    ts = time.time()
    np.save(path_to_npy + gen + " Octet", _mean(data1))
    print "Time it took for Octet of {} was {:.3f}s.".format(gen, time.time()-ts) + "s"


def prep_img(array):
    array = np.reshape(array, [25, 25])
    fig = plt.imshow(array, cmap=plt.get_cmap('seismic'))
    plt.xlabel("Prop. to translated azimuthal angle")
    plt.ylabel("Prop. to pseudorapidity")
    plt.colorbar(fig)


def avg_img(name):
    if not os.path.exists(path_to_single):
        os.makedirs(path_to_single)

    singlet = np.ma.log(np.load(path_to_npy + name + " Singlet.npy"))
    octet = np.ma.log(np.load(path_to_npy + name + " Octet.npy"))

    singlet = np.subtract(singlet, np.mean(singlet))
    octet = np.subtract(octet, np.mean(octet))

    singlet = np.ma.masked_where(singlet < -10, singlet)
    octet = np.ma.masked_where(octet < -10, octet)

    data = {'Octet': octet, 'Singlet': singlet, 'Octet minus Singlet': np.subtract(octet, singlet)}
    for jet, jet_img in data.iteritems():
        prep_img(jet_img)
        plt.title("{gen} {t}".format(gen=name, t=jet))
        plt.savefig("{path}average {gen} {t}".format(path=path_to_single, gen=name, t=jet))
        plt.show()
        plt.close()


def avg_dif_img(name1, name2):
    if not os.path.exists(path_to_diff):
        os.makedirs(path_to_diff)

    singlet1 = np.ma.log(np.load(path_to_npy + name1 + " Singlet.npy"))
    octet1 = np.ma.log(np.load(path_to_npy + name1 + " Octet.npy"))

    singlet2 = np.ma.log(np.load(path_to_npy + name2 + " Singlet.npy"))
    octet2 = np.ma.log(np.load(path_to_npy + name2 + " Octet.npy"))

    prep_img(np.subtract(octet1, octet2))
    plt.title(name1 + " minus " + name2 + " Octet")
    plt.savefig("{path}average {gen1} minus {gen2} Octet".format(path=path_to_diff, gen1=name1, gen2=name2))
    plt.show()
    plt.close()

    prep_img(np.subtract(singlet1, singlet2))
    plt.title(name1 + " minus " + name2 + " Singlet")
    plt.savefig("{path}average {gen1} minus {gen2} Singlet".format(path=path_to_diff, gen1=name1, gen2=name2))
    plt.show()
    plt.close()


for g1 in generators:
    for g2 in generators:
        if not g1 == g2:
            avg_dif_img(g1, g2)
