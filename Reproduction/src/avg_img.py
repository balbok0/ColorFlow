from get_file_names import get_raw_names, generators
import numpy as np
from keras.utils.io_utils import HDF5Matrix
import time
import os
import matplotlib.pyplot as plt

path_to_avg = '../images/avg_img/'
path_to_npy = path_to_avg + 'npy/'


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
    singlet = np.ma.log(np.load(path_to_npy + name + " Singlet.npy"))
    octet = np.ma.log(np.load(path_to_npy + name + " Octet.npy"))

    singlet = np.subtract(singlet, np.mean(singlet))
    octet = np.subtract(octet, np.mean(octet))

    singlet = np.ma.masked_where(singlet < -10, singlet)
    octet = np.ma.masked_where(octet < -10, octet)

    prep_img(octet)
    plt.title(name + " Octet")
    plt.savefig(path_to_avg_img + "average " + name + " Octet")
    plt.show()
    plt.close()

    prep_img(singlet)
    plt.title(name + " Singlet")
    plt.savefig(path_to_avg_img + "average " + name + " Singlet")
    plt.show()
    plt.close()

    prep_img(np.subtract(octet, singlet))
    plt.title(name + " Octet minus Singlet")
    plt.savefig(path_to_avg_img + "average " + name + " Octet minus Singlet")
    plt.show()
    plt.close()


def avg_dif_img(name1, name2):
    singlet1 = np.ma.log(np.load(path_to_npy + name1 + " Singlet.npy"))
    octet1 = np.ma.log(np.load(path_to_npy + name1 + " Octet.npy"))

    singlet2 = np.ma.log(np.load(path_to_npy + name2 + " Singlet.npy"))
    octet2 = np.ma.log(np.load(path_to_npy + name2 + " Octet.npy"))

    prep_img(np.subtract(octet1, octet2))
    plt.title(name1 + " minus " + name2 + " Octet")
    plt.savefig("{path}differences/average {gen1} minus {gen2} Octet".format(path=path_to_avg_img, gen1=name1,
                                                                             gen2=name2))
    plt.show()
    plt.close()

    prep_img(np.subtract(singlet1, singlet2))
    plt.title(name1 + " minus " + name2 + " Singlet")
    plt.savefig("{path}differences/average {gen1} minus {gen2} Singlet".format(path=path_to_avg_img, gen1=name1,
                                                                               gen2=name2))
    plt.show()
    plt.close()



for g in generators[3:]:
    avg_img_npy(g)
