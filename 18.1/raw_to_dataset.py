import h5py
import numpy as np
import time
from imblearn.over_sampling import SMOTE
import os
import sklearn.utils
from methods import zero_pad


def pre_process(x0, x1):
    x0 = np.reshape(x0, [len(x0), 625])
    x1 = np.reshape(x1, [len(x1), 625])

    x0 = sklearn.utils.shuffle(x0)
    x1 = sklearn.utils.shuffle(x1)

    x0 = np.reshape(x0, [len(x0), 25, 25])
    x1 = np.reshape(x1, [len(x1), 25, 25])

    x0tr = int(len(x0) * 0.6)
    x1tr = int(len(x1) * 0.6)
    x0val = int(len(x0) * 0.8)
    x1val = int(len(x1) * 0.8)
    return x0, x1, x0tr, x1tr, x0val, x1val


# Implement also large dataset method, which split raw samples
def raw_data_to_ready_data(fname0, fname1):

    ts = time.time()

    dir_name = os.path.dirname(fname0)

    x0 = h5py.File(fname0)['images']
    x1 = h5py.File(fname1)['images']
    x0, x1, x0tr, x1tr, x0val, x1val = pre_process(x0, x1)

    with h5py.File(dir_name + '/data.h5', 'w') as h:
        t = h.create_group('test')
        t.create_dataset('x', data=np.concatenate((x0[x0val:], x1[x1val:])))
        t.create_dataset('y', data=np.concatenate((np.zeros(len(x0[x0val:])),
                                                   np.ones(len(x1[x1val:])))))
        t = h.create_group('val')
        t.create_dataset('x', data=np.concatenate((x0[x0tr:x0val], x1[x1tr:x1val])))
        t.create_dataset('y', data=np.concatenate((np.zeros(x0val - x0tr),
                                                   np.ones(x1val - x1tr))))
    x0 = x0[:x0tr]
    x1 = x1[:x1tr]
    x0 = np.reshape(x0, [len(x0), 625])
    x1 = np.reshape(x1, [len(x1), 625])

    s = SMOTE()
    x, y = s.fit_sample(np.concatenate((x0, x1)), np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))))
    del x0, x1

    x, y = sklearn.utils.shuffle(x, y)
    x = np.reshape(x, [len(x), 25, 25])
    with h5py.File(dir_name + '/data.h5', 'a') as h:
        t = h.create_group('train')
        t.create_dataset('x', data=x)
        t.create_dataset('y', data=y)
    print "Time method took was", time.time() - ts, "seconds for", dir_name


def large_helper(x0, x1, dir_name):
    x0, x1, x0tr, x1tr, x0val, x1val = pre_process(x0, x1)

    with h5py.File(dir_name + '/data.h5', 'a') as h:
        xtest = np.concatenate((x0[x0val:], x1[x1val:]))
        xtest = zero_pad(xtest)
        h["test/x"].resize((h["test/x"].shape[0] + len(xtest)), axis=0)
        h["test/x"][-len(xtest):] = xtest

        ytest = np.concatenate((np.zeros(len(x0[x0val:])), np.ones(len(x1[x1val:]))))
        h['test/y'].resize((h["test/y"].shape[0] + len(ytest)), axis=0)
        h["test/y"][-len(ytest):] = ytest

        xval = np.concatenate((x0[x0tr:x0val], x1[x1tr:x1val]))
        xval = zero_pad(xval)
        h["val/x"].resize((h["val/x"].shape[0] + len(xval)), axis=0)
        h["val/x"][-len(xval):] = xval

        yval = np.concatenate((np.zeros(x0val - x0tr), np.ones(x1val - x1tr)))
        h['val/y'].resize((h["val/y"].shape[0] + len(yval)), axis=0)
        h["val/y"][-len(yval):] = yval

    x0 = x0[:x0tr]
    x1 = x1[:x1tr]
    x0 = np.reshape(x0, [len(x0), 625])
    x1 = np.reshape(x1, [len(x1), 625])
    s = SMOTE()
    x, y = s.fit_sample(np.concatenate((x0, x1)), np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))))
    del x0, x1
    x, y = sklearn.utils.shuffle(x, y)
    x = np.reshape(x, [len(x), 25, 25])
    x = zero_pad(x)
    with h5py.File(dir_name + '/data.h5', 'a') as h:
        h["train/x"].resize((h["train/x"].shape[0] + len(x)), axis=0)
        h["train/x"][-len(x):] = x

        h['train/y'].resize((h["train/y"].shape[0] + len(y)), axis=0)
        h["train/y"][-len(y):] = y


def first_large_helper(x0, x1, dir_name):
    x0, x1, x0tr, x1tr, x0val, x1val = pre_process(x0, x1)
    with h5py.File(dir_name + '/data.h5', 'w') as h:
        t = h.create_group('test')
        xtest = np.concatenate((x0[x0val:], x1[x1val:]))
        xtest = zero_pad(xtest)
        t.create_dataset('x', data=xtest, shape=xtest.shape, maxshape=(None, 33, 33, 1))
        ytest = np.concatenate((np.zeros(len(x0[x0val:])), np.ones(len(x1[x1val:]))))
        t.create_dataset('y', data=ytest, shape=ytest.shape, maxshape=[None])

        t = h.create_group('val')
        xval = np.concatenate((x0[x0tr:x0val], x1[x1tr:x1val]))
        xval = zero_pad(xval)
        t.create_dataset('x', data=xval, shape=xval.shape, maxshape=(None, 33, 33, 1))
        yval = np.concatenate((np.zeros(x0val - x0tr), np.ones(x1val - x1tr)))
        t.create_dataset('y', data=yval, shape=yval.shape, maxshape=[None])

    x0 = x0[:x0tr]
    x1 = x1[:x1tr]
    x0 = np.reshape(x0, [len(x0), 625])
    x1 = np.reshape(x1, [len(x1), 625])
    s = SMOTE()
    x, y = s.fit_sample(np.concatenate((x0, x1)), np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))))
    del x0, x1
    x, y = sklearn.utils.shuffle(x, y)
    x = np.reshape(x, [len(x), 25, 25])
    x = zero_pad(x)
    with h5py.File(dir_name + '/data.h5', 'a') as h:
        t = h.create_group('train')
        t.create_dataset('x', data=x, shape=x.shape, maxshape=(None, 33, 33, 1))
        t.create_dataset('y', data=y, shape=[len(y)], maxshape=[None])


def test_dimensions(fname):
    with h5py.File(fname, 'r') as f:
        for k in f.keys():
            for v in f[k].keys():
                print k, v, "shape is", f[k][v].shape


def large_raw_data_to_ready_data(max_size, fname0, fname1):
    ts = time.time()
    dir_name = os.path.dirname(fname0)
    size_0 = len(h5py.File(fname0, 'r')['images'])
    size_1 = len(h5py.File(fname1, 'r')['images'])
    splits = int(np.ceil(max(size_0, size_1)/max_size))  # Celling function
    increment_0 = int(size_0 / splits)
    increment_1 = int(size_1 / splits)
    print "Splitting data into", splits, "parts."
    print "Resulting data sets will be approximately of size",\
        max(increment_1, increment_0)*splits, "images"
    print "Training, Validation, Tests combined."
    print "Increment of 0's taken is", increment_0
    print "Increment of 1's taken is", increment_1
    print "--------------------------------------"
    print "Split 0 started"
    ti = time.time()
    first_large_helper(h5py.File(fname0)['images'][:increment_0],
                       h5py.File(fname1)['images'][:increment_1],
                       dir_name)
    print "Split 1 ended. It took %0.2f seconds." % (time.time() - ti)
    for i in range(1, splits):
        print "Split", i + 1, "started"
        ti = time.time()
        large_helper(h5py.File(fname0)['images'][i * increment_0:(i + 1) * increment_0],
                     h5py.File(fname1)['images'][i * increment_1:(i + 1) * increment_1],
                     dir_name)
        print "Split", i + 1, "ended. It took %0.2f seconds." % (time.time() - ti)

    # Print how long it took
    hs = int((time.time() - ts) / 3600)
    ms = int((time.time() - ts) % 3600 / 60)
    s = (time.time() - ts) % 3600 % 60
    print "Time method took was", hs, "hours,", ms, "minutes, %0.2f seconds for" % s, dir_name

    print ""
    test_dimensions(dir_name + "/data.h5")


# large_raw_data_to_ready_data(10000, '/media/balbok/Seagate Expansion Drive/Research/raw data/Sherpa/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5',
#                            '/media/balbok/Seagate Expansion Drive/Research/raw data/Sherpa/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5')
# large_raw_data_to_ready_data(10000,"/media/balbok/Seagate Expansion Drive/Research/raw data/Herwig/Dipole/QCD_Dipole250-300_j1p0_sj0p30_delphes_jets_images.h5",
#                             "/media/balbok/Seagate Expansion Drive/Research/raw data/Herwig/Dipole/WZ_combined_j1p0_sj0p30_delphes_jets_images.h5")
# large_raw_data_to_ready_data(10000,"/media/balbok/Seagate Expansion Drive/Research/raw data/Pythia/Vincia/qcd_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5",
#                             "/media/balbok/Seagate Expansion Drive/Research/raw data/Pythia/Vincia/w_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5")
# large_raw_data_to_ready_data(10000,"/media/balbok/Seagate Expansion Drive/Research/raw data/Pythia/Standard/qcd_j1p0_sj0p30_delphes_jets_pileup_images.h5",
#                             "/media/balbok/Seagate Expansion Drive/Research/raw data/Pythia/Standard/w_j1p0_sj0p30_delphes_jets_pileup_images.h5")
