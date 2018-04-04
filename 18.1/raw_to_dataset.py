import h5py
import numpy as np
import time
from imblearn.under_sampling import RandomUnderSampler
import sklearn.utils
from get_fnames import *
from methods import zero_pad

max_chunks = 10000


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


def helper(x0, x1, dir_name):
    x0, x1, x0tr, x1tr, x0val, x1val = pre_process(x0, x1)

    with h5py.File(dir_name + '/data.h5', 'a') as h:
        xtest = zero_pad(np.concatenate((x0[x0val:], x1[x1val:])))
        h["test/x"].resize((h["test/x"].shape[0] + len(xtest)), axis=0)
        h["test/x"][-len(xtest):] = xtest

        ytest = np.concatenate((np.zeros(len(x0[x0val:])), np.ones(len(x1[x1val:]))))
        h['test/y'].resize((h["test/y"].shape[0] + len(ytest)), axis=0)
        h["test/y"][-len(ytest):] = ytest

        xval = zero_pad(np.concatenate((x0[x0tr:x0val], x1[x1tr:x1val])))
        h["val/x"].resize((h["val/x"].shape[0] + len(xval)), axis=0)
        h["val/x"][-len(xval):] = xval

        yval = np.concatenate((np.zeros(x0val - x0tr), np.ones(x1val - x1tr)))
        h['val/y'].resize((h["val/y"].shape[0] + len(yval)), axis=0)
        h["val/y"][-len(yval):] = yval

    x0 = x0[:x0tr]
    x1 = x1[:x1tr]
    x0 = np.reshape(x0, [len(x0), 625])
    x1 = np.reshape(x1, [len(x1), 625])

    r = RandomUnderSampler()
    x, y = r.fit_sample(np.concatenate((x0, x1)), np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))))
    x, y = sklearn.utils.shuffle(x, y)

    x = zero_pad(np.reshape(x, [len(x), 25, 25]))
    with h5py.File(dir_name + '/data.h5', 'a') as h:
        h["train/x"].resize((h["train/x"].shape[0] + len(x)), axis=0)
        h["train/x"][-len(x):] = x

        h['train/y'].resize((h["train/y"].shape[0] + len(y)), axis=0)
        h["train/y"][-len(y):] = y


def first_helper(x0, x1, dir_name):
    x0, x1, x0tr, x1tr, x0val, x1val = pre_process(x0, x1)
    with h5py.File(dir_name + '/data.h5', 'w') as h:
        t = h.create_group('test')
        xtest = zero_pad(np.concatenate((x0[x0val:], x1[x1val:])))
        t.create_dataset('x', data=xtest, shape=xtest.shape, maxshape=([None] + list(xtest.shape[1:])))
        ytest = np.concatenate((np.zeros(len(x0[x0val:])), np.ones(len(x1[x1val:]))))
        t.create_dataset('y', data=ytest, shape=ytest.shape, maxshape=[None])

        t = h.create_group('val')
        xval = zero_pad(np.concatenate((x0[x0tr:x0val], x1[x1tr:x1val])))
        t.create_dataset('x', data=xval, shape=xval.shape, maxshape=([None] + list(xtest.shape[1:])))
        yval = np.concatenate((np.zeros(x0val - x0tr), np.ones(x1val - x1tr)))
        t.create_dataset('y', data=yval, shape=yval.shape, maxshape=[None])

    x0 = x0[:x0tr]
    x1 = x1[:x1tr]
    x0 = np.reshape(x0, [len(x0), 625])
    x1 = np.reshape(x1, [len(x1), 625])

    r = RandomUnderSampler()
    x, y = r.fit_sample(np.concatenate((x0, x1)), np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))))
    x, y = sklearn.utils.shuffle(x, y)

    x = zero_pad(np.reshape(x, [len(x), 25, 25]))

    with h5py.File(dir_name + '/data.h5', 'a') as h:
        t = h.create_group('train')
        t.create_dataset('x', data=x, shape=x.shape, maxshape=([None] + list(xtest.shape[1:])))
        t.create_dataset('y', data=y, shape=[len(y)], maxshape=[None])


# Implements also large dataset method, which splits raw samples into max_size ones, and saves them continuously.
def raw_data_to_ready_data(max_size, gen):
    ts = time.time()

    fname0, fname1 = get_raw_names()[gen]
    dir_name = drive_path + "ready data/" + gen.replace(' ', '/')
    if not os.path.exists(dir_name + '/'):
        os.makedirs(dir_name + '/')

    size_0 = len(h5py.File(fname0, 'r')['images'])
    size_1 = len(h5py.File(fname1, 'r')['images'])
    splits = int(np.ceil(max(size_0, size_1) / max_size))  # Celling function

    increment_0 = int(size_0 / splits)
    increment_1 = int(size_1 / splits)

    print gen
    print
    print "Splitting data into", splits, "parts."
    print "Resulting data sets will be approximately of size", \
        min(increment_1, increment_0) * splits, "images."
    print "Training, Validation, Tests combined."
    print "Increment of 0's taken is", increment_0
    print "Increment of 1's taken is", increment_1
    print "--------------------------------------"
    print "Split 1 started"

    ti = time.time()
    first_helper(h5py.File(fname0)['images'][:increment_0],
                 h5py.File(fname1)['images'][:increment_1],
                 dir_name)
    print "Split 1 ended. It took %0.2f seconds." % (time.time() - ti)
    for i in range(1, splits):
        print "Split", i + 1, "started"
        ti = time.time()
        helper(h5py.File(fname0)['images'][i * increment_0:(i + 1) * increment_0],
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


def test_dimensions(fname):
    with h5py.File(fname, 'r') as f:
        for k in f.keys():
            for v in f[k].keys():
                print k, v, "shape is", f[k][v].shape


raw_data_to_ready_data(max_chunks, 'Pythia Standard')
