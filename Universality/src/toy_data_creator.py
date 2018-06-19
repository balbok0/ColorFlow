from __future__ import print_function

from os import mkdir, makedirs
from os.path import exists

import h5py

from get_fnames import *

toy_path = '../toy'
toy_size = 1000
f_paths = get_ready_names()
print(f_paths)
if not exists(toy_path):
    mkdir(toy_path)
for gen, path in f_paths.items():
    dir = toy_path + "/" + gen.replace(" ", "/")
    if not exists(dir):
        makedirs(dir)
    with h5py.File(dir + "/data.h5", 'w') as w:
        with h5py.File(path, 'r') as r:
            for key_main in r.keys():
                for key in r[key_main].keys():
                    w[key_main + "/" + key] = r[key_main + "/" + key][:toy_size]
