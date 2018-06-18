import os

import local_vars

drive_path = local_vars.basic_dir
raw_dir = local_vars.raw_dir
ready_path = local_vars.ready_dir
models_path = local_vars.models

generators = ['Herwig Angular', 'Herwig Dipole', 'Sherpa',
              'Pythia Standard', 'Pythia Vincia']


def get_ready_path(gen):
    gen_path = ready_path + gen.replace(' ', '/') + '/data.h5'
    if not os.path.exists(gen_path):
        raise IOError("Generator " + gen + " not found, at path: " + gen_path)
    return gen_path
