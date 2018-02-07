import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from methods import *

herwig_angular = pickle.load(open('models_data/SM_history_Herwig Angular.p', 'r'))
herwig_dipole = pickle.load(open('models_data/SM_history_Herwig Dipole.p', 'r'))
sherpa1 = pickle.load(open('models_data/SM_history_Sherpa.p', 'r'))
sherpa2 = pickle.load(open('models_data/SM_history_Sherpa2.p', 'r'))
sherpa = combine_dict(sherpa1, sherpa2)

print sherpa