import pickle

from methods import *

# gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
# gen_used = "Herwig Dipole"

model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"

data = pickle.load(open('../models_data/{model}_history_{g}.p'
                        .format(model=model_name, g=gen_used)))
graph_learning_curves(data, model_name + " " + gen_used)
