from methods import *
import pickle

# gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
gen_used = "Herwig Dipole"

model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"

data = pickle.load(open('models_data/{model}_history_combined.p'
                        .format(model=model_name)))
graph_learning_curves(data, model_name + " combined")
