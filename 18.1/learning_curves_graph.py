from methods import *
import pickle

gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
# gen_used = "Herwig Dipole"

model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"

print 'models_data/{model}_history_{generator}.p'.format(model=model_name, generator=gen_used)
data = pickle.load(open('models_data/{}_history_{}.p'))
graph_learning_curves(data, model_name + " " + gen_used)
