from methods import roc_curve
from get_fnames import *
gen_used = "Sherpa"
# gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
# gen_used = "Herwig Dipole"

model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"

for gen in generators:
    if not ['Pythia Standard', 'Sherpa', 'Herwig Dipole', 'Herwig Angular'].__contains__(gen):
        roc_curve(model_name, gen)
