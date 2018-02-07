import networks as net
import gc
import h5py
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

'''
- WZ is 1, QCD/JZ is 0
- TO DO: You messed up Pythia Vincia. Retrain. ROC curves
- Done:
    -   Sherpa
    -   Herwig Angular/Dipole
    -   Model Visualization
'''

with h5py.File("data/Pythia/Vincia/qcd_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5", 'r') as f:
    x0 = np.array(f['images'][550000:])
print x0.shape
with h5py.File("data/Pythia/Vincia/w_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5", 'r') as f:
    x1 = np.array(f['images'][550000:])
print x1.shape
# gen_used = "Sherpa"
gen_used = "Pythia Vincia"
# gen_used = "Pythia Standard"
# gen_used = "Herwig Angular"
# gen_used = "Herwig Dipole"

model_name = "SM"
# model_name = "lanet"
# model_name = "lanet2"
# model_name = "lanet3"

'''
00 - 60%   - training
60 - 80%   - validation
80 - 100%  - test
'''
x0tr = int(len(x0) * 0)
x0val = int(len(x0) * 0.6)
x0test = int(len(x0) * 0.8)
x1tr = int(len(x1) * 0)
x1val = int(len(x1) * 0.6)
x1test = int(len(x1) * 0.8)
xtr = np.concatenate((x0[x0tr:x0val], x1[x1tr:x1val]))
xval = np.concatenate((x0[x0val:x0test], x1[x1val:x1test]))

# Free up memory. Needed, otherwise MemoryError is raised
del x0, x1
gc.collect()

# ZERO PADS. Since python does not allow passing by reference. Has to be done here
xval = np.concatenate((np.zeros(([len(xval), 25, 4])), xval), axis=2)
xval = np.concatenate((xval, np.zeros(([len(xval), 25, 4]))), axis=2)

xval = np.concatenate((np.zeros(([len(xval), 4, 33])), xval), axis=1)
xval = np.concatenate((xval, np.zeros(([len(xval), 4, 33]))), axis=1)
# ZERO PADS
xtr = np.concatenate((np.zeros(([len(xtr), 25, 4])), xtr), axis=2)
xtr = np.concatenate((xtr, np.zeros(([len(xtr), 25, 4]))), axis=2)

xtr = np.concatenate((np.zeros(([len(xtr), 4, 33])), xtr), axis=1)
xtr = np.concatenate((xtr, np.zeros(([len(xtr), 4, 33]))), axis=1)

xtr = xtr[..., np.newaxis]
xval = xval[..., np.newaxis]

ytr = np.concatenate((np.zeros(x0val - x0tr), np.ones(x1val - x1tr)))
yval = np.concatenate((np.zeros(x0test - x0val), np.ones((x1test - x1val))))

# MODEL LOADING
model = net.get_model(model_name)
# model = load_model("models/best_" + model_name + "_" + gen_used)
model.summary()

history = model.fit(x=xtr, y=ytr, epochs=5,
                    callbacks=[ModelCheckpoint(filepath="models/best_" + model_name + "_" +
                                                        gen_used, save_best_only=True)],
                    validation_data=(xval, yval), shuffle=True)

model.save("models/" + model_name + " " + gen_used)
with open('models_data/' + model_name + "_history_" + gen_used + "2.p", 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
