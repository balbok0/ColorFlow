import h5py
from keras.models import load_model
import numpy as np
import methods
import matplotlib.pyplot as plt

model = load_model("models/best_SM_Herwig Dipole")

with h5py.File("data/Sherpa/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", 'r') as f:
    x1 = np.array(f['images'])
    x1 = x1[int(len(x1)*0.8):]
    x1 = x1[..., np.newaxis]
with h5py.File("data/Sherpa/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", 'r') as f:
    x0 = np.array(f['images'])
    x0 = x0[int(len(x0) * 0.8):]
    x0 = x0[..., np.newaxis]
x = np.concatenate((x0, x1))
x = methods.zero_pad(x)
print x.shape
y = np.concatenate((np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)))
print y.shape
y_pred = model.predict(x)
print y_pred.shape
methods.roc_curve(y_pred, y, "Sherpa", "SM", "navy")

with h5py.File("data/Herwig/Angular/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", 'r') as f:
    x1 = np.array(f['images'])
    x1 = x1[int(len(x1)*0.8):]
    x1 = x1[..., np.newaxis]
with h5py.File("data/Herwig/Angular/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", 'r') as f:
    x0 = np.array(f['images'])
    x0 = x0[int(len(x0) * 0.8):]
    x0 = x0[..., np.newaxis]
x = np.concatenate((x0, x1))
x = methods.zero_pad(x)
print x.shape
y = np.concatenate((np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)))
print y.shape
y_pred = model.predict(x)
print y_pred.shape
methods.roc_curve(y_pred, y, "Herwig Angular", "SM", "darkorange")

with h5py.File("data/Herwig/Dipole/WZ_combined_j1p0_sj0p30_delphes_jets_images.h5", 'r') as f:
    x1 = np.array(f['images'])
    x1 = x1[int(len(x1)*0.8):]
    x1 = x1[..., np.newaxis]
with h5py.File("data/Herwig/Dipole/QCD_Dipole250-300_j1p0_sj0p30_delphes_jets_images.h5", 'r') as f:
    x0 = np.array(f['images'])
    x0 = x0[int(len(x0) * 0.8):]
    x0 = x0[..., np.newaxis]
x = np.concatenate((x0, x1))
x = methods.zero_pad(x)
print x.shape
y = np.concatenate((np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)))
print y.shape
y_pred = model.predict(x)
print y_pred.shape
methods.roc_curve(y_pred, y, "Herwig Dipole", "SM", "chartreuse")
plt.title("Herwig Dipole SM ROC Curve")
plt.legend(loc=3)
plt.savefig("images/" + "Herwig Dipole SM ROC")
plt.show()
