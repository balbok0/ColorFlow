from get_fnames import generators, get_ready_names
import numpy as np
import sklearn
from keras.models import load_model
from keras.utils import HDF5Matrix


def make_table(f=sklearn.metrics.roc_auc_score):
    col_rows = generators
    data = []
    i = 0
    for g in generators:
        data.append([])
        model = load_model("models/validated {} {}".format('SM', g))
        for g_i in generators:
            y_pred = model.predict(HDF5Matrix(get_ready_names()[g_i], 'test/x'))
            y_pred = np.array(y_pred, dtype=np.float64)
            y_actual = HDF5Matrix(get_ready_names()[g_i], 'test/y')
            y_actual = np.array(y_actual, dtype=np.float64)

            value = f(y_actual, y_pred)
            data[i].append(value)
        i += 1
