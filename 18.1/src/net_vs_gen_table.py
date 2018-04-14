from get_fnames import generators, get_ready_names
import numpy as np
import sklearn
from keras.models import load_model
from keras.utils import HDF5Matrix
import matplotlib.pyplot as plt


def make_table(f=sklearn.metrics.roc_auc_score):
    data = []
    i = 0
    for g in generators:
        data.append([])
        model = load_model("../models/validated {} {}".format('SM', g))
        for g_i in generators:
            y_pred = model.predict(HDF5Matrix(get_ready_names()[g_i], 'test/x'))
            y_pred = np.array(y_pred, dtype=np.float64)
            y_actual = HDF5Matrix(get_ready_names()[g_i], 'test/y')
            y_actual = np.array(y_actual, dtype=np.float64)

            value = f(y_actual, y_pred)
            data[i].append(value)
        i += 1

        # Prepares labels for rows and columns
        col_labels = generators
        row_labels = []
        for i in col_labels:
            row_labels.append(i + ' net')

        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        ax.table(cellText=data, colLabels=col_labels, rowLabels=row_labels, loc='right')

        fig.tight_layout()

        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
