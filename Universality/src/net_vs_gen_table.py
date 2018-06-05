import h5py as h5
import matplotlib.pyplot as plt

from get_fnames import generators

roc_img_dir = '../ROC/'
roc_data_dir = roc_img_dir + 'data/'


def make_table():
    data = []
    i = 0
    for g in generators:
        data.append([])
        with h5.File('{directory}{g}.h5'.format(directory=roc_data_dir, g=g)) as h:
            for g_i in generators:
                data[i].append('%0.4f' % h['%s/auc' % g_i][0])
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

    t = ax.table(cellText=data, colLabels=col_labels, rowLabels=row_labels, loc='center',
                 colWidths=[0.14 for x in col_labels])

    fig.tight_layout()

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    plt.savefig('{directory}ROC table.png'.format(directory=roc_img_dir), dpi=300)


make_table()
