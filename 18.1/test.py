import numpy as np
from src.get_fnames import *
import matplotlib.pyplot as plt

cols = generators
rows = []
for i in cols:
    rows.append(i + ' net')
a = np.arange(0, 25)
a = np.reshape(a, [5, 5])
print a

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText=a, colLabels=cols, rowLabels=rows, loc='right')

fig.tight_layout()

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
