import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

a = np.arange(.0002, .0040, .0002)
a = np.concatenate((a, [np.inf]))

a = np.ma.log(a)
print a
print np.mean(a)
a = np.subtract(a, np.mean(a))
print a
print np.ma.masked_where(a < 0, a)