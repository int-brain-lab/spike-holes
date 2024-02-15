import pickle

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from pykilosort.learn import mexMPnu8
from ibllib.plots import Density
from neurodsp.voltage import destripe


with open('las-04.pkl', 'rb') as f:
    kwargs = pickle.load(f)
# Params, dataRAW, U, W, mu, iC, iW, UtU, iList, wPCA, params

dataRAW = kwargs['dataRAW']

st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp = mexMPnu8(**kwargs)

t0s, t1s = 0, .1
fs = 3e4
i0 = int(t0s * fs)
i1 = int(t1s * fs)

# Raw data snippet.
raws = dataRAW[i0:i1, :].get()
raws = destripe(raws.T, fs=fs)

# Find quantiles.
q = .01
vmin, vmax = np.quantile(raws, q), np.quantile(raws, 1 - q)

fig, ax = plt.subplots(figsize=(12, 6))
d1 = Density(raws, fs=fs, taxis=1, ax=ax, vmin=vmin, vmax=vmax, cmap='Greys')

si = st0.get() / fs
si = si[(si >= t0s) & (si < t1s)]
ci = np.random.randint(size=si.shape, low=0, high=384)
ax.vlines(si * 1e3, ymin=0, ymax=384, color='r')

plt.show()
