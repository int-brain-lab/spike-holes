# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from pathlib import Path
import shutil

import joblib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from one.api import ONE
from one.remote import aws
from ibllib.plots import Density
from neurodsp.voltage import destripe
from brainbox.io.one import SpikeSortingLoader


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

WIN_SIZE = .1
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data/'
PYKS2_DIR = (ROOT_DIR / '..').resolve()
GLOBAL_ONE = None


def get_one():
    if not globals()['GLOBAL_ONE']:
        globals()['GLOBAL_ONE'] = ONE(
            base_url="https://alyx.internationalbrainlab.org", cache_dir=DATA_DIR)
    return globals()['GLOBAL_ONE']


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

def make_snippet(pid, t, win_size=WIN_SIZE):
    one = get_one()
    eid, pname = one.pid2eid(pid)
    session_path = one.eid2path(eid)
    t0 = t - win_size / 2.0
    t1 = t + win_size / 2.0
    return (pid, t0, t1)


def get_plot_path(snippet):
    pid, t0, t1 = snippet
    pid, t0, t1 = snippet
    t = (t0 + t1) / 2.0
    return DATA_DIR / pid / f'T{t:.3f}'


# -------------------------------------------------------------------------------------------------
# Downloading
# -------------------------------------------------------------------------------------------------

def download_raw_data(snippet):
    pid, t0, t1 = snippet
    one = get_one()

    ssl = SpikeSortingLoader(pid=pid, one=one)
    channels = ssl.load_channels()
    sr_ap = ssl.raw_electrophysiology(band="ap", stream=True)

    first, last = (int(t0 * sr_ap.fs), int(t1 * sr_ap.fs))
    raw_ap = sr_ap[first:last, :-sr_ap.nsync].T
    return raw_ap, sr_ap.fs


# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------

def make_plot(snippet, raw, fs=None, spike_times=None):
    assert fs
    pid, t0, t1 = snippet
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.imshow(raw)
    vmin = vmax = None
    d1 = Density(raw, fs=fs, taxis=1, ax=ax, vmin=vmin, vmax=vmax, cmap='Greys')
    ax.set_title(f"{pid}: {t0}-{t1}")
    plt.show()


# -------------------------------------------------------------------------------------------------
# Entry-point
# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # PIDs.
    pid = '84bb830f-b9ff-4e6b-9296-f458fb41d160'
    # pid = '5a34d971-1cb3-4f0e-8dfe-e51e2313a668'

    snippet = make_snippet(pid, 1500)
    path = get_plot_path(snippet)
    raw_ap, fs = download_raw_data(snippet)
    destriped = destripe(raw_ap, fs=fs)

    make_plot(snippet, destriped, fs=fs)
