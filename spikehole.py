# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from pathlib import Path
import shutil
from urllib.parse import urlparse

import joblib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from one.api import ONE
from one.remote import aws
import one.alf.io as alfio
from ibllib.plots import Density
from neurodsp.voltage import destripe
from brainbox.io.one import SpikeSortingLoader
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params, download_test_data


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

WIN_SIZE = .1
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data/'
SCRATCH_DIR = ROOT_DIR / 'scratch'
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
    # one = get_one()
    # eid, pname = one.pid2eid(pid)
    # session_path = one.eid2path(eid)
    t0 = t - win_size / 2.0
    t1 = t + win_size / 2.0
    return (pid, t0, t1)


def get_snippet_path(snippet):
    pid, t0, t1 = snippet
    t = (t0 + t1) / 2.0
    return DATA_DIR / "snippets" / pid / f'T{t:.3f}'


def get_cbin_snippet_path(snippet, sr_ap):
    """Recover the path to the .cbin cached to disk, to run pyks2 on it."""
    pid, t0, t1 = snippet
    path = DATA_DIR / 'cache' / 'ap' / urlparse(sr_ap.url_cbin).path[1:]
    dir = path.parent.resolve()
    name = path.name
    l = list(name.split('.'))
    l[3] = 'stream'
    name = '.'.join(l)
    return dir / f'chunk_{t0-1:06d}_to_{t1-1:06d}' / name


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
    return raw_ap, sr_ap


def download_spike_data(snippet):
    pid, t0, t1 = snippet
    one = get_one()
    ssl = SpikeSortingLoader(pid=pid, one=one)
    spikes, clusters, channels = ssl.load_spike_sorting()
    return spikes, clusters, channels


# -------------------------------------------------------------------------------------------------
# Spike sorting
# -------------------------------------------------------------------------------------------------

def run_pyks2(snippet, bin_file, force=False):
    # Get the pyks2 parameters.
    params = ibl_pykilosort_params(bin_file)
    params['Th'] = [6, 3]

    # pyks2 output directory.
    ks_output_dir = get_snippet_path(snippet)
    ks_output_dir.mkdir(exist_ok=True, parents=True)

    alf_path = ks_output_dir / 'alf'
    alf_path.mkdir(exist_ok=True, parents=True)

    # Run the spike sorting on the raw data snippet.
    if force or not (alf_path / 'spikes.samples.npy').exists():
        print("Running spike sorting...")
        run_spike_sorting_ibl(
            bin_file, scratch_dir=SCRATCH_DIR, ks_output_dir=ks_output_dir, alf_path=alf_path,
            delete=False, log_level='INFO', params=params)

    spikes = alfio.load_object(alf_path, 'spikes')
    clusters = alfio.load_object(alf_path, 'clusters')
    channels = alfio.load_object(alf_path, 'channels')

    return spikes, clusters, channels


# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------

def select_raw(snippet, raw, fs=None, show_interval=None):
    if not show_interval:
        return raw

    assert fs
    pid, t0, t1 = snippet

    t0s, t1s = show_interval
    t0s -= t0
    t1s -= t0

    first, last = (int(t0s * fs), int(t1s * fs))
    return raw[:, first:last]


def select_times_channels(spikes, clusters, show_interval):
    t0s, t1s = show_interval or (t0, t1)

    s = spikes.times
    c = clusters.channels[spikes.clusters]

    idx = (s >= t0s) & (s < t1s)
    si = 1000 * (s[idx] - t0s)
    ci = c[idx]

    return si, ci


def make_plot(
        snippet, raw, fs=None, show_interval=None, spikes=None, clusters=None,
        spikes_rerun=None, clusters_rerun=None):
    assert fs
    pid, t0, t1 = snippet

    # Snippet to plot.
    t0s, t1s = show_interval or t0, t1

    # Raw data snippet.
    raws = select_raw(snippet, raw, fs=fs, show_interval=show_interval)
    raws = destripe(raws, fs=fs)

    # Find quantiles.
    q = .005
    vmin, vmax = np.quantile(raws, q), np.quantile(raw, 1 - q)

    # Figure.
    fig, ax = plt.subplots(figsize=(12, 6))

    # Density plot.
    d1 = Density(raws, fs=fs, taxis=1, ax=ax, vmin=vmin, vmax=vmax, cmap='Greys')

    # Overlay spikes.
    if spikes is not None and clusters is not None:
        si, ci = select_times_channels(spikes, clusters, show_interval)
        ax.scatter(si, ci, s=50, color=[1, 0, 0], alpha=.5)
    if spikes_rerun is not None and clusters_rerun is not None:
        si, ci = select_times_channels(spikes_rerun, clusters_rerun, show_interval)
        ax.scatter(si, ci, s=50, color=[0, 1, 0], alpha=.5)

    # Figure title.
    ax.set_title(f"{pid}: {t0s}-{t1s}")

    # Show the figure.
    plt.show()


# -------------------------------------------------------------------------------------------------
# Entry-point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # PIDs.
    pid = '84bb830f-b9ff-4e6b-9296-f458fb41d160'
    # pid = '5a34d971-1cb3-4f0e-8dfe-e51e2313a668'

    # Time.
    t0 = 1495
    t1 = 1505
    T = 1500
    dt = .1

    # Snippet.
    snippet = (pid, t0, t1)
    show_interval = (T, T + dt)

    # Raw data.
    raw_ap, sr_ap = download_raw_data(snippet)
    fs = sr_ap.fs

    # Spikes.

    # Load existing spikes.
    spikes_orig, clusters_orig, channels_orig = download_spike_data(snippet)

    # Also, run spike sorting again.
    spikes_rerun, clusters_rerun, channels_rerun = run_pyks2(
        snippet, get_cbin_snippet_path(snippet, sr_ap))
    spikes_rerun.times += t0 - 1

    # Show the spikes on top of the raw data.
    make_plot(
        snippet, raw_ap, fs=fs, show_interval=show_interval,
        spikes=spikes_orig, clusters=clusters_orig,
        spikes_rerun=spikes_rerun, clusters_rerun=clusters_rerun)
