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


# -------------------------------------------------------------------------------------------------
# Entry-point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # PIDs.
    pid = '84bb830f-b9ff-4e6b-9296-f458fb41d160'
    # pid = '5a34d971-1cb3-4f0e-8dfe-e51e2313a668'

    snippet = make_snippet(pid, 1500)
    print(snippet)
