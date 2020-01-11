import json
import os

from dask import delayed
from numba import njit
import numpy as np
import pandas as pd

from preprocessing.utils import get_data


@njit('f8[:](f8[:])')
def et_miss(event):
    pts = event[::3]
    phis = event[2::3]
    px = np.sum(pts * np.cos(phis))
    py = np.sum(pts * np.sin(phis))
    # $E_{T,miss}$ is defined as minus the sum of the transverse momenta
    return -1 * np.array([px, py])


@delayed
def get_et_miss(input_file, output_file):
    output = list()
    df = get_data(input_file, key='df')
    for event in df.values:
        output.append(et_miss(event))
    np.savez(output_file, np.array(output))


if __name__ == '__main__':
    with open('./preprocessing/configs.json') as f:
        config = json.load(f)
    input_dir = config['rawData']
    output_dir = config['processedData']

    fnames = [fname for fname in os.listdir(input_dir) if fname.endswith('.h5')]
    for fname in fnames:
        input_file = os.path.join(input_dir, fname)
        output_file = os.path.join(output_dir, 'etmiss_{}.npz'.format(fname.split('.')[0]))
        et_misses = get_et_miss(input_file, output_file)
        et_misses.compute()
