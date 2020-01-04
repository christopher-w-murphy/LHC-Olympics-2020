import argparse
import json
import os
import subprocess
import pandas as pd
from preprocessing import combiner
from preprocessing.utils import get_hdf_keys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--has_labels', default=False, action='store_true', help='Are the data labeled?')
    parser.add_argument('--file_name', type=str, help='The name of the file to process')
    return parser.parse_args()


def combine(data_path, file_name, has_labels):
    hdf_file = os.path.join(data_path, file_name)
    hdf_keys = get_hdf_keys(hdf_file)

    if has_labels:
        signal_keys = [key for key in hdf_keys if key.startswith('signal')]
        background_keys = [key for key in hdf_keys if key.startswith('background')]
        df_signal = combiner.combine_samples(hdf_file, signal_keys)
        df_background = combiner.combine_samples(hdf_file, background_keys)
        df = combiner.combine_signal_background(df_signal, df_background)
    else:
        unlabeled_keys = [key for key in hdf_keys if key.startswith('unlabeled')]
        df = combiner.combine_samples(hdf_file, unlabeled_keys)

    kin = combiner.Kinematics()
    df['delta_phi_jj'] = kin.delta_phi(df['phi_1'], df['phi_2'])
    df['delta_R_jj'] = kin.delta_R(df['eta_1'], df['eta_2'], df['phi_1'], df['phi_2'])
    df['mass_jj'] = kin.invariant_mass(df['mass_1'], df['mass_2'], df['pt_1'], df['pt_2'], df['eta_1'], df['eta_2'], df['phi_1'], df['phi_2'])
    df['mass_1+mass_2'] = kin.mass_sum(df['mass_1'], df['mass_2'])
    df['|mass_1-mass_2|'] = kin.mass_diff(df['mass_1'], df['mass_2'])

    output_file = os.path.join(data_path, 'processed_{}'.format(''.join(file_name.split('_')[1:])))
    df.to_hdf(output_file, key='processed', format='table', mode='a')

    subprocess.call(['rm', hdf_file])


if __name__ == '__main__':
    options = parse_args()
    has_labels = options.has_labels
    file_name = options.file_name

    with open('./preprocessing/configs.json') as f:
        config = json.load(f)
    data_path = config.processedData

    combine(data_path, file_name, has_labels)
