import argparse
import logging
import os
from dask import delayed
import pandas as pd
from preprocessing.preprocessing import ClusterJets, ProcessData
from preprocessing.utils import df_to_hdf, get_data, timer

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_row', type=int, default=0, help='The 1st row of the table to be read')
    parser.add_argument('--stop_row', type=int, default=10000, help='The last row of the table to be read')
    parser.add_argument('--has_labels', default=False, action='store_true', help='Are the data labeled?')
    parser.add_argument('--file_name', type=str, help='The name of the file to process')
    return parser.parse_args()

@timer
def process_raw_events(input_file, output_file, n_start, n_stop, has_labels):
    df_raw = get_data(input_file, start_row=n_start, stop_row=n_stop)

    cluster_jets = ClusterJets(has_labels)
    cluster_jets.cluster_jets(df_raw)

    process_data = ProcessData()
    if has_labels:
        logging.info('processing signal events')
        df_signal = process_data.create_df(cluster_jets.alljets['signal'])
        logging.info('processing background events')
        df_background = process_data.create_df(cluster_jets.alljets['background'])
    else:
        logging.info('processing events')
        df_events = process_data.create_df(cluster_jets.alljets['unlabeled'])

    logging.info('writing to HDF')
    if has_labels:
        df_to_hdf(df_signal, output_file, key='signal_{}'.format(n_stop), format='table', mode='a')
        df_to_hdf(df_background, output_file, key='background_{}'.format(n_stop), format='table', mode='a')
    else:
        df_to_hdf(df_events, output_file, key='unlabeled_{}'.format(n_stop), format='table', mode='a')


if __name__ == '__main__':
    OPTIONS = parse_args()
    n_start = OPTIONS.start_row
    n_stop = OPTIONS.stop_row
    has_labels = OPTIONS.has_labels
    file_name = OPTIONS.file_name

    INPUT_DIR = '/Users/christopherwmurphy/Documents/projects/lhc_olympics/LHC-Olympics-2020/data_raw'
    input_file = os.path.join(INPUT_DIR, file_name)
    OUTPUT_DIR = '/Users/christopherwmurphy/Documents/projects/lhc_olympics/LHC-Olympics-2020/data_processed'
    output_file = os.path.join(OUTPUT_DIR, 'clustered_{}'.format(file_name))

    logging.info('Processing events {start} to {stop}'.format(start=n_start, stop=n_stop))
    processor = delayed(process_raw_events)(input_file, output_file, n_start, n_stop, has_labels)
    processor.compute()
