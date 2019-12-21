import logging
import time
import h5py
import pandas as pd

logging.basicConfig(level=logging.INFO)


def get_data(input_file, key=None, start_row=0, stop_row=None):
    return pd.read_hdf(input_file, key=key, start=start_row, stop=stop_row)


def get_hdf_keys(file_name):
    with h5py.File(file_name, 'r') as f:
        return list(f.keys())


def timer(func):
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        time_delta = time.time() - start_time
        logging.info('Ran in %i min %0.1f sec' %(int(time_delta // 60), time_delta % 60))
    return timer_wrapper


# this is an inelegant way to handle multiprocess writing to hdf5
def df_to_hdf(dataframe, filename, *args, **kwargs):
    while True:
        try:
            dataframe.to_hdf(filename, *args, **kwargs)
            break
        except HDF5ExtError as e:
            logging.error(e)
            time.sleep(5)
