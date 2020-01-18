import numpy as np


def bin_means(binvals):
    return np.array([0.5 * (binvals[i] + binvals[i + 1]) for i in range(0, len(binvals) - 1)])


def bin_widths(binvals):
    return np.array([-binvals[i] + binvals[i + 1] for i in range(0, len(binvals) - 1)])


def get_histogram(df, bins, q=0.0, feature='pca'):
    # select the 1 - q fraction of events with the highest outlier scores
    qcut = df[feature] > df[feature].quantile(q)
    # create historgram of selected scores
    return np.histogram(df[qcut]['mass_jj'], bins=bins)


def get_histograms(df, bins, feature='pca', quantiles=[0.0, 0.7, 0.9]):
    return [get_histogram(df, bins, q=q, feature=feature) for q in quantiles]


def get_mask(histogram, signal_region_min, signal_region_max):
    min_idx = np.where(bin_means(histogram[1]) == signal_region_min)[0][0]
    max_idx = np.where(bin_means(histogram[1]) == signal_region_max)[0][0]
    return np.arange(min_idx, max_idx + 1)


def get_sideband_indices(histogram, sideband_min, sideband_max):
    min_idx = np.where(bin_means(histogram[1]) == sideband_min)[0][0]
    max_idx = np.where(bin_means(histogram[1]) == sideband_max)[0][0]
    return min_idx, max_idx
