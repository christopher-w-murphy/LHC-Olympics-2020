import logging

import matplotlib.pyplot as plt
from numdifftools import Jacobian
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm

from .utils import bin_means, bin_widths

logging.basicConfig(level=logging.INFO)


class BackgroundParameterization():
    """
    See the ATLAS diboson resonance search: https://arxiv.org/pdf/1708.04445.pdf
    """
    def __init__(self, xi=0., S=13000.):
        self.xi = xi
        self.S = S

    def fit_func(self, x, p1, p2, p3):
        y = x / self.S
        return p1 * (1. - y)**(p2 - self.xi * p3) * y**-p3

    def fit_func_array(self, parr, xdata):
        return np.array([self.fit_func(x, *parr) for x in xdata])


def minus_log_likelihood(nuisance_arr, observed, expected, x_signal_cov=1.):
    # For a single counting experiment
    expected_nuisance = nuisance_arr[0]
    if len(nuisance_arr) > 1:
        predicted = nuisance_arr[1]
    else:
        predicted = 0

    # Poisson terms, starting with lambda
    pois_lambda = expected + expected_nuisance + predicted
    # Prevent prediction from going negative
    if pois_lambda < 10**-10:
        pois_lambda = 10**-10
    # Poisson term, ignore the factorial piece which will cancel in likelihood ratio
    log_likelihood = observed * np.log(pois_lambda) - pois_lambda

    # Gaussian nuisance term
    nuisance_term = -0.5 * expected_nuisance**2 / x_signal_cov
    log_likelihood += nuisance_term

    return -1. * log_likelihood


def compute_p_value(observed, expected, x_signal_cov=1., num_nuis_arr_init=[0.], num_bounds=None, den_nuis_arr_init=[0., 1.], den_bounds=None, verbose=False):
    log_like_args = (observed, expected, x_signal_cov)
    # Numerator of likelihood ratio
    minimize_log_numerator = minimize(minus_log_likelihood, num_nuis_arr_init, args=log_like_args, bounds=num_bounds)
    if verbose:
        logging.info('Numerator:')
        for key, val in minimize_log_numerator.items():
            logging.info('{key} = {val}'.format(key=key, val=val))
    # Numerator of likelihood ratio
    minimize_log_denominator = minimize(minus_log_likelihood, den_nuis_arr_init, args=log_like_args, bounds=den_bounds)
    if verbose:
        logging.info('Denominator:')
        for key, val in minimize_log_denominator.items():
            logging.info('{key} = {val}'.format(key=key, val=val))

    if minimize_log_denominator.x[-1] < 0:
        Zval = 0
        neglognum = 0
        neglogden = 0
    else:
        neglognum = minus_log_likelihood(minimize_log_numerator.x, *log_like_args)
        neglogden = minus_log_likelihood(minimize_log_denominator.x, *log_like_args)
        Zval = np.sqrt(2 * (neglognum - neglogden))

    p0 = 1 - norm.cdf(Zval)
    return p0, Zval


def plot_events_w_pvalue(xdata, ydata, yerr, ydata_fit, y_unc, plotfile='show', title=None):
    plt.fill_between(xdata, ydata_fit + y_unc, ydata_fit - y_unc, color='gray', alpha=0.4)
    plt.errorbar(xdata, ydata, yerr, None, 'bo', label='data', markersize=4)
    plt.plot(xdata, ydata_fit, 'r--', label='data')
    plt.yscale('log')
    plt.ylabel('Num events / 100 GeV')
    plt.xlabel('mJJ / GeV')
    if title:
        plt.title(title)
    if plotfile == 'show':
        plt.show()
    else:
        plt.savefig(plotfile)


class GetPValue(BackgroundParameterization):
    """
    This is a modified version of https://github.com/Jackadsa/CWoLa-Hunting/blob/master/code/cwola_utils.py
    I needed something more flexible
    Citation: https://arxiv.org/pdf/1805.02664.pdf, https://arxiv.org/pdf/1902.02634.pdf

    We'll use asymptotic formulae for p0 from Cowan et al arXiv:1007.1727
    and systematics procedure from https://cds.cern.ch/record/2242860/files/NOTE2017_001.pdf
    """
    def __init__(self, default_bin_width=100, xi=0., S=13000.):
        self.x_dbw = default_bin_width

        super().__init__(xi=xi, S=S)

    def get_p_value(self, ydata, binvals, mask, verbose=False, plotfile=None, yerr=None):
        ydata = np.array(ydata)
        # Assume poisson is gaussian with N+1 variance
        if not yerr:
            yerr = np.sqrt(ydata + 1)
        else:
            yerr = np.array(yerr)

        xdata = bin_means(binvals)
        xwidths = bin_widths(binvals)

        # Assuming inputs are bin counts, this is needed to get densities. Important for variable-width bins
        ydata = ydata * self.x_dbw / xwidths
        yerr = yerr * self.x_dbw / xwidths

        # Least square fit, masking out the signal region
        popt, pcov = curve_fit(self.fit_func, np.delete(xdata, mask), np.delete(ydata, mask), sigma=np.delete(yerr, mask), maxfev=3000, absolute_sigma=True)
        if verbose:
            logging.info('Fit params:')
            for param in popt:
                logging.info('{}'.format(param))

        ydata_fit = self.fit_func_array(popt, xdata)

        jac = Jacobian(self.fit_func_array)
        x_cov = np.dot(np.dot(jac(popt, xdata), pcov), jac(popt, xdata).T)
        # For plot, take systematic error band as the diagonal of the covariance matrix
        y_unc = np.sqrt([row[i] for i, row in enumerate(x_cov)])

        # First get systematics in the signal region
        def signal_fit_func_array(parr):
            # This function returns array of signal predictions in the signal region
            return np.sum(self.fit_func_array(parr, xdata[mask]) * xwidths[mask] / self.x_dbw)

        # Get covariance matrix of prediction uncertainties in the signal region
        jac_sig = Jacobian(signal_fit_func_array)
        x_signal_cov = np.dot(np.dot(jac_sig(popt), pcov), jac_sig(popt).T).item()

        # Get observed and predicted event counts in the signal region
        observed = np.sum(ydata[mask] * xwidths[mask] / self.x_dbw)
        expected = signal_fit_func_array(popt)
        if verbose:
            logging.info("Number of expected events = {}".format(expected))
            logging.info("Number of observed events = {}".format(observed))

        # Initialization of nuisance params
        num_nuis_arr_init = [0.02]
        # Set bounds for bg nuisance at around 8 sigma
        num_bounds = [[-8 * y_unc[mask[0]], 8 * y_unc[mask[0]]]]
        # initizalization for minimization
        den_nuis_arr_init = [0.01, 1.]

        # Get likelihood ratio, perform minimization over nuisance parameters
        pval_kwargs = {'x_signal_cov': x_signal_cov, 'num_nuis_arr_init': num_nuis_arr_init, 'num_bounds': num_bounds, 'den_nuis_arr_init': den_nuis_arr_init, 'verbose': verbose}
        p0, Zval = compute_p_value(observed, expected, **pval_kwargs)

        if verbose:
            logging.info("Zval = {}".format(Zval))
            logging.info("p0 = {}".format(p0))

        if plotfile:
            plot_events_w_pvalue(xdata, ydata, yerr, ydata_fit, y_unc, plotfile, p0)

        return p0, Zval
