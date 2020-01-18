import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class TestOutlierDetector():
    def __init__(self):
        self.targets = ['is_signal', 'mass_jj']

    def create_testing_dataframe(self, df, feature_cols=None, sample_frac=1.):
        # optinally use only a subset of features
        if feature_cols:
            df_sb = df[feature_cols + self.targets].copy()
        else:
            df_sb = df.copy()
        # optionally further reduce signal-to-background ratio from default of 1:10
        if (sample_frac < 1.) & (sample_frac > 0.):
            df_s = df_sb[df_sb['is_signal'] == 1].sample(frac=sample_frac)
            df_b = df_sb[df_sb['is_signal'] == 0]
            df_sb = df_s.append(df_b).sample(frac=1).reset_index(drop=True)
        return df_sb

    def score_outlier_detector(self, df, outlier_detector, spliter, transformer=None, score_name='score'):
        # get features and targets
        X=df.drop(columns = self.targets)
        y=df['is_signal']
        # create df to store results
        df_split_scores=pd.DataFrame(columns = [score_name, 'index'])
        # cross validation
        for tr_idx, te_idx in spliter.split(X, y):
            # optionally apply feature transformation
            if transformer:
                X_tr=transformer.fit_transform(X.iloc[tr_idx])
                X_te=transformer.transform(X.iloc[te_idx])
            else:
                X_tr=X.iloc[tr_idx]
                X_te=X.iloc[te_idx]
            # intialize and fit model
            od=outlier_detector
            od.fit(X_tr)
            # get results
            df_split_score=pd.DataFrame(data = {score_name: od.decision_function(X_te), 'index': te_idx})
            df_split_scores=df_split_scores.append(df_split_score)
        # merge resuls with other necessary info
        result=pd.merge(df[self.targets], df_split_score, how = 'left', left_index = True, right_on = 'index')
        return result.drop(columns = ['index'])

    def test_outlier_detector(self, df, outlier_detector, spliter, transformer=None, feature_cols=None, sample_frac=1.):
        # create signal+background dataframe
        df_sb=self.create_testing_dataframe(df, feature_cols = feature_cols, sample_frac = sample_frac)
        # and background only dataframe
        df_b=df_sb[df_sb['is_signal'] == 0]
        # compute scores for signal+background
        sb_scores=self.score_outlier_detector(df_sb, outlier_detector, spliter, transformer = transformer, score_name = 'score_sb')
        # and background only
        b_scores=self.score_outlier_detector(df_b, outlier_detector, spliter, transformer = transformer, score_name = 'score_b')
        # delete unnecessary data
        del df_sb, df_b
        # return scores
        return (sb_scores, b_scores)

    def get_bin_edges(self, df, ntile=0.01, bin_width=100):
        # create bins, default width = 100 GeV
        left = np.round(df['mass_jj'].quantile(ntile), -2)
        right = np.round(df['mass_jj'].quantile(1 - ntile), -2)
        return np.arange(left - bin_width / 2, right + 3 * bin_width / 2, bin_width)

    def bin_means(self, bins):
        return [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

    def get_histogram(self, df, bins, q=0.0, feature='score'):
        # select the 1 - q fraction of events with the highest outlier scores
        qcut = df[feature] > df[feature].quantile(q)
        # create historgram of selected scores
        return np.histogram(df[qcut]['mass_jj'], bins=bins)

    def get_histograms(self, sb_scores, b_scores, quantiles=[0.0, 0.7, 0.9], ntile=0.01):
        # create bins for histograms
        hist_bins = self.get_bin_edges(sb_scores, ntile=ntile)
        # signal+background histogram
        hist_sb = [self.get_histogram(sb_scores, hist_bins, q=q, feature='score_sb') for q in quantiles]
        # background histogram
        hist_b = [self.get_histogram(b_scores, hist_bins, q=q, feature='score_b') for q in quantiles]
        return (hist_sb, hist_b)

    def plot_outlier_detections(self, sb_scores_list, b_scores_list, plot_signal_region=1, quantiles=[0.0, 0.7, 0.9], figsize=(9, 9), title_list=None, ntile=0.01, file_name=None, file_types=['.png']):
        n = len(sb_scores_list)
        plt.figure(figsize=figsize)

        for i in range(n):
            plt.subplot(int(np.ceil(n/2)), 2, i+1)
            sb_scores = sb_scores_list[i]
            b_scores = b_scores_list[i]

            # ratio of # of signal+background events to just background events
            r = len(sb_scores) / len(b_scores)
            # get score histograms
            hist_sb, hist_b = self.get_histograms(sb_scores, b_scores, quantiles=quantiles, ntile=ntile)

            # plot background only scores
            for hist in hist_b:
                plt.plot(self.bin_means(hist[1]), r * hist[0], color='r', linewidth=1)
            # plot signal+background scores
            for hist in hist_sb:
                plt.scatter(self.bin_means(hist[1]), hist[0], marker='.', color='b')
            # plot true signal histogram
            plt.hist(sb_scores[sb_scores['is_signal'] == 1]['mass_jj'], bins=hist_sb[0][1], alpha=0.4, color='y')
            # find the center of the signal region
            sig_maxs = argrelextrema(hist_sb[plot_signal_region][0], np.greater)[0]
            bkg_maxs = argrelextrema(hist_b[plot_signal_region][0], np.greater)[0]
            # the background should only have one peak
            sig_maxs_sorted = sorted(sig_maxs, key=lambda x: (x - bkg_maxs[0])**2)
            # find the mass associated with that index
            keys = sig_maxs_sorted[1:]
            masses = [self.bin_means(hist_sb[plot_signal_region][1][key:key+2]) for key in keys]
            # indicate signal region with dashed lines
            plt.vlines(masses, 10**1, 10**6, linestyles='dashed')
            # label plot
            if title_list is not None:
                plt.title(title_list[i])
            plt.xlabel(r'$m_{JJ} /\rm\, GeV$')
            binwidth = int(hist_sb[plot_signal_region][1][1] - hist_sb[plot_signal_region][1][0])
            plt.ylabel(r'$\rm Events\, /\, %i\, GeV$' %binwidth)
            plt.yscale('log')
            plt.ylim(10**1, 10**6)

        plt.tight_layout(True)
        if file_name:
            for file_type in file_types:
                plt.savefig(file_name + file_type)
        plt.show()
