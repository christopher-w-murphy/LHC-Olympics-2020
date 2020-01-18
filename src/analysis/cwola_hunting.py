import numpy as np
import pandas as pd

from .utils import get_histograms


def cwola_hunting(df, classifier, spliter, sideband_min, sideband_max, signal_region_min, signal_region_max, feature_cols=None, transformer=None, bin_width=50, clf_quantiles=[0.0, 0.7, 0.9]):
    sideband_region = (df['mass_jj'] > sideband_min-bin_width/2) & (df['mass_jj'] < sideband_max+bin_width/2)
    df_cwola = df[sideband_region].copy().reset_index(drop=True)
    signal_region = (df_cwola['mass_jj'] > signal_region_min) & (df_cwola['mass_jj'] < signal_region_max)
    df_cwola['signal_region'] = signal_region.astype('int64')

    if feature_cols is None:
        feature_cols = df.columns.drop(['mass_jj'])
    X = df_cwola[feature_cols]
    y = df_cwola['signal_region']

    df_clf = pd.DataFrame(columns=['index', 'xgb_score'])
    for tr_idx, te_idx in spliter.split(X, y):
        if transformer is not None:
            trans = transformer
            X_tr = trans.fit_transform(X.iloc[tr_idx])
            X_te = trans.transform(X.iloc[te_idx])
        else:
            X_tr = X.iloc[tr_idx]
            X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        clf = classifier
        clf.fit(X_tr, y_tr)
        df_clf = df_clf.append(pd.DataFrame(data={'index': te_idx, 'score': clf.predict_proba(X_te).T[1]}))

    df_te = df_cwola[['mass_jj']].merge(df_clf, how='left', left_index=True, right_on='index').drop(columns=['index'])
    clf_hist_bins = np.arange(sideband_min-bin_width/2, sideband_max+3*bin_width/2, bin_width)
    return get_histograms(df_te, clf_hist_bins, quantiles=clf_quantiles, feature='score')
