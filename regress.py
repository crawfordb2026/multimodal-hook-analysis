"""Phase 4b: predict continuous within-creator outperformance (regression).

Our own results showed the 3-bucket tercile label threw away ranking signal
(rank-acc beat tercile accuracy relative to chance). This regresses the
continuous target instead -- log(engagement / creator median) -- and adds
gradient boosting (XGBoost), which is often the strongest model on small
tabular/embedding data (recall our transformer lost to the MLP).

Same GroupKFold-by-creator protocol. Reported on pooled out-of-fold predictions:
    rank-acc     within-creator pairwise ranking (chance 0.50) -- the headline
    spearman     rank correlation of prediction vs true engagement
    tercile-acc  predictions re-bucketed into within-creator thirds vs true
                 terciles (comparable to the classification runs; chance ~0.33)

Usage:
    python regress.py
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from dataset import load_dataset
from train import within_creator_concordance

MODALITIES = ('clip', 'text', 'clap', 'audiometa')


def tercile_accuracy(pred, y_true, groups):
    """Re-bucket predictions into within-creator thirds, compare to true terciles."""
    correct = total = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 3:
            continue
        ranks = pd.Series(pred[idx]).rank(method='first')
        pred_tercile = pd.qcut(ranks, 3, labels=[0, 1, 2]).astype(int).to_numpy()
        correct += (pred_tercile == y_true[idx]).sum()
        total += len(idx)
    return correct / total


def evaluate(make_model, X, ds, n_splits=5):
    oof = np.zeros(len(ds))
    for train, test in GroupKFold(n_splits=n_splits).split(X, ds.outperf, ds.groups):
        model = make_model()
        model.fit(X[train], ds.outperf[train])
        oof[test] = model.predict(X[test])
    rank = within_creator_concordance(oof, ds.rate, ds.groups)
    rho = spearmanr(oof, ds.rate).statistic
    tacc = tercile_accuracy(oof, ds.y, ds.groups)
    return rank, rho, tacc


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ds = load_dataset()
    X = ds.flat(MODALITIES)
    print(f'{len(ds)} videos, {len(set(ds.groups))} creators | '
          f'target = continuous within-creator outperformance\n')

    models = {
        'Ridge (linear)': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'XGBoost': lambda: XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=2.0, n_jobs=-1),
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0),
    }

    print(f'{"model":<24}{"rank-acc":<12}{"spearman":<12}{"tercile-acc":<12}')
    print('-' * 60)
    for name, mk in models.items():
        rank, rho, tacc = evaluate(mk, X, ds)
        print(f'{name:<24}{rank:.3f}       {rho:+.3f}       {tacc:.3f}')
    print(f'\nchance: rank-acc 0.500, spearman 0.000, tercile-acc {1/3:.3f}')
    print('(compare rank-acc to run1 neural: MLP 0.559, transformer 0.551)')


if __name__ == '__main__':
    main()
