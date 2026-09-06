"""Phase 4 baselines: linear models on the cached features, GroupKFold by creator.

Establishes the performance floor everything else is measured against, and
doubles as a first modality ablation (how much signal is in frames vs caption
vs audio, alone and combined). Folds hold out whole creators, so scores reflect
generalization to creators the model has never seen.

Chance is ~0.33 (three within-creator terciles, balanced by construction);
DummyClassifier gives the empirical floor.

Usage:
    python baselines.py            # 5-fold GroupKFold, all modality combos
    python baselines.py --folds 6
"""

import argparse

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import load_dataset

MODALITY_SETS = [
    ('clip',),
    ('text',),
    ('clap',),
    ('clip', 'text'),
    ('clip', 'clap'),
    ('text', 'clap'),
    ('clip', 'text', 'clap'),
]


def evaluate(make_model, X, y, groups, n_splits):
    """Return (acc_mean, acc_std, f1_mean, f1_std) over GroupKFold."""
    accs, f1s = [], []
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model()
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        accs.append(accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, average='macro'))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    ds = load_dataset()
    print(f'{len(ds)} videos, {len(set(ds.groups))} creators, '
          f'{args.folds}-fold GroupKFold by creator\n')

    def logreg():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    rows = []
    # empirical floors
    for name, strat in [('most-frequent', 'most_frequent'), ('stratified', 'stratified')]:
        acc, acc_sd, f1, f1_sd = evaluate(
            lambda s=strat: DummyClassifier(strategy=s, random_state=0),
            ds.flat(('clip',)), ds.y, ds.groups, args.folds)
        rows.append((name, acc, acc_sd, f1, f1_sd))

    # logistic regression per modality combo
    for mods in MODALITY_SETS:
        X = ds.flat(mods)
        acc, acc_sd, f1, f1_sd = evaluate(logreg, X, ds.y, ds.groups, args.folds)
        rows.append(('+'.join(mods), acc, acc_sd, f1, f1_sd))

    print(f'{"model":<22}{"accuracy":<18}{"macro-F1":<18}')
    print('-' * 58)
    for name, acc, acc_sd, f1, f1_sd in rows:
        print(f'{name:<22}{acc:.3f} ± {acc_sd:.3f}     {f1:.3f} ± {f1_sd:.3f}')
    print(f'\nchance ≈ {1/3:.3f}   (n={len(ds)}, {len(set(ds.groups))} creators)')


if __name__ == '__main__':
    main()
