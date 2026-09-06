"""Does adding the song-type signal (original vs licensed audio) help?

CLAP encodes what the audio sounds like, not which song it is. This adds a
3-way one-hot (original / licensed / unknown) as an extra feature and re-runs
the same GroupKFold-by-creator evaluation, to test whether "song choice" — the
hypothesis that near-identical videos differ mainly by their music — carries
signal our acoustic features miss.

Compares, mean +/- std over seeds:
    audiometa alone            how much signal is in song-type by itself
    MLP  clip+text+clap        vs  + audiometa
    transformer 10-token       vs  11-token (+ audiometa)

Usage:
    python audio_test.py            # 5 seeds, 5-fold
"""

import argparse

import numpy as np

from dataset import load_dataset
from train import pick_device, train_eval


def run(ds, kind, modalities, seeds, folds, epochs, lr, wd, device):
    accs, f1s, cc = [], [], []
    for seed in seeds:
        fa, ff, c, _, _ = train_eval(ds, kind, folds, epochs, lr, wd, device,
                                     seed, modalities)
        accs.append(np.mean(fa))
        f1s.append(np.mean(ff))
        cc.append(c)
    return np.mean(accs), np.std(accs), np.mean(f1s), cc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    args = parser.parse_args()

    ds = load_dataset()
    device = pick_device(args.device)
    seeds = list(range(args.seeds))
    n_orig = int(ds.audio_meta[:, 0].sum())
    n_lic = int(ds.audio_meta[:, 1].sum())
    n_unk = int(ds.audio_meta[:, 2].sum())
    print(f'{len(ds)} videos | audio type: {n_orig} original, {n_lic} licensed, '
          f'{n_unk} unknown | device {device}, {args.seeds} seeds\n')

    runs = [
        ('audiometa alone (MLP)', 'mlp', ('audiometa',), 1e-3, 1e-4),
        ('MLP clip+text+clap', 'mlp', ('clip', 'text', 'clap'), 1e-3, 1e-4),
        ('MLP  + audiometa', 'mlp', ('clip', 'text', 'clap', 'audiometa'), 1e-3, 1e-4),
        ('transformer 10-token', 'transformer', ('clip', 'text', 'clap'), 5e-4, 1e-4),
        ('transformer 11-token', 'transformer',
         ('clip', 'text', 'clap', 'audiometa'), 5e-4, 1e-4),
    ]

    print(f'{"model":<26}{"accuracy":<18}{"macro-F1":<12}{"rank-acc":<10}')
    print('-' * 66)
    for name, kind, mods, lr, wd in runs:
        acc, acc_sd, f1, cc = run(ds, kind, mods, seeds, args.folds, args.epochs,
                                  lr, wd, device)
        print(f'{name:<26}{acc:.3f} ± {acc_sd:.3f}     {f1:.3f}      '
              f'{np.mean(cc):.3f}')
    print(f'\nchance: acc ≈ {1/3:.3f}, rank-acc = 0.500')


if __name__ == '__main__':
    main()
