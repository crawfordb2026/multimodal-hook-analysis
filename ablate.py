"""Phase 4 ablations on the best model (late-fusion MLP), averaged over seeds.

Two questions, one GroupKFold-by-creator protocol:

1. Modality ablation: how much does each of frames / caption / audio contribute,
   alone and combined? Reported as mean +/- std over SEEDS random seeds.

2. Silent-video check: audio should help where there is real audio and not where
   there isn't. Compares the full model (clip+text+clap) against the no-audio
   model (clip+text) separately on the videos that HAVE an audio track vs the
   125 silent ones (whose CLAP feature is just synthesized silence).

Usage:
    python ablate.py                    # 5 seeds, 5-fold
    python ablate.py --seeds 3 --epochs 60
"""

import argparse

import numpy as np

from dataset import load_dataset
from train import pick_device, train_eval

MODALITY_SETS = [
    ('clip',), ('text',), ('clap',),
    ('clip', 'text'), ('clip', 'clap'), ('text', 'clap'),
    ('clip', 'text', 'clap'),
]


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
    print(f'{len(ds)} videos, {len(set(ds.groups))} creators | device {device} | '
          f'MLP, {args.folds}-fold, {args.seeds} seeds\n')

    oof_by_mods = {}  # for the silent-video breakdown
    print(f'{"modalities":<22}{"accuracy":<18}{"macro-F1":<18}{"rank-acc":<16}')
    print('-' * 74)
    for mods in MODALITY_SETS:
        accs, f1s, concords, preds = [], [], [], []
        for seed in seeds:
            fa, ff, cc, oof_pred, _ = train_eval(
                ds, 'mlp', args.folds, args.epochs, 1e-3, 1e-4, device, seed, mods)
            accs.append(np.mean(fa))
            f1s.append(np.mean(ff))
            concords.append(cc)
            preds.append(oof_pred)
        oof_by_mods[mods] = np.array(preds)  # [seeds, N]
        print(f'{"+".join(mods):<22}'
              f'{np.mean(accs):.3f} ± {np.std(accs):.3f}     '
              f'{np.mean(f1s):.3f} ± {np.std(f1s):.3f}     '
              f'{np.mean(concords):.3f} ± {np.std(concords):.3f}')

    # silent-video breakdown: full vs no-audio, on real-audio rows vs silent rows
    full = oof_by_mods[('clip', 'text', 'clap')]
    noaud = oof_by_mods[('clip', 'text')]

    def subset_acc(preds, mask):
        return np.mean([(preds[s][mask] == ds.y[mask]).mean() for s in range(len(preds))])

    has_audio = ~ds.silent
    print(f'\naudio ablation by track presence (accuracy, {args.seeds} seeds):')
    print(f'{"subset":<26}{"clip+text":<14}{"+audio":<14}{"delta":<8}')
    print('-' * 62)
    for name, mask in [(f'has audio (n={has_audio.sum()})', has_audio),
                       (f'silent (n={ds.silent.sum()})', ds.silent)]:
        a0, a1 = subset_acc(noaud, mask), subset_acc(full, mask)
        print(f'{name:<26}{a0:.3f}         {a1:.3f}         {a1 - a0:+.3f}')
    print('\nexpect audio to help on has-audio rows and not on silent rows')


if __name__ == '__main__':
    main()
