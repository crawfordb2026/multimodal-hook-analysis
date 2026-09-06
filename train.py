"""Phase 4 neural models: late-fusion MLP and a 10-token transformer.

Both are evaluated with the same GroupKFold-by-creator protocol as the
baselines, on cached features, and reported three ways:

    accuracy / macro-F1   3-way tercile classification (chance ~0.33)
    within-creator rank   given two of a creator's videos, does the model
                          order them by true engagement? (chance 0.50)

The ranking metric is the honest, more useful question for a hard task: not
"which absolute tercile" but "which of this creator's hooks does better". It's
computed on pooled out-of-fold predictions so every creator is scored once,
never having been seen in training.

The transformer's 10 tokens are the 8 CLIP frame embeddings + 1 caption token
+ 1 audio token, each projected to a shared width with a learned modality-type
embedding, then a small transformer encoder and a mean-pooled classifier head.

Usage:
    python train.py                     # MLP + transformer, 5-fold
    python train.py --folds 6 --epochs 80 --device cpu
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from dataset import load_dataset


def pick_device(requested: str) -> str:
    if requested != 'auto':
        return requested
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def within_creator_concordance(scores, rate, groups) -> float:
    """Fraction of same-creator video pairs ordered correctly by score."""
    correct = total = 0.0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        s, r = scores[idx], rate[idx]
        ds = s[:, None] - s[None, :]
        dr = r[:, None] - r[None, :]
        pairs = np.triu(np.ones_like(ds, dtype=bool), 1) & (dr != 0)
        correct += (np.sign(ds) == np.sign(dr))[pairs].sum()
        total += pairs.sum()
    return correct / total if total else float('nan')


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x):
        return self.net(x)


class MMTransformer(nn.Module):
    """10 tokens (8 frames + caption + audio) -> encoder -> pooled head."""

    def __init__(self, d: int = 128, nhead: int = 4, layers: int = 2, p: float = 0.3,
                 use_audio_meta: bool = False):
        super().__init__()
        self.clip_proj = nn.Linear(512, d)
        self.text_proj = nn.Linear(384, d)
        self.clap_proj = nn.Linear(512, d)
        # token type embeddings: frame / text / audio / audio-metadata
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        self.meta_proj = nn.Linear(3, d) if use_audio_meta else None
        enc = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4 * d,
                                         dropout=p, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 3))

    def forward(self, clip, text, clap, audio_meta=None):
        f = self.clip_proj(clip) + self.type_emb[0]              # [B,8,d]
        t = self.text_proj(text).unsqueeze(1) + self.type_emb[1]  # [B,1,d]
        a = self.clap_proj(clap).unsqueeze(1) + self.type_emb[2]  # [B,1,d]
        tokens = [f, t, a]
        if audio_meta is not None and self.meta_proj is not None:
            m = self.meta_proj(audio_meta).unsqueeze(1) + self.type_emb[3]  # [B,1,d]
            tokens.append(m)                                     # -> 11 tokens
        h = self.encoder(torch.cat(tokens, dim=1)).mean(dim=1)
        return self.head(self.norm(h))


def fit_scalers(ds, train):
    """Per-modality StandardScalers fit on the training rows only."""
    sc = {
        'clip': StandardScaler().fit(ds.clip[train].reshape(-1, 512)),
        'text': StandardScaler().fit(ds.text[train]),
        'clap': StandardScaler().fit(ds.clap[train]),
    }
    return sc


def make_inputs(ds, idx, scalers, kind, device, modalities):
    clip = scalers['clip'].transform(ds.clip[idx].reshape(-1, 512)).reshape(len(idx), 8, 512)
    text = scalers['text'].transform(ds.text[idx])
    clap = scalers['clap'].transform(ds.clap[idx])
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    if kind == 'mlp':
        parts = {'clip': clip.mean(axis=1), 'text': text, 'clap': clap,
                 'audiometa': ds.audio_meta[idx]}
        flat = np.concatenate([parts[m] for m in modalities], axis=1)
        return (t(flat),)
    inputs = [t(clip), t(text), t(clap)]
    if 'audiometa' in modalities:
        inputs.append(t(ds.audio_meta[idx]))  # 11th token
    return tuple(inputs)


def train_eval(ds, kind, folds, epochs, lr, wd, device, seed=0,
               modalities=('clip', 'text', 'clap')):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = len(ds)
    oof_pred = np.zeros(N, dtype=int)
    oof_score = np.zeros(N, dtype=float)
    fold_acc, fold_f1 = [], []

    for train, test in GroupKFold(n_splits=folds).split(ds.clip, ds.y, ds.groups):
        scalers = fit_scalers(ds, train)
        Xtr = make_inputs(ds, train, scalers, kind, device, modalities)
        Xte = make_inputs(ds, test, scalers, kind, device, modalities)
        ytr = torch.tensor(ds.y[train], device=device)

        model = (MLP(Xtr[0].shape[1]) if kind == 'mlp'
                 else MMTransformer(use_audio_meta='audiometa' in modalities)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(train), device=device)
            for s in range(0, len(train), 128):
                b = perm[s:s + 128]
                opt.zero_grad()
                out = model(*(x[b] for x in Xtr))
                loss_fn(out, ytr[b]).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            probs = model(*Xte).softmax(dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        score = probs @ np.array([0, 1, 2])  # expected tercile, for ranking
        oof_pred[test] = pred
        oof_score[test] = score
        fold_acc.append(accuracy_score(ds.y[test], pred))
        fold_f1.append(f1_score(ds.y[test], pred, average='macro'))

    concord = within_creator_concordance(oof_score, ds.rate, ds.groups)
    return fold_acc, fold_f1, concord, oof_pred, oof_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    args = parser.parse_args()

    ds = load_dataset()
    device = pick_device(args.device)
    print(f'{len(ds)} videos, {len(set(ds.groups))} creators | device {device} | '
          f'{args.folds}-fold GroupKFold\n')

    configs = [
        ('late-fusion MLP', 'mlp', 1e-3, 1e-4),
        ('10-token transformer', 'transformer', 5e-4, 1e-4),
    ]
    print(f'{"model":<24}{"accuracy":<18}{"macro-F1":<18}{"rank-acc":<10}')
    print('-' * 70)
    for name, kind, lr, wd in configs:
        acc, f1, concord, _, _ = train_eval(ds, kind, args.folds, args.epochs, lr, wd, device)
        print(f'{name:<24}{np.mean(acc):.3f} ± {np.std(acc):.3f}     '
              f'{np.mean(f1):.3f} ± {np.std(f1):.3f}     {concord:.3f}')
    print(f'\nchance: accuracy ≈ {1/3:.3f}, rank-acc = 0.500')


if __name__ == '__main__':
    main()
