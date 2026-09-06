"""Assemble the cached features + labels into arrays for modeling.

Loads every data/features/<shortcode>.npz alongside metadata.parquet and
returns aligned arrays. Labels are the within-creator engagement terciles
(low/mid/high -> 0/1/2); groups are creator usernames, so GroupKFold can hold
out whole creators and no creator leaks across train/test.

    clip        [N, 8, 512]   per-frame CLIP features (8 tokens)
    text        [N, 384]      MiniLM caption feature
    clap        [N, 512]      CLAP audio feature
    audio_meta  [N, 3]        one-hot: [original, licensed, unknown] audio type
    y           [N]           tercile label 0/1/2
    rate        [N]           raw engagement_rate (for within-creator ranking)
    groups      [N]           creator username per row
    silent      [N]           bool, True where the video had no audio track

audio_meta is the song-type signal CLAP can't see: whether a video uses the
creator's original audio or a licensed/trending song (unknown where the scraper
didn't record it). Use it via .flat(('...', 'audiometa')).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
FEATURE_DIR = ROOT / 'data' / 'features'
SILENT_PATH = ROOT / 'data' / 'silent_videos.csv'
LABELS = ['low', 'mid', 'high']


@dataclass
class Dataset:
    clip: np.ndarray
    text: np.ndarray
    clap: np.ndarray
    audio_meta: np.ndarray
    y: np.ndarray
    rate: np.ndarray
    outperf: np.ndarray
    groups: np.ndarray
    silent: np.ndarray
    shortcodes: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def flat(self, modalities: tuple[str, ...]) -> np.ndarray:
        """Concatenate the requested modalities into a flat [N, D] matrix.

        CLIP's 8 frame tokens are mean-pooled for the flat (non-sequential)
        baselines; the transformer later uses all 8 separately.
        """
        parts = {'clip': self.clip.mean(axis=1), 'text': self.text,
                 'clap': self.clap, 'audiometa': self.audio_meta}
        return np.concatenate([parts[m] for m in modalities], axis=1)


def load_dataset() -> Dataset:
    df = pd.read_parquet(METADATA_PATH).sort_values('shortcode').reset_index(drop=True)
    if df['label'].isna().any():
        raise ValueError('metadata has unlabeled rows - run make_labels.py')

    clip, text, clap = [], [], []
    for code in df['shortcode']:
        d = np.load(FEATURE_DIR / f'{code}.npz')
        clip.append(d['clip'])
        text.append(d['text'])
        clap.append(d['clap'])

    silent_codes = set()
    if SILENT_PATH.exists():
        silent_codes = set(pd.read_csv(SILENT_PATH)['shortcode'])
    silent = df['shortcode'].isin(silent_codes).to_numpy()

    # mask audio for silent videos: a video with no audio track has no real
    # sound, so its CLAP vector (encoded from synthesized silence) is noise.
    # zero it out rather than feed a misleading "silence" embedding.
    clap = np.stack(clap).astype('float32')
    clap[silent] = 0.0

    # continuous within-creator outperformance target = log(rate / creator
    # median rate). positive => the video beat that creator's typical post.
    # replaces the 3-bucket tercile label, which discarded ranking signal.
    er = df['engagement_rate'].to_numpy().astype('float32')
    med = df.groupby('creator')['engagement_rate'].transform('median').to_numpy()
    outperf = np.log((er + 1e-9) / (med + 1e-9)).astype('float32')

    # one-hot audio type from uses_original_audio (True/False/None -> 3 columns)
    uoa = df['uses_original_audio']
    audio_meta = np.zeros((len(df), 3), dtype='float32')
    audio_meta[uoa.eq(True).to_numpy(), 0] = 1     # original
    audio_meta[uoa.eq(False).to_numpy(), 1] = 1    # licensed
    audio_meta[~uoa.isin([True, False]).to_numpy(), 2] = 1  # unknown

    y = pd.Categorical(df['label'], categories=LABELS, ordered=True).codes
    return Dataset(
        clip=np.stack(clip).astype('float32'),
        text=np.stack(text).astype('float32'),
        clap=clap,
        audio_meta=audio_meta,
        y=y.astype('int64'),
        rate=er,
        outperf=outperf,
        groups=df['creator'].to_numpy(),
        silent=silent,
        shortcodes=df['shortcode'].to_numpy(),
    )


if __name__ == '__main__':
    ds = load_dataset()
    print(f'{len(ds)} videos, {len(set(ds.groups))} creators')
    print(f'clip {ds.clip.shape}  text {ds.text.shape}  clap {ds.clap.shape}')
    print(f'label balance: {np.bincount(ds.y)}  (low/mid/high)')
    print(f'silent: {ds.silent.sum()}')
