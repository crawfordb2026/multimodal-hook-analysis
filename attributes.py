"""Phase 4c: interpretable creative attributes of each hook.

Turns the opaque CLIP embeddings into human-readable creative features, then
asks which ones relate to engagement -- the "which hooks work" analysis.

Two cheap sources, no new heavy model:
  - CLIP zero-shot: score each video's frames against plain-English concepts
    (face, on-screen text, product close-up, ...) using the cached embeddings
    and CLIP's text encoder. Score = max cosine similarity over the 8 frames.
  - OpenCV: cut count and motion energy over the first 3s (what CLIP frames miss).

Writes data/attributes.parquet and prints each attribute's mean within-creator
Spearman correlation with engagement (chance 0.0), so the ranking reflects
"does more of X go with a creator's better posts", free of cross-creator/size
effects.

Usage:
    python attributes.py [--workers 8]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
FEATURE_DIR = ROOT / 'data' / 'features'
VIDEO_DIR = ROOT / 'data' / 'videos'
ATTR_PATH = ROOT / 'data' / 'attributes.parquet'
CLIP_MODEL = 'openai/clip-vit-base-patch32'

# interpretable creative concepts to probe (CLIP zero-shot)
ATTRIBUTES = {
    'face': "a close-up of a person's face",
    'text_overlay': 'a video frame with large text captions on screen',
    'product_closeup': 'a close-up of a drink can or beverage bottle',
    'talking_head': 'a person talking directly to the camera',
    'group': 'a group of several people together',
    'outdoor': 'an outdoor scene outside',
    'hands_product': 'hands holding or pouring a drink',
    'colorful': 'a bright, vibrant, colorful image',
    'food': 'food or a meal on a table',
}


def clip_scores(device: str) -> pd.DataFrame:
    """Max cosine similarity of each video's 8 frames to each attribute prompt."""
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    prompts = list(ATTRIBUTES.values())
    with torch.no_grad():
        tok = proc(text=prompts, return_tensors='pt', padding=True).to(device)
        txt = model.get_text_features(**tok)
        txt = getattr(txt, 'pooler_output', txt)
        txt = torch.nn.functional.normalize(txt, dim=1).cpu().numpy()  # [A,512]

    df = pd.read_parquet(METADATA_PATH)
    rows = []
    for code in df['shortcode']:
        frames = np.load(FEATURE_DIR / f'{code}.npz')['clip']  # [8,512]
        frames = frames / (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-9)
        sims = frames @ txt.T                                  # [8,A]
        rows.append(dict(zip(ATTRIBUTES, sims.max(axis=0), strict=True),
                         shortcode=code))
    return pd.DataFrame(rows)


def _motion_cuts(code: str, window: float = 3.0, size: int = 64):
    cap = cv2.VideoCapture(str(VIDEO_DIR / f'{code}.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(window * fps)
    prev, diffs, i = None, [], 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY)
        g = g.astype('float32') / 255.0
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev, i = g, i + 1
    cap.release()
    d = np.array(diffs) if diffs else np.array([0.0])
    return {'shortcode': code, 'motion_energy': float(d.mean()),
            'cut_count': int((d > 0.25).sum())}


def opencv_features(codes, workers: int) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_motion_cuts, codes))
    return pd.DataFrame(rows)


def within_creator_corr(values, rate, groups) -> float:
    corrs = []
    for g in np.unique(groups):
        idx = groups == g
        v, r = values[idx], rate[idx]
        if len(v) >= 5 and np.std(v) > 0 and np.std(r) > 0:
            corrs.append(spearmanr(v, r).statistic)
    return float(np.nanmean(corrs)) if corrs else float('nan')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    meta = pd.read_parquet(METADATA_PATH)[['shortcode', 'creator', 'engagement_rate']]

    print(f'scoring {len(meta)} videos: CLIP zero-shot ({device}) + OpenCV motion...')
    attrs = clip_scores(device)
    motion = opencv_features(meta['shortcode'].tolist(), args.workers)
    df = meta.merge(attrs, on='shortcode').merge(motion, on='shortcode')
    df.to_parquet(ATTR_PATH, index=False)
    print(f'wrote {ATTR_PATH.relative_to(ROOT)}\n')

    feature_cols = list(ATTRIBUTES) + ['motion_energy', 'cut_count']
    groups, rate = df['creator'].to_numpy(), df['engagement_rate'].to_numpy()
    ranked = sorted(
        ((c, within_creator_corr(df[c].to_numpy(), rate, groups)) for c in feature_cols),
        key=lambda x: x[1], reverse=True)

    print('attribute -> mean within-creator correlation with engagement:')
    print(f'{"attribute":<18}{"corr":>8}')
    print('-' * 26)
    for name, corr in ranked:
        print(f'{name:<18}{corr:+.3f}')
    print('\npositive = more of this attribute goes with a creator\'s better posts')


if __name__ == '__main__':
    main()
