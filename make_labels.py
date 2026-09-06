"""Generate within-creator engagement tercile labels.

Drops creators with too few videos to split reliably (< MIN_POSTS_PER_CREATOR),
then splits each remaining creator's videos into thirds by engagement_rate
((likes + comments) / plays) and writes a 'label' column (low / mid / high)
back into data/metadata.parquet. Finally reconciles data/videos/ so it holds
exactly the mp4s referenced by the final dataset (orphans from dropped posts or
creators are deleted).

Run this last, after ingest.py + boost_filter.py, and re-run it any time the
data changes — labels are relative to whatever videos survive the filters.

Usage:
    python make_labels.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
VIDEO_DIR = ROOT / 'data' / 'videos'

# a creator needs at least this many videos to form meaningful terciles;
# below this the low/mid/high split is statistical noise, so drop the creator
MIN_POSTS_PER_CREATOR = 10


def main() -> None:
    if not METADATA_PATH.exists():
        sys.exit('no metadata.parquet - run ingest.py first')
    df = pd.read_parquet(METADATA_PATH)

    # a row with no downloaded mp4 (expired/failed URL) can't be preprocessed or
    # encoded, so it can't be modeled — drop before labeling
    no_video = df['video_path'].isna()
    if no_video.any():
        print(f'dropping {int(no_video.sum())} videos with no downloaded mp4: '
              f'{df.loc[no_video, "shortcode"].tolist()}')
        df = df[~no_video].reset_index(drop=True)

    counts = df['creator'].value_counts()
    too_small = counts[counts < MIN_POSTS_PER_CREATOR]
    if len(too_small):
        print(f'dropping {len(too_small)} creator(s) with < {MIN_POSTS_PER_CREATOR} '
              f'videos: {dict(too_small)}')
        df = df[~df['creator'].isin(too_small.index)].reset_index(drop=True)

    # rank before qcut so tied engagement_rate values can't collapse the bin
    # edges — guarantees three roughly equal groups per creator
    df['label'] = (
        df.groupby('creator')['engagement_rate']
        .transform(lambda s: pd.qcut(s.rank(method='first'), 3,
                                     labels=['low', 'mid', 'high']))
    )
    df.to_parquet(METADATA_PATH, index=False)

    kept_codes = set(df['shortcode'])
    orphans = [p for p in VIDEO_DIR.glob('*.mp4') if p.stem not in kept_codes]
    for p in orphans:
        p.unlink()
    if orphans:
        print(f'reconciled videos/: deleted {len(orphans)} orphaned mp4(s)')

    print(f'\nlabeled {len(df)} videos across {df["creator"].nunique()} creators')
    print(df.groupby(['creator', 'label'], observed=True).size().unstack(fill_value=0))


if __name__ == '__main__':
    main()
