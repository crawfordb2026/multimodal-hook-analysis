"""Screen for paid-boosted posts per creator, using two conditions.

A boosted post buys reach from non-followers: plays balloon while likes don't,
so it shows up as BOTH a low like-rate AND an abnormally high play count for
that creator. Requiring both conditions is what separates bought reach from a
creator's genuinely weak posts (normal plays, low engagement) — the latter are
the real "low" tercile and must be kept.

A post is flagged as boosted when, within its creator:
    like_rate < max(floor, median - 2*MAD)      (low engagement), AND
    plays     > PLAY_MULTIPLE * median(plays)    (abnormal reach)

Usage:
    python boost_filter.py                 # analyze: per-creator summary + plots
    python boost_filter.py --apply         # drop flagged posts per creator
    python boost_filter.py --apply --floor 0.01 --play-multiple 8

Apply mode permanently drops the flagged posts, deletes their mp4s, records
them in dropped.csv, and prints the overall filter funnel.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
DROPPED_PATH = ROOT / 'data' / 'dropped.csv'
VIDEO_DIR = ROOT / 'data' / 'videos'
PLOT_DIR = ROOT / 'data' / 'plots'

DEFAULT_FLOOR = 0.008     # like-rate floor: flag below max(this, median - 2*MAD)
DEFAULT_PLAY_MULTIPLE = 5  # and only if plays exceed this * the creator's median


def like_rate_cutoff(rates: pd.Series, floor: float) -> float:
    med = rates.median()
    mad = (rates - med).abs().median()
    return max(floor, med - 2 * mad)


def flag_boosted(grp: pd.DataFrame, floor: float, play_multiple: float):
    """Boolean mask of boosted posts in one creator's group, plus the cutoffs."""
    cutoff = like_rate_cutoff(grp['like_rate'], floor)
    play_gate = play_multiple * grp['plays'].median()
    mask = (grp['like_rate'] < cutoff) & (grp['plays'] > play_gate)
    return mask, cutoff, play_gate


def analyze(df: pd.DataFrame, floor: float, play_multiple: float) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for creator, grp in df.groupby('creator'):
        mask, cutoff, play_gate = flag_boosted(grp, floor, play_multiple)
        low_rate_only = int((grp['like_rate'] < cutoff).sum())
        summary.append({
            'creator': creator, 'n': len(grp),
            'median_rate': round(grp['like_rate'].median(), 4),
            'rate_cutoff': round(cutoff, 4),
            'play_gate': int(play_gate),
            'low_rate': low_rate_only,      # would-be drops on like-rate alone
            'boosted': int(mask.sum()),     # actual drops (also above play gate)
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(grp['plays'], grp['like_rate'], s=18, alpha=0.6)
        ax.scatter(grp['plays'][mask], grp['like_rate'][mask], s=40,
                   color='red', label='flagged boosted')
        ax.axhline(cutoff, color='red', linestyle='--', linewidth=0.8,
                   label=f'rate cutoff {cutoff:.3f}')
        ax.axvline(play_gate, color='purple', linestyle=':', linewidth=0.8,
                   label=f'play gate {play_gate:,.0f}')
        ax.set_xscale('log')
        ax.set_xlabel('plays (log)')
        ax.set_ylabel('like rate')
        ax.set_title(f'@{creator} (n={len(grp)})')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'boost_{creator}.png', dpi=150)
        plt.close(fig)

    table = pd.DataFrame(summary).sort_values('boosted', ascending=False)
    print(f'per-creator boost screen (floor={floor}, play_multiple={play_multiple}x), '
          f'{len(table)} creators:')
    print(table.to_string(index=False))
    saved = int(table['low_rate'].sum() - table['boosted'].sum())
    print(f'\nboosted (dropped): {table["boosted"].sum()} of {len(df)}   '
          f'weak posts kept by the play gate: {saved}')
    print(f'plots -> {PLOT_DIR.relative_to(ROOT)}/boost_<creator>.png')
    print('\nrun with --apply to drop the flagged posts')


def apply_cutoffs(df: pd.DataFrame, floor: float, play_multiple: float) -> None:
    keep_mask = pd.Series(True, index=df.index)
    per_creator = []
    for creator, grp in df.groupby('creator'):
        mask, cutoff, _ = flag_boosted(grp, floor, play_multiple)
        keep_mask.loc[grp.index[mask]] = False
        if mask.any():
            per_creator.append((creator, cutoff, int(mask.sum())))

    boosted = df[~keep_mask]
    kept = df[keep_mask].reset_index(drop=True)
    if boosted.empty:
        print('nothing flagged; no changes made')
        return

    dropped = boosted.copy()
    dropped['drop_reason'] = 'boosted'
    if DROPPED_PATH.exists():
        dropped = pd.concat([pd.read_csv(DROPPED_PATH), dropped], ignore_index=True)
    dropped.to_csv(DROPPED_PATH, index=False)

    for code in boosted['shortcode']:
        (VIDEO_DIR / f'{code}.mp4').unlink(missing_ok=True)

    kept.to_parquet(METADATA_PATH, index=False)
    print(f'dropped {len(boosted)} boosted posts across {len(per_creator)} creators '
          f'(mp4s deleted), {len(kept)} videos remain\n')
    for creator, cutoff, n in sorted(per_creator, key=lambda x: -x[2]):
        print(f'  {n:>2} dropped  @{creator} (rate cutoff {cutoff:.4f})')
    print_funnel(kept)


def print_funnel(kept: pd.DataFrame) -> None:
    reasons = pd.read_csv(DROPPED_PATH)['drop_reason'] if DROPPED_PATH.exists() \
        else pd.Series(dtype=str)
    n_leak = int((reasons == 'collab_leakin').sum())
    n_collab = int(reasons.isin(['coauthored', 'paid_partnership']).sum())
    n_hidden = int((reasons == 'hidden_likes').sum())
    n_boost = int(reasons.str.startswith('boosted').sum())
    n_floor = int(reasons.str.startswith(('under_', 'missing_play')).sum())
    n_other = int(len(reasons) - n_leak - n_collab - n_hidden - n_boost - n_floor)
    total = len(kept) + len(reasons)

    print('\nfilter funnel:')
    print(f'  {total:>4}  scraped OK')
    print(f'  -{n_leak:>3}  collab leak-in (owner not on creators list)')
    print(f'  -{n_collab:>3}  coauthored / paid partnership')
    print(f'  -{n_hidden:>3}  hidden likes (unlabelable)')
    print(f'  -{n_boost:>3}  boosted (low like rate + abnormal plays)')
    print(f'  -{n_floor:>3}  under play-count floor / missing plays')
    if n_other:
        print(f'  -{n_other:>3}  other')
    print(f'  {len(kept):>4}  final')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true',
                        help='drop flagged boosted posts per creator')
    parser.add_argument('--floor', type=float, default=DEFAULT_FLOOR,
                        help=f'like-rate floor (default {DEFAULT_FLOOR})')
    parser.add_argument('--play-multiple', type=float, default=DEFAULT_PLAY_MULTIPLE,
                        help=f'plays must exceed this * creator median '
                             f'(default {DEFAULT_PLAY_MULTIPLE})')
    args = parser.parse_args()

    if not METADATA_PATH.exists():
        sys.exit('no metadata.parquet - run ingest.py first')
    df = pd.read_parquet(METADATA_PATH)

    if args.apply:
        apply_cutoffs(df, args.floor, args.play_multiple)
    else:
        analyze(df, args.floor, args.play_multiple)


if __name__ == '__main__':
    main()
