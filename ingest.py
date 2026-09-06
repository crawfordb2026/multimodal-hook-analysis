"""Ingest Apify Instagram reel scraper exports into the modeling dataset.

Reads every JSON export in data/raw_json/, flattens the records into one row
per reel, applies the permanent quality filters, downloads the mp4s, and
maintains a single deduplicated data/metadata.parquet across runs.

A video's creator is its ownerUsername. The real creators are defined by the
master list in data/creators.txt (one username per line); this file is
required. Ingest flags any listed creator missing from the scrape and warns
about any heavily-posting owner that is NOT on the list (a likely forgotten
creator). Collab posts owned by other accounts are dropped as collab_leakin.

Permanent filters (rows land in dropped.csv with the reason):
    collab_leakin      owner is not on the creators list
    not_a_reel         productType != 'clips'
    coauthored         has coauthors other than the owner
    paid_partnership   sponsor flag in metadata or #ad/#sponsored/#partner
    hidden_likes       likesCount < 0 (creator hid public likes; unlabelable)
    missing_play_count / under_1000_plays

The boosted-post like-rate filter is NOT applied here — it needs a per-creator
threshold. Run boost_filter.py after ingest.

Usage:
    python ingest.py                 # process everything in data/raw_json/
    python ingest.py export1.json    # process specific export file(s)
    python ingest.py --dry-run       # report the roster + funnel, skip downloads

Outputs:
    data/metadata.parquet    one row per clean reel (the modeling dataset)
    data/videos/<code>.mp4   downloaded video files
    data/failed_scrapes.csv  post URLs the scraper errored on, for re-running
    data/dropped.csv         rows that failed a quality filter, with the reason
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
RAW_JSON_DIR = ROOT / 'data' / 'raw_json'
VIDEO_DIR = ROOT / 'data' / 'videos'
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
FAILED_PATH = ROOT / 'data' / 'failed_scrapes.csv'
DROPPED_PATH = ROOT / 'data' / 'dropped.csv'
CREATORS_PATH = ROOT / 'data' / 'creators.txt'

MIN_PLAYS = 1_000
DOWNLOAD_TIMEOUT_S = 120

# an owner NOT on the creators list with at least this many posts gets a
# warning — it is probably a creator that was scraped but left off the list
STRAY_OWNER_MIN_POSTS = 15

# sponsored-content caption tags; the Apify export has no paid-partnership
# metadata field, so the caption is the only reliable signal
AD_TAG_RE = re.compile(r'#(?:ad|sponsored|partner)\b', re.IGNORECASE)


def parse_record(item: dict) -> dict:
    """Flatten one raw Apify record into a metadata row."""
    likes = item.get('likesCount')
    comments = item.get('commentsCount')
    plays = item.get('videoPlayCount')
    owner = item.get('ownerUsername')

    like_rate = None
    engagement_rate = None
    if likes is not None and plays:
        like_rate = likes / plays
        if comments is not None:
            engagement_rate = (likes + comments) / plays

    # coauthors OTHER than the owner: some exports include the owner
    # themselves in coauthorProducers, which is not a real collab
    coauthors = sorted({
        c.get('username') for c in (item.get('coauthorProducers') or [])
        if c.get('username') and c.get('username') != owner
    })

    # no export seen so far carries these keys, but check them anyway in case
    # Apify adds them back
    sponsored_flag = bool(item.get('isSponsored') or item.get('paidPartnership')
                          or item.get('isPaidPartnership') or item.get('sponsorTags'))

    # kept for auditing promo posts, not used as a filter: brands tag verified
    # retailers/influencers on most organic posts too
    verified_tags = sorted({
        t.get('username') or t.get('full_name') or ''
        for t in (item.get('taggedUsers') or []) if t.get('is_verified')
    })

    return {
        'shortcode': item.get('shortCode'),
        'creator': owner,
        'creator_id': item.get('ownerId'),
        'url': item.get('url'),
        'posted_at': item.get('timestamp'),
        'caption': item.get('caption') or '',
        'hashtags': ' '.join(item.get('hashtags') or []),
        'likes': likes,
        'comments': comments,
        'views': item.get('videoViewCount'),
        'plays': plays,
        'like_rate': like_rate,
        'engagement_rate': engagement_rate,
        'duration_s': item.get('videoDuration'),
        'uses_original_audio': (item.get('musicInfo') or {}).get('uses_original_audio'),
        'audio_id': (item.get('musicInfo') or {}).get('audio_id'),
        'song_name': (item.get('musicInfo') or {}).get('song_name'),
        'apify_transcript': item.get('transcript') or '',
        'product_type': item.get('productType'),
        'coauthors': ' '.join(coauthors),
        'is_sponsored': sponsored_flag,
        'mentions': ' '.join(item.get('mentions') or []),
        'verified_tags': ' '.join(verified_tags),
        'video_url': item.get('videoUrl'),
        'apify_video_url': item.get('downloadedVideo'),
    }


def filter_reason(row: dict, primary_creators: set[str]) -> str | None:
    """Return why a row should be excluded, or None if it passes."""
    if row['creator'] not in primary_creators:
        return 'collab_leakin'
    if row['product_type'] != 'clips':
        return 'not_a_reel'
    if row['coauthors']:
        return 'coauthored'
    if row['is_sponsored'] or AD_TAG_RE.search(row['caption']):
        return 'paid_partnership'
    # instagram returns likesCount = -1 when a creator hides public likes; those
    # posts have no usable engagement signal and can't be labeled
    if row['likes'] is None or row['likes'] < 0:
        return 'hidden_likes'
    if not row['plays']:
        return 'missing_play_count'
    if row['plays'] < MIN_PLAYS:
        return f'under_{MIN_PLAYS}_plays'
    return None


def download_video(row: dict) -> Path | None:
    """Download the mp4 for one row, preferring Apify's copy over the IG CDN.

    Returns the local path, or None if every source failed. Skips files that
    already exist so re-runs are cheap.
    """
    dest = VIDEO_DIR / f'{row["shortcode"]}.mp4'
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    sources = [u for u in (row['apify_video_url'], row['video_url']) if u]
    for url in sources:
        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix('.part')
                with open(tmp, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
            return dest
        except requests.RequestException as e:
            print(f'  download failed for {row["shortcode"]} ({type(e).__name__}), '
                  f'trying next source' if url != sources[-1] else
                  f'  all sources failed for {row["shortcode"]}: {e}')
    return None


def main() -> None:
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if not CREATORS_PATH.exists():
        sys.exit(f'missing {CREATORS_PATH} - list one creator username per line')
    primary_creators = {ln.strip() for ln in CREATORS_PATH.read_text().splitlines()
                        if ln.strip()}

    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    file_args = [p for p in args if not p.startswith('--')]
    json_paths = [Path(p) for p in file_args] or sorted(RAW_JSON_DIR.glob('*.json'))
    if not json_paths:
        sys.exit(f'no JSON exports found in {RAW_JSON_DIR}')

    failures, rows = [], []
    for path in json_paths:
        with open(path) as f:
            records = json.load(f)

        file_failures = [r for r in records if r.get('error')]
        failures.extend(file_failures)
        good = [r for r in records if not r.get('error')]
        print(f'{path.name}: {len(records)} records, {len(good)} good')
        rows.extend(parse_record(r) for r in good)

    # a video's creator is its owner; every owner not on the creators list is a
    # collab leak-in owned by some other account.
    owner_counts = Counter(row['creator'] for row in rows)
    missing = sorted(c for c in primary_creators if owner_counts.get(c, 0) == 0)
    stray = sorted(o for o in owner_counts
                   if o not in primary_creators and owner_counts[o] >= STRAY_OWNER_MIN_POSTS)
    print(f'\n{len(primary_creators)} creators listed in {CREATORS_PATH.name}, '
          f'{len(primary_creators) - len(missing)} present in the scrape')
    if missing:
        print(f'  {len(missing)} listed but MISSING from scrape (re-scrape these): {missing}')
    if stray:
        print(f'  {len(stray)} heavy owners NOT on your list (dropped as leak-ins, '
              f'add them if they belong): {stray}')
    for creator, n in owner_counts.most_common():
        if creator in primary_creators:
            print(f'  {n:>3}  {creator}')

    if failures:
        pd.DataFrame(
            [{'url': r.get('url') or r.get('inputUrl'), 'error': r.get('error')}
             for r in failures]
        ).to_csv(FAILED_PATH, index=False)
        print(f'{len(failures)} scraper failures -> {FAILED_PATH.name} (re-run these URLs)')
    else:
        FAILED_PATH.unlink(missing_ok=True)

    # boost drops are decided per creator in boost_filter.py; carry them across
    # re-ingests so re-running this script can't resurrect them
    prior_boost = pd.DataFrame()
    if DROPPED_PATH.exists():
        prior = pd.read_csv(DROPPED_PATH)
        prior_boost = prior[prior['drop_reason'].str.startswith('boosted')]
    boosted_codes = set(prior_boost['shortcode']) if len(prior_boost) else set()

    kept, dropped = [], []
    for row in rows:
        if row['shortcode'] in boosted_codes:
            continue
        reason = filter_reason(row, primary_creators)
        if reason:
            dropped.append({**row, 'drop_reason': reason})
        else:
            kept.append(row)

    if dropped or len(prior_boost):
        out = pd.concat([pd.DataFrame(dropped), prior_boost], ignore_index=True)
        out.to_csv(DROPPED_PATH, index=False)
        by_reason = Counter(d['drop_reason'] for d in dropped)
        print(f'{len(dropped)} rows filtered out -> {DROPPED_PATH.name}  {dict(by_reason)}')
    else:
        DROPPED_PATH.unlink(missing_ok=True)

    if dry_run:
        print(f'\n[dry run] {len(kept)} clean rows would be kept across '
              f'{len({r["creator"] for r in kept})} creators; no videos downloaded')
        return

    print(f'{len(kept)} clean rows, downloading videos...')
    for row in kept:
        path = download_video(row)
        row['video_path'] = str(path.relative_to(ROOT)) if path else None

    df = pd.DataFrame(kept)
    df['posted_at'] = pd.to_datetime(df['posted_at'])

    # merge with previous runs. drop any shortcode reprocessed this run from the
    # old data first: kept rows are re-added below with fresh counts, and rows
    # now filtered out must not linger from an earlier, more lenient ingest.
    if METADATA_PATH.exists():
        previous = pd.read_parquet(METADATA_PATH)
        reprocessed = {row['shortcode'] for row in rows}
        previous = previous[~previous['shortcode'].isin(reprocessed)]
        df = pd.concat([previous, df], ignore_index=True)
        df = df.drop_duplicates(subset='shortcode', keep='last')

    df = df.sort_values(['creator', 'posted_at']).reset_index(drop=True)
    df.to_parquet(METADATA_PATH, index=False)

    downloaded = df['video_path'].notna().sum()
    print(f'\ndataset now: {len(df)} videos ({downloaded} with mp4) '
          f'across {df["creator"].nunique()} creator(s)')
    print(df.groupby('creator').size().to_string())
    print('\nnext: run boost_filter.py to screen for paid-boosted posts, then')
    print('make_labels.py — ingest refreshes engagement counts, which resets labels')


if __name__ == '__main__':
    main()
