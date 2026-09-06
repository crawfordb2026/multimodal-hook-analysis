"""Phase 2: turn each video's first 3 seconds into encoder-ready inputs.

For every video in data/metadata.parquet this extracts, from the opening
WINDOW_S seconds (the "hook"):
    - NUM_FRAMES evenly-spaced frames, center-cropped to FRAME_SIZE, for CLIP
    - one mono wav at AUDIO_SR, for the audio encoder (CLAP / Whisper)

Outputs are cached on disk and skipped if already present, so re-runs are
cheap and the job can be interrupted and resumed. Videos shorter than WINDOW_S
use their full length. Silent videos (no audio track) get a synthesized silent
wav so every clip carries an audio input, and are recorded so phase 3 can treat
their audio modality as absent. Anything ffmpeg can't process is logged.

Usage:
    python preprocess.py                # process everything in metadata
    python preprocess.py --workers 8    # parallel ffmpeg jobs (default 4)

Outputs:
    data/frames/<shortcode>/01.jpg .. 08.jpg   frames for CLIP
    data/audio/<shortcode>.wav                 first-3s audio (real or silence)
    data/silent_videos.csv                     shortcodes with no audio track
    data/preprocess_failed.csv                 videos that failed, with reason
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
FRAME_DIR = ROOT / 'data' / 'frames'
AUDIO_DIR = ROOT / 'data' / 'audio'
FAILED_PATH = ROOT / 'data' / 'preprocess_failed.csv'
SILENT_PATH = ROOT / 'data' / 'silent_videos.csv'

WINDOW_S = 3.0      # seconds of the opening hook to analyze
NUM_FRAMES = 8      # frames handed to CLIP
FRAME_SIZE = 224    # CLIP input side, center-cropped
AUDIO_SR = 48_000   # mono; CLAP's native rate, Whisper resamples down from it

# cover FRAME_SIZE on the short side, then center-crop to a square
SCALE_CROP = (f'scale={FRAME_SIZE}:{FRAME_SIZE}:force_original_aspect_ratio=increase,'
              f'crop={FRAME_SIZE}:{FRAME_SIZE}')


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def frames_done(code: str) -> bool:
    d = FRAME_DIR / code
    return d.is_dir() and len(list(d.glob('*.jpg'))) == NUM_FRAMES


def extract_one(video: Path, code: str, duration: float) -> tuple[str | None, bool]:
    """Extract frames + audio for one video.

    Returns (error, is_silent): error is a string on failure else None, and
    is_silent is True when the video had no audio track and a silent wav was
    synthesized in its place.
    """
    window = min(WINDOW_S, duration) if duration and duration > 0 else WINDOW_S

    if not frames_done(code):
        out_dir = FRAME_DIR / code
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob('*.jpg'):
            stale.unlink()
        fps = NUM_FRAMES / window
        r = _run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
                  '-i', str(video), '-t', f'{window:.3f}',
                  '-vf', f'fps={fps:.6f},{SCALE_CROP}',
                  '-frames:v', str(NUM_FRAMES), '-q:v', '2',
                  str(out_dir / '%02d.jpg')])
        n = len(list(out_dir.glob('*.jpg')))
        if r.returncode != 0:
            return f'frames: {r.stderr.strip()[:120]}', False
        if n != NUM_FRAMES:
            return f'frames: got {n}, expected {NUM_FRAMES}', False

    audio_path = AUDIO_DIR / f'{code}.wav'
    silent = False
    if not audio_path.exists():
        r = _run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
                  '-i', str(video), '-t', f'{window:.3f}',
                  '-vn', '-ac', '1', '-ar', str(AUDIO_SR),
                  str(audio_path)])
        if r.returncode != 0 or not audio_path.exists():
            # video has no audio track: synthesize silence so every clip still
            # carries an audio input, and flag it as silent for phase 3
            s = _run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
                      '-f', 'lavfi', '-i', f'anullsrc=r={AUDIO_SR}:cl=mono',
                      '-t', f'{window:.3f}', str(audio_path)])
            if s.returncode != 0 or not audio_path.exists():
                return f'audio: {r.stderr.strip()[:120]}', False
            silent = True

    return None, silent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4,
                        help='parallel ffmpeg jobs (default 4)')
    args = parser.parse_args()

    if not METADATA_PATH.exists():
        sys.exit('no metadata.parquet - run the ingest pipeline first')
    df = pd.read_parquet(METADATA_PATH)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for _, r in df.iterrows():
        video = ROOT / r['video_path']
        if not video.exists():
            jobs.append((r['shortcode'], 'video file missing'))
            continue
        jobs.append((r['shortcode'], video, r['duration_s']))

    total = len(df)
    print(f'preprocessing {total} videos ({args.workers} workers)...')
    failed, silent_codes, done = [], [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for job in jobs:
            if len(job) == 2:  # missing file, no ffmpeg to run
                failed.append({'shortcode': job[0], 'error': job[1]})
                continue
            code, video, duration = job
            futures[pool.submit(extract_one, video, code, duration)] = code
        for fut in as_completed(futures):
            code = futures[fut]
            err, silent = fut.result()
            if err:
                failed.append({'shortcode': code, 'error': err})
            elif silent:
                silent_codes.append(code)
            done += 1
            if done % 100 == 0:
                print(f'  {done}/{len(futures)} done, {len(failed)} failed')

    if failed:
        pd.DataFrame(failed).to_csv(FAILED_PATH, index=False)
        print(f'\n{len(failed)} videos failed -> {FAILED_PATH.name}')
    else:
        FAILED_PATH.unlink(missing_ok=True)

    if silent_codes:
        pd.DataFrame({'shortcode': sorted(silent_codes)}).to_csv(SILENT_PATH, index=False)
        print(f'{len(silent_codes)} silent videos (no audio track, silence '
              f'synthesized) -> {SILENT_PATH.name}')

    ok = total - len(failed)
    print(f'\ndone: {ok}/{total} videos preprocessed')
    print(f'  frames -> {FRAME_DIR.relative_to(ROOT)}/<shortcode>/01..{NUM_FRAMES:02d}.jpg')
    print(f'  audio  -> {AUDIO_DIR.relative_to(ROOT)}/<shortcode>.wav')


if __name__ == '__main__':
    main()
