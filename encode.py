"""Phase 3: run frozen encoders over the preprocessed clips, cache as numpy.

Each modality is embedded once by a pretrained, frozen encoder and cached to
disk so training (phase 4) never re-runs the heavy models:

    frames  -> CLIP ViT-B/32   -> [NUM_FRAMES, 512]  (one token per frame)
    caption -> MiniLM-L6-v2     -> [384]
    audio   -> CLAP htsat       -> [512]

Per-video caches are written to data/features/<shortcode>.npz and skipped if
present, so the job is resumable. Runs on Apple MPS when available, else CPU.

Usage:
    python encode.py                 # encode everything missing
    python encode.py --limit 20      # only the first N (for benchmarking)
    python encode.py --device cpu    # force CPU

Outputs:
    data/features/<shortcode>.npz    keys: clip [8,512], text [384], clap [512]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
FRAME_DIR = ROOT / 'data' / 'frames'
AUDIO_DIR = ROOT / 'data' / 'audio'
FEATURE_DIR = ROOT / 'data' / 'features'

CLIP_MODEL = 'openai/clip-vit-base-patch32'
TEXT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CLAP_MODEL = 'laion/clap-htsat-unfused'
AUDIO_SR = 48_000
NUM_FRAMES = 8
BATCH_VIDEOS = 32  # videos per encode batch (frames/audio)


def pick_device(requested: str) -> str:
    import torch
    if requested != 'auto':
        return requested
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_frames(code: str):
    from PIL import Image
    paths = sorted((FRAME_DIR / code).glob('*.jpg'))
    return [Image.open(p).convert('RGB') for p in paths]


def load_audio(code: str) -> np.ndarray:
    import soundfile as sf
    wav, _ = sf.read(AUDIO_DIR / f'{code}.wav', dtype='float32')
    return wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='encode only the first N videos')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    args = parser.parse_args()

    if not METADATA_PATH.exists():
        sys.exit('no metadata.parquet - run the pipeline first')
    df = pd.read_parquet(METADATA_PATH)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    todo = df[~df['shortcode'].map(lambda c: (FEATURE_DIR / f'{c}.npz').exists())]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        print('all videos already encoded; nothing to do')
        return

    import torch
    from transformers import (ClapModel, ClapProcessor, CLIPModel, CLIPProcessor)
    from sentence_transformers import SentenceTransformer

    device = pick_device(args.device)
    print(f'device: {device} | encoding {len(todo)} of {len(df)} videos')

    t0 = time.time()
    print('loading models...')
    clip = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clap = ClapModel.from_pretrained(CLAP_MODEL).to(device).eval()
    clap_proc = ClapProcessor.from_pretrained(CLAP_MODEL)
    text_model = SentenceTransformer(TEXT_MODEL, device=device)
    print(f'models loaded in {time.time() - t0:.0f}s')

    # captions embed fast in one batched call
    codes = todo['shortcode'].tolist()
    captions = todo['caption'].fillna('').tolist()
    text_emb = text_model.encode(captions, batch_size=128, convert_to_numpy=True,
                                 show_progress_bar=False)
    text_by_code = dict(zip(codes, text_emb))

    t_enc = time.time()
    done = 0
    for start in range(0, len(codes), BATCH_VIDEOS):
        batch = codes[start:start + BATCH_VIDEOS]

        # CLIP: encode all frames of the batch as one pass, reshape per video
        frames_flat, counts = [], []
        for code in batch:
            imgs = load_frames(code)
            counts.append(len(imgs))
            frames_flat.extend(imgs)
        with torch.no_grad():
            px = clip_proc(images=frames_flat, return_tensors='pt').to(device)
            img_feats = clip.get_image_features(**px).pooler_output.cpu().numpy()
        clip_by_code, offset = {}, 0
        for code, k in zip(batch, counts):
            clip_by_code[code] = img_feats[offset:offset + k]
            offset += k

        # CLAP: batch the audio
        audios = [load_audio(code) for code in batch]
        with torch.no_grad():
            ain = clap_proc(audio=audios, sampling_rate=AUDIO_SR,
                            return_tensors='pt', padding=True).to(device)
            aud_feats = clap.get_audio_features(**ain).pooler_output.cpu().numpy()

        for i, code in enumerate(batch):
            np.savez(FEATURE_DIR / f'{code}.npz',
                     clip=clip_by_code[code].astype('float32'),
                     text=text_by_code[code].astype('float32'),
                     clap=aud_feats[i].astype('float32'))
        done += len(batch)
        rate = done / (time.time() - t_enc)
        print(f'  {done}/{len(codes)} encoded ({rate:.1f} vid/s)')

    elapsed = time.time() - t_enc
    print(f'\nencoded {done} videos in {elapsed:.0f}s ({done / elapsed:.1f} vid/s)')
    print(f'features -> {FEATURE_DIR.relative_to(ROOT)}/<shortcode>.npz')


if __name__ == '__main__':
    main()
