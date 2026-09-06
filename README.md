# Multimodal Hook Analysis

Predicting Instagram Reel engagement from the **first 3 seconds** of a video —
the "hook" — using frozen pretrained encoders over vision, text, and audio, with
a deliberately honest evaluation protocol.

The headline result is not a big accuracy number. It's a clear-eyed answer to a
genuinely hard question, with a clean multimodal ablation and an evaluation that
doesn't fool itself.

## The question

For a given creator, can the first 3 seconds of a Reel tell you whether it will
be one of their **better** or **worse** performing posts? The label is each
video's engagement tercile (`low` / `mid` / `high`) **computed within its own
creator**, so the task is about hook quality, not which creator is bigger.

Crucially, evaluation uses **GroupKFold by creator**: every creator is held out
entirely in one fold, so the model is always scored on creators it has never
seen. This is the difference between "can you recognize this creator's style"
(easy, and a leak) and "can you judge a hook from a creator you don't know"
(hard, and the real question).

## Results

5-fold GroupKFold by creator. Chance accuracy ≈ 0.333; `rank-acc` is the
within-creator pairwise ranking metric (given two of a creator's videos, does
the model order them by true engagement? chance = 0.500). Neural rows are mean ±
std over 5 seeds.

| Model | Accuracy | Macro-F1 | Rank-acc |
|---|---|---|---|
| Most-frequent floor | 0.341 | 0.170 | — |
| Logistic regression (all modalities) | 0.381 | 0.376 | — |
| 10-token transformer | 0.389 ± 0.004 | 0.383 | 0.551 |
| **Late-fusion MLP (all modalities)** | **0.395 ± 0.006** | 0.389 | **0.559** |
| Late-fusion MLP (**CLIP frames only**) | **0.403 ± 0.013** | 0.397 | 0.554 |

### Modality ablation (late-fusion MLP, 5 seeds)

| Modalities | Accuracy | Rank-acc |
|---|---|---|
| CLIP (frames) | **0.403 ± 0.013** | 0.554 |
| MiniLM (caption) | 0.360 ± 0.004 | 0.526 |
| CLAP (audio) | 0.326 ± 0.008 | 0.483 |
| CLIP + text | 0.385 ± 0.009 | 0.555 |
| CLIP + audio | 0.380 ± 0.008 | 0.547 |
| text + audio | 0.359 ± 0.008 | 0.519 |
| CLIP + text + audio | 0.395 ± 0.006 | **0.559 ± 0.005** |

### Continuous target + gradient boosting

Because within-creator ranking carried more signal than tercile classification,
a later pass regresses a continuous **within-creator outperformance** target
(log of engagement over the creator's median) and adds gradient boosting:

| Model | Rank-acc | Tercile-acc |
|---|---|---|
| Ridge | 0.536 | 0.371 |
| **XGBoost** | **0.564** | 0.392 |
| HistGradientBoosting | 0.548 | **0.409** |

XGBoost on the continuous target edges out the neural models, confirming the
3-bucket label was discarding ranking signal.

## What makes a good hook (interpretable analysis)

Opaque embeddings predict best, but they don't *explain*. Two cheap analyses
turn them into human-readable findings, ranked by within-creator association
with engagement (so creator size can't confound them):

- **Caption theme is the strongest lever.** BERTopic on captions:
  **giveaway/contest hooks** ("tag a friend, link in bio") beat a creator's
  median post by far (**+0.56** log-outperformance); generic "new drop / shop
  now / summer vibes" promo hooks *underperform* (−0.09 to −0.19).
- **Visual style, via CLIP zero-shot + OpenCV:** more **colorful/vibrant**
  hooks (+0.16) and **product close-ups** (+0.08) go with a creator's better
  posts; **talking-to-camera** hooks (−0.07) go with worse ones.
- **~30 interpretable features nearly match 1,408-dim embeddings** on ranking
  (0.551 vs 0.564) — you can go almost fully explainable at little cost.

## What we learned

**The task is hard, and the signal is weak but real.** The best model beats the
most-frequent floor by ~5 accuracy points. Predicting engagement for unseen
creators from a 3-second hook is inherently difficult — engagement depends on
audience, timing, and luck as much as on hook content. Reporting a modest,
honest number here is the point.

**Vision dominates.** CLIP frames alone reach the best accuracy (0.403) — better
than the full multimodal model on raw accuracy. Caption and audio carry much
less. The modalities are largely redundant with vision; adding them helps
within-creator *ranking* slightly (best rank-acc, 0.559) but not classification.

**Ranking is the better framing.** "Which of this creator's hooks will do
better" (rank-acc 0.559, vs 0.500 chance) is both more learnable and more useful
than the absolute tercile. It's the metric to lead with for a task this hard.

**Audio helps exactly where it exists — a clean validation of the pipeline.**
10% of Reels have no audio track. Splitting the audio ablation by whether a real
track is present:

| Subset | CLIP+text | +audio | Δ |
|---|---|---|---|
| Has audio (n=1112) | 0.380 | 0.393 | **+0.013** |
| Silent (n=125) | 0.426 | 0.410 | **−0.016** |

Adding audio helps on videos that actually have sound and *hurts* on the silent
ones (whose audio feature is synthesized silence). This both explains why audio
looks weak in aggregate and validates tracking silent videos separately — a
future model should mask audio for them.

**A transformer does not beat a simple MLP here.** With ~1,000 training rows per
fold this is a data-limited regime; the smaller model generalizes better. Worth
stating plainly rather than hiding.

## Pipeline

Each stage caches to disk and is re-runnable; run them in order.

| Stage | Script | Output |
|---|---|---|
| 1. Ingest & filter | `ingest.py` | `data/metadata.parquet`, downloaded mp4s |
| 2. Boost screen | `boost_filter.py --apply` | drops paid-boosted posts |
| 3. Labels | `make_labels.py` | within-creator tercile labels |
| 4. Preprocess | `preprocess.py` | 8 frames + audio wav per video (first 3s) |
| 5. Encode | `encode.py` | `data/features/<code>.npz` (CLIP/MiniLM/CLAP) |
| 6. Baselines | `baselines.py` | logistic GroupKFold table |
| 7. Neural models | `train.py` | MLP + transformer |
| 8. Ablations | `ablate.py` | modality + silent-video ablation |
| 9. Regression | `regress.py` | continuous outperformance + XGBoost |
| 10. Attributes | `attributes.py` | CLIP zero-shot + OpenCV creative features |
| 11. Topics | `topics.py` | BERTopic caption themes vs engagement |

### Data curation

Starting from ~1,682 scraped Reels across a **creators.txt** master list, a
sequence of permanent filters yields the modeling set:

```
1682  scraped OK
-172  collab leak-in     (owner not on the creators list)
-156  coauthored / paid partnership
- 65  hidden likes       (likesCount = -1; unlabelable)
- 43  boosted            (low like-rate AND abnormal plays)
-  2  under 1k plays
1244  ────────────────
-  7  creators/videos below the 10-video minimum + 1 unrecoverable mp4
1237  final  (28 creators, terciles 422 / 402 / 413)
```

Two curation decisions worth highlighting:

- **Boosted ≠ weak.** A paid-boosted post shows a low like-rate *and* abnormally
  high plays; a genuinely weak post has a low like-rate at *normal* plays — and
  those weak posts are exactly the `low` tercile we need. The boost filter
  requires both conditions (like-rate below a per-creator cutoff **and** plays >
  5× the creator's median) so it never deletes the low class.
- **Within-creator labels + creator-held-out folds** together remove the obvious
  leak: the model can't use "who is this" as a shortcut for "how did it do."

## Features

Frozen, pretrained, run once and cached as numpy (26 MB total):

- **CLIP ViT-B/32** on 8 frames → `[8, 512]` (one token per frame)
- **MiniLM-L6-v2** on the caption → `[384]`
- **CLAP htsat** on the first-3s audio → `[512]`

## Reproducing

Requires **ffmpeg** and a **native arm64 Python 3.12** (PyTorch has no Intel-mac
or Python-3.14 wheels; on Apple Silicon use a managed arm64 interpreter):

```bash
brew install ffmpeg
uv venv .venv --python 3.12 --python-preference only-managed
uv pip install pandas pyarrow requests matplotlib torch transformers \
               sentence-transformers pillow soundfile scikit-learn \
               xgboost opencv-python-headless bertopic scipy
# XGBoost needs an arm64 libomp; torch bundles one:
ln -sf "$(pwd)/.venv/lib/python3.12/site-packages/torch/lib/libomp.dylib" \
       "$HOME/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/libomp.dylib"
# then run the pipeline stages in order (see table above)
./.venv/bin/python baselines.py
./.venv/bin/python train.py
./.venv/bin/python ablate.py
```

Encoding runs on Apple MPS at ~13 videos/sec (full set ~1.5 min after a one-time
model download).

## Limitations & next steps

- **Small data** (1,237 videos, 28 creators) caps model capacity — the
  transformer can't outrun the MLP here. More creators would be the highest-value
  addition.
- **Silent videos** should have their audio modality masked, not encoded as
  silence; the ablation shows this would recover a small amount of signal.
- **3 seconds may be too short** — a longer or ranking-native objective is worth
  exploring.
- Labels are relative terciles; a **regression or learning-to-rank** objective
  might expose more of the (real, if modest) signal the ranking metric already
  detects.
