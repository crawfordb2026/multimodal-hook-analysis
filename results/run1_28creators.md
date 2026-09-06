# Run 1 — 28 creators (2026-08-07)

Baseline snapshot to compare against after expanding the creator pool.
Machine-readable copy in `run1_28creators.json`.

**Dataset:** 1,237 videos, 28 creators, terciles 422 low / 402 mid / 413 high.
5-fold GroupKFold by creator. Chance acc ≈ 0.333; rank-acc chance = 0.500.

## Headline models

| Model | Accuracy | Macro-F1 | Rank-acc |
|---|---|---|---|
| Most-frequent floor | 0.341 | 0.170 | — |
| Logistic (all modalities) | 0.381 | 0.376 | — |
| 10-token transformer (5 seeds) | 0.389 ± 0.004 | 0.383 | 0.551 |
| Late-fusion MLP, all (5 seeds) | 0.395 ± 0.006 | 0.389 | 0.559 |
| Late-fusion MLP, CLIP-only (5 seeds) | 0.403 ± 0.013 | 0.397 | 0.554 |

## MLP modality ablation (5 seeds)

| Modalities | Accuracy | Rank-acc |
|---|---|---|
| clip | 0.403 ± 0.013 | 0.554 |
| text | 0.360 ± 0.004 | 0.526 |
| clap | 0.326 ± 0.008 | 0.483 |
| clip+text | 0.385 ± 0.009 | 0.555 |
| clip+clap | 0.380 ± 0.008 | 0.547 |
| text+clap | 0.359 ± 0.008 | 0.519 |
| clip+text+clap | 0.395 ± 0.006 | 0.559 |

## Silent-video audio ablation (accuracy)

| Subset | CLIP+text | +audio | Δ |
|---|---|---|---|
| Has audio (n=1112) | 0.380 | 0.393 | +0.013 |
| Silent (n=125) | 0.426 | 0.410 | −0.016 |

## Audio-metadata test (song type: original vs licensed)

Testing the "song choice matters" hypothesis with the cheap proxy we already
have (original / licensed / unknown one-hot). 5 seeds.

| Model | Accuracy | Rank-acc |
|---|---|---|
| audiometa alone | 0.333 ± 0.001 | — (chance) |
| MLP clip+text+clap | 0.395 ± 0.006 | 0.559 |
| MLP + audiometa | 0.394 ± 0.005 | 0.555 |
| transformer 10-token | 0.383 ± 0.005 | 0.544 |
| transformer 11-token (+audiometa) | 0.384 ± 0.011 | 0.544 |

**Result: no lift.** Original-vs-licensed alone is at chance, and adding it to
the full model changes nothing. The weak raw association (licensed posts did
slightly better on average) does not survive honest GroupKFold evaluation.
This tests only a crude proxy — NOT song *popularity*. The real test (each
audio_id's platform-wide reel count) needs a separate audio-page scrape; the
`audio_id`/`song_name` columns are now stored so that test is ready to build.

## Notes for comparison

- Watch whether **accuracy over the floor** and **rank-acc over 0.500** grow with
  more creators — a bigger/more varied pool is the main lever, and should also
  let the transformer close on or pass the MLP.
- Re-run `baselines.py`, `train.py`, `ablate.py` after expanding, then write
  `run2_<N>creators.md` alongside this.
