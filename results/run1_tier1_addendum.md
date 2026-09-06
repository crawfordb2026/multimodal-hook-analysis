# Run 1 — Tier 1 upgrades (28 creators, 2026-08-11)

Additions on the same 1,237-video / 28-creator set. Still 5-fold GroupKFold by
creator. Compare to `run1_28creators.md`.

## Continuous outperformance target + gradient boosting (regress.py)

Target switched from 3-way terciles to continuous within-creator outperformance
= log(engagement / creator median). Silent videos now have their CLAP vector
masked to zero (they have no real audio).

| Model | rank-acc | spearman | tercile-acc |
|---|---|---|---|
| Ridge (linear) | 0.536 | +0.156 | 0.371 |
| **XGBoost** | **0.564** | +0.148 | 0.392 |
| HistGradientBoosting | 0.548 | +0.117 | **0.409** |

XGBoost on the continuous target beats run1's best (MLP rank-acc 0.559), and
HistGB gives the best tercile accuracy (0.409 vs run1 0.395–0.403). Modest but
real gains from the continuous target + boosting + silent-audio masking.

## Interpretable creative attributes (attributes.py)

CLIP zero-shot concepts + OpenCV motion, ranked by mean within-creator
correlation with engagement:

| Attribute | corr |
|---|---|
| colorful | +0.157 |
| product_closeup | +0.082 |
| group | +0.056 |
| motion_energy | +0.035 |
| face / cut_count / text_overlay | ~0.00 |
| talking_head | −0.066 |

Within a creator, more colorful/vibrant hooks and product close-ups go with
better posts; talking-to-camera hooks go with worse ones.

## Caption topics (topics.py, BERTopic)

22 topics (34% unclustered). By mean within-creator outperformance:

- **Giveaways / contests** ("tag, win, link in bio"): **+0.558** — by far the
  strongest theme.
- Recipe / mocktail / hydration themes: modestly positive (+0.05 to +0.08).
- Generic promo / "new drop / shop now" / "sparkling summer vibes": negative
  (−0.09 to −0.19).

## Do interpretable features improve prediction? (XGBoost, rank-acc)

| Feature set | rank-acc |
|---|---|
| embeddings only | 0.564 |
| + attributes | 0.548 |
| + attributes + topics | 0.553 |
| interpretable features only (~30) | 0.551 |

They don't *add* signal (embeddings already encode the content), but ~30
human-readable features nearly match the 1,408-dim embeddings — you can go
almost fully interpretable at little accuracy cost.

## Still pending (need new scraping)

- **Audio popularity** (real test of the song hypothesis): scrape each
  `audio_id`'s platform-wide reel count. Infra ready (`audio_id` stored).
- **Expanded creator pool** → `run2`, the main lever for absolute performance.
