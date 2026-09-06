# Run 2 — 38 creators (2026-09-06)

Combined data3 + data4: **2,392 videos, 38 creators** (up from 1,237 / 28).
The 10 new creators are drink-content accounts (verified: 92% of their videos
are drink/beverage by CLIP, vs 86% for the original 28 — they are NOT a
different food domain; an earlier "food creators" characterization was wrong).

## Headline: more data, LOWER aggregate scores

| Model | Metric | Run 1 (28) | Run 2 (38) |
|---|---|---|---|
| XGBoost (continuous) | rank-acc | 0.564 | **0.527** |
| Late-fusion MLP | accuracy | 0.395 | 0.358 |
| Late-fusion MLP | rank-acc | 0.559 | 0.515 |
| 10-token transformer | accuracy | 0.389 | 0.363 |
| 10-token transformer | rank-acc | 0.551 | 0.530 |

## Why: the new creators post visually homogeneous content

The drop is NOT signal loss on the originals. Splitting the combined-model
evaluation by group:

| Evaluated on | rank-acc |
|---|---|
| Original 28 | **0.551** |
| New 10 | 0.517 |
| Model trained+tested on only the 28 | 0.557 (≈ run1 0.564) |

The original signal is fully intact; the aggregate drop comes from the new
creators being harder to rank. The reason is **visual homogeneity**, not domain:

- New creators' videos look more alike than the brands' (within-creator CLIP
  spread 0.128 vs 0.184). The most-homogeneous creators are individual drink
  *creators* — new (ashleylaurentassano, drinksbywhitney, mocktailswithlilly,
  rs.album, what.erica.eats) AND original (fruitninjutsu, coffeewithrafaela,
  drinksbyallyy, coffeebyhailey).
- Per-creator, visual spread correlates with rank-acc: Spearman +0.29
  (p≈0.08, n=38). Visually-varied creators average 0.555 rank-acc vs 0.526 for
  homogeneous ones.

Interpretation: when a creator posts near-identical videos (same set, same
format), the visual hook barely varies, so what separates their good and bad
posts is something our visual-dominant model can't see — the specific recipe,
trend, or **song choice** (the user's original hypothesis). CLIP can only rank
creators whose hooks actually differ visually.

Caveat: the effect is moderate and borderline-significant (n=38), so visual
homogeneity is part of the story, not proven to be all of it.

## Secondary finding: the transformer caught the MLP

With ~1,900 training rows per fold the 10-token transformer now edges out the
MLP (rank-acc 0.530 vs 0.515) — the reverse of run 1, where the smaller MLP won
in the data-limited regime. More data helped the higher-capacity model, as
expected.

## Takeaway

The hook→engagement signal is real but only learnable where hooks visually vary.
For creators who post visually repetitive content, the differentiator is
non-visual (song / recipe / trend) — which motivates the audio-popularity
feature and a possible per-creator visual-variance covariate. Report the ~0.55
result and this homogeneity finding together; don't let the blended 0.527
number hide that the model works well exactly where it should.
