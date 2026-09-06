"""Phase 4d: caption topics via BERTopic, and their link to engagement.

Clusters the captions into themes (giveaways, flavor drops, collabs, ...) and
asks which themes go with a creator's better or worse posts. The text-side
counterpart to attributes.py. Reuses the cached MiniLM caption embeddings so
BERTopic doesn't re-embed and the topics are consistent with the text modality.

Engagement is measured as within-creator outperformance (log of engagement over
the creator's median), so a topic's score reflects "posts on this theme beat
that creator's typical post", not just which big creators use the theme.

Writes data/topics.parquet and prints each topic's keywords, size, and mean
within-creator outperformance.

Usage:
    python topics.py [--min-topic-size 20]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / 'data' / 'metadata.parquet'
FEATURE_DIR = ROOT / 'data' / 'features'
TOPICS_PATH = ROOT / 'data' / 'topics.parquet'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-topic-size', type=int, default=12)
    args = parser.parse_args()

    df = pd.read_parquet(METADATA_PATH)
    captions = df['caption'].fillna('').tolist()
    embeddings = np.stack([np.load(FEATURE_DIR / f'{c}.npz')['text']
                           for c in df['shortcode']])

    # within-creator outperformance (same definition as the modeling target)
    med = df.groupby('creator')['engagement_rate'].transform('median')
    df['outperf'] = np.log((df['engagement_rate'] + 1e-9) / (med + 1e-9))

    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
               metric='cosine', random_state=42)
    # drop stopwords/emeoji-noise from the keyword extraction so topic labels
    # are the actual theme words, not "the / and / to"
    vectorizer = CountVectorizer(stop_words='english', min_df=5,
                                 ngram_range=(1, 2))
    model = BERTopic(umap_model=umap, vectorizer_model=vectorizer,
                     min_topic_size=args.min_topic_size,
                     calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(captions, embeddings=embeddings)
    df['topic'] = topics

    info = model.get_topic_info()
    n_topics = int((info['Topic'] >= 0).sum())
    outlier = int((df['topic'] == -1).sum())
    print(f'{len(df)} captions -> {n_topics} topics '
          f'({outlier} unclustered / {100 * outlier / len(df):.0f}%)\n')

    df[['shortcode', 'creator', 'topic', 'outperf']].to_parquet(TOPICS_PATH, index=False)

    # each topic's keywords, size, and mean within-creator outperformance
    rows = []
    for t in info[info['Topic'] >= 0]['Topic']:
        sub = df[df['topic'] == t]
        words = ', '.join(w for w, _ in model.get_topic(t)[:4])
        rows.append((t, len(sub), sub['outperf'].mean(), words))
    rows.sort(key=lambda r: r[2], reverse=True)

    print(f'{"topic":<6}{"n":>5}  {"outperf":>8}  keywords')
    print('-' * 66)
    for t, n, op, words in rows:
        print(f'{t:<6}{n:>5}  {op:+8.3f}  {words}')
    print('\noutperf > 0: theme tends to beat the creator\'s median post')


if __name__ == '__main__':
    main()
