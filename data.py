"""Synthetic dataset for adaptive-k vs fixed-k RAG retrieval evaluation.

Generates a fixed 16-document corpus (4 topics x 4 docs) with disjoint
topic vocabularies plus shared filler words, and produces n_samples
queries of 5 words each (3 topic-relevant + 2 cross-topic noise).

The gold label y[i] is the index of the document most relevant to
query i (the document sharing the most content words with the query).

No disk writes. All randomness is seeded.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Corpus definition: 4 topics, 4 docs each, disjoint vocabularies
# ---------------------------------------------------------------------------
# Each topic has 8 content words; each doc contains 4 of them (cycled).
# Shared filler words appear in every document to simulate stopword-like noise.

_FILLER = ["the", "and", "is", "of", "in", "a", "to", "for"]

_TOPIC_VOCAB = [
    # Topic 0: astronomy
    ["star", "galaxy", "nebula", "cosmos", "orbit", "planet", "eclipse", "solar"],
    # Topic 1: oceanography
    ["tide", "reef", "current", "abyss", "coral", "brine", "fathom", "squall"],
    # Topic 2: metallurgy
    ["alloy", "forge", "ingot", "temper", "copper", "zinc", "smelt", "crucible"],
    # Topic 3: botany
    ["petal", "spore", "venom", "bark", "sapling", "chlorophyll", "thorn", "root"],
]

_DOCS_PER_TOPIC = 4
_WORDS_PER_DOC = 4  # content words per document (from the topic vocab)


def _build_corpus() -> list[str]:
    """Build the 16-document corpus.

    Each document contains 4 topic content words (cycled from the topic
    vocabulary) plus all 8 filler words, shuffled deterministically by
    document index for variety.

    Returns
    -------
    list[str]
        16 document strings.
    """
    corpus: list[str] = []
    for t in range(4):
        vocab = _TOPIC_VOCAB[t]
        for d in range(_DOCS_PER_TOPIC):
            # Pick 4 content words starting at offset d
            content = [vocab[(d + j) % len(vocab)] for j in range(_WORDS_PER_DOC)]
            words = content + list(_FILLER)
            # Deterministic shuffle seeded by global doc index
            rng = np.random.default_rng(1000 + t * 10 + d)
            rng.shuffle(words)
            corpus.append(" ".join(words))
    return corpus


_CORPUS: list[str] = _build_corpus()


def _compute_gold_label(query_words: list[str]) -> int:
    """Return the index of the corpus document sharing the most content words
    with the given query words.

    Parameters
    ----------
    query_words : list[str]
        The words in the query.

    Returns
    -------
    int
        Document index (0-15) with maximum content-word overlap.
    """
    query_set = set(w.lower() for w in query_words)
    best_idx = 0
    best_overlap = -1
    for idx, doc in enumerate(_CORPUS):
        doc_set = set(doc.lower().split())
        overlap = len(query_set & doc_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx


def make_dataset(
    seed: int = 42, n_samples: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic RAG queries and gold document labels.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of queries to generate. Must be >= 32.

    Returns
    -------
    X : np.ndarray, shape (n_samples,), dtype=object
        Query strings, each containing 5 words (3 topic-relevant + 2 noise).
    y : np.ndarray, shape (n_samples,), dtype=int64
        Gold document index for each query.

    Raises
    ------
    ValueError
        If n_samples < 32.
    """
    if n_samples < 32:
        raise ValueError("n_samples must be >= 32")

    rng = np.random.default_rng(seed)
    X = np.empty(n_samples, dtype=object)
    y = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        # Choose a primary topic (0-3)
        topic = int(rng.integers(0, 4))
        vocab = _TOPIC_VOCAB[topic]

        # Pick 3 distinct content words from the primary topic
        content_words = list(rng.choice(vocab, size=3, replace=False))

        # Pick a noise topic (different from primary) and 2 words from it
        noise_topic = int(rng.integers(0, 3))
        if noise_topic >= topic:
            noise_topic += 1
        noise_vocab = _TOPIC_VOCAB[noise_topic]
        noise_words = list(rng.choice(noise_vocab, size=2, replace=False))

        # Combine: 3 content + 2 noise, then shuffle
        query_words = content_words + noise_words
        rng.shuffle(query_words)
        query_str = " ".join(query_words)

        X[i] = query_str
        y[i] = _compute_gold_label(query_words)

    return X, y
