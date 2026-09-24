"""Adaptive-k vs fixed-k RAG retrieval context selection evaluation.

This module implements two retrieval context-selection strategies over a
16-document synthetic corpus and measures their hit rate and context
efficiency:

* **Baseline (fixed top-3):** always returns the 3 highest-scoring documents
  for every query, regardless of score magnitude.
* **Algorithm (adaptive-k):** for each query, selects documents whose
  cosine similarity exceeds ``mean(sims) + 0.5 * std(sims)``.  If no document
  passes the threshold, falls back to the single top-ranked document.

Metrics
-------
* ``hit_at_1`` – fraction of queries where the gold document is ranked #1.
* ``hit_at_k`` – fraction of queries where the gold document appears in the
  selected context set.
* ``avg_context_size`` – mean number of documents selected per query
  (lower is better for context efficiency).

Determinism
-----------
All randomness is seeded via ``make_dataset``.  The retrieval and scoring
steps are purely deterministic functions of the corpus and queries, so two
calls with identical inputs produce identical outputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import make_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FIXED_K = 3  # baseline always returns this many docs
_ADAPTIVE_STD_MULTIPLIER = 0.5  # threshold = mean + 0.5 * std


def _build_tfidf_index(corpus: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fit a TF-IDF vectorizer on the corpus and return (vectorizer, matrix).

    Parameters
    ----------
    corpus : list[str]
        The document corpus.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer (1-gram, sublinear TF).
    tfidf_matrix : np.ndarray, shape (n_docs, n_features)
        TF-IDF representation of the corpus.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        sublinear_tf=True,
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
    return vectorizer, tfidf_matrix


def _retrieve_fixed_k(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    k: int = _FIXED_K,
) -> list[int]:
    """Return the indices of the top-k documents by cosine similarity.

    Parameters
    ----------
    query : str
        The query string.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    tfidf_matrix : np.ndarray
        TF-IDF matrix of the corpus.
    k : int
        Number of documents to return.

    Returns
    -------
    list[int]
        Indices of the top-k documents (descending similarity).
    """
    query_vec = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # argsort descending; take top k
    top_indices = np.argsort(sims)[::-1][:k]
    return [int(i) for i in top_indices]


def _retrieve_adaptive_k(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    std_multiplier: float = _ADAPTIVE_STD_MULTIPLIER,
) -> list[int]:
    """Return document indices whose similarity exceeds the adaptive threshold.

    Threshold = mean(sims) + std_multiplier * std(sims).
    Fallback: if no document passes, return only the top-1 document.

    Parameters
    ----------
    query : str
        The query string.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    tfidf_matrix : np.ndarray
        TF-IDF matrix of the corpus.
    std_multiplier : float
        Multiplier for the standard deviation in the threshold.

    Returns
    -------
    list[int]
        Indices of selected documents (descending similarity).
    """
    query_vec = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))
    threshold = mean_sim + std_multiplier * std_sim

    selected = [int(i) for i in np.where(sims > threshold)[0]]

    # Fallback: if nothing passes, use top-1
    if not selected:
        top_idx = int(np.argmax(sims))
        selected = [top_idx]

    # Sort by descending similarity for consistent ordering
    selected.sort(key=lambda i: -sims[i])
    return selected


def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    selected_lists: list[list[int]],
    ranked_lists: list[list[int]],
) -> dict[str, float]:
    """Compute hit_at_1, hit_at_k, and avg_context_size.

    Parameters
    ----------
    X : np.ndarray
        Query strings (unused here but kept for interface consistency).
    y : np.ndarray
        Gold document indices.
    selected_lists : list[list[int]]
        For each query, the list of selected document indices.
    ranked_lists : list[list[int]]
        For each query, the full ranking (descending similarity).

    Returns
    -------
    dict[str, float]
        Metrics dictionary.
    """
    n = len(y)
    hit_at_1_count = 0
    hit_at_k_count = 0
    total_context_size = 0

    for i in range(n):
        gold = int(y[i])
        # hit_at_1: gold is the first in the full ranking
        if ranked_lists[i] and ranked_lists[i][0] == gold:
            hit_at_1_count += 1
        # hit_at_k: gold is in the selected set
        if gold in selected_lists[i]:
            hit_at_k_count += 1
        total_context_size += len(selected_lists[i])

    return {
        "hit_at_1": hit_at_1_count / n,
        "hit_at_k": hit_at_k_count / n,
        "avg_context_size": total_context_size / n,
    }


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """Run the adaptive-k vs fixed-k retrieval experiment.

    Parameters
    ----------
    seed : int
        Random seed for query generation.
    n_samples : int
        Number of queries to evaluate.

    Returns
    -------
    dict
        JSON-serializable dictionary with keys:
        - ``n_samples`` (int)
        - ``metrics`` (dict[str, float]): adaptive-k metrics
        - ``baseline_metrics`` (dict[str, float]): fixed top-3 metrics
        - ``explanation`` (str): brief description of the setup
    """
    # Generate queries and gold labels
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    # Build the TF-IDF index from the corpus (imported from data module)
    from data import _CORPUS

    vectorizer, tfidf_matrix = _build_tfidf_index(_CORPUS)

    # Pre-compute full rankings for hit_at_1 (both methods use the same ranking)
    all_rankings: list[list[int]] = []
    baseline_selected: list[list[int]] = []
    adaptive_selected: list[list[int]] = []

    for i in range(n_samples):
        query = str(X[i])
        query_vec = vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked = [int(j) for j in np.argsort(sims)[::-1]]
        all_rankings.append(ranked)

        # Baseline: fixed top-3
        baseline_selected.append(ranked[:_FIXED_K])

        # Adaptive: threshold-based selection
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        threshold = mean_sim + _ADAPTIVE_STD_MULTIPLIER * std_sim
        selected = [int(j) for j in np.where(sims > threshold)[0]]
        if not selected:
            selected = [int(np.argmax(sims))]
        selected.sort(key=lambda j: -sims[j])
        adaptive_selected.append(selected)

    # Compute metrics
    metrics = _compute_metrics(X, y, adaptive_selected, all_rankings)
    baseline_metrics = _compute_metrics(X, y, baseline_selected, all_rankings)

    return {
        "n_samples": int(n_samples),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "baseline_metrics": {k: float(v) for k, v in baseline_metrics.items()},
        "explanation": (
            "Adaptive-k selects docs where cosine similarity exceeds "
            "mean + 0.5*std of per-query similarities (fallback: top-1). "
            "Baseline always returns fixed top-3. "
            "Corpus: 16 docs, 4 topics x 4 docs, disjoint topic vocabularies. "
            "Queries: 5 words (3 topic-relevant + 2 cross-topic noise). "
            "No train/test split needed (unsupervised retrieval); all queries evaluated."
        ),
    }
