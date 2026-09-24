# Adaptive-k vs Fixed-k RAG Retrieval Context Selection

This project evaluates two context-selection strategies for Retrieval-Augmented Generation (RAG) pipelines on a small, synthetic document corpus. It isolates the retrieval step: deciding which passages to include in the context window before an (out-of-scope) answer generator is invoked.

## Problem

In a RAG system, the retriever returns a set of documents that are concatenated into a prompt for a downstream model. A common default is **fixed top-k** retrieval (e.g., always return the 3 most similar documents). However, if the query is very specific, 3 documents may be wasteful; if the query is ambiguous, 3 may be insufficient.

This project compares:

* **Baseline (fixed top-3):** Always returns the 3 highest-scoring documents, regardless of score distribution.
* **Algorithm (adaptive-k):** For each query, selects documents whose cosine similarity exceeds `mean(sims) + 0.5 * std(sims)`. If no document passes the threshold, it falls back to the single top-ranked document.

The goal is to measure the tradeoff between **retrieval recall** (does the gold document appear in the context?) and **context efficiency** (how many documents are selected on average?).

## Implementation

### Architecture

1. **Corpus (`data.py`):** A fixed 16-document corpus is constructed at import time. It consists of 4 topics × 4 documents each. Each topic has a disjoint vocabulary of 8 content words. Each document contains 4 content words (cycled from the topic vocabulary) plus 8 shared filler words (simulating stopwords).
2. **Query Generation (`data.py`):** `make_dataset` generates `n_samples` queries. Each query contains 5 words: 3 from a randomly chosen primary topic and 2 from a different "noise" topic. The gold label `y[i]` is the index of the document with the maximum content-word overlap with the query.
3. **Retrieval (`app.py`):**
   * A TF-IDF vectorizer (1-gram, sublinear TF) is fitted on the 16-document corpus.
   * For each query, cosine similarity is computed against all 16 documents.
   * **Baseline:** The top 3 indices by similarity are selected.
   * **Adaptive-k:** Indices where similarity > `mean + 0.5 * std` are selected. If empty, top-1 is used.
4. **Metrics (`app.py`):**
   * `hit_at_1`: Fraction of queries where the gold document is ranked #1.
   * `hit_at_k`: Fraction of queries where the gold document is in the selected context set.
   * `avg_context_size`: Mean number of documents selected per query.

### Algorithm vs Baseline

* **Baseline (Fixed Top-3):**
  * *Pros:* Simple, predictable context size.
  * *Cons:* May include irrelevant documents if the top-3 scores are low, or may exclude relevant documents if the query is broad (though with a small corpus, this is less of an issue).
  * *Context Size:* Always exactly 3.

* **Algorithm (Adaptive-k):**
  * *Pros:* Adapts context size to query specificity. For specific queries, it may select only 1-2 documents, reducing token cost. For ambiguous queries, it may select more.
  * *Cons:* Threshold is heuristic. May miss the gold document if its similarity is below the dynamic threshold but above the baseline's 3rd-ranked document.
  * *Context Size:* Variable, typically 1-4 documents.

**Note:** The adaptive method is not guaranteed to outperform the baseline. It trades some recall for context efficiency. The measured results in `example_results.json` and `validation_report.json` show the actual tradeoff for the given seed and sample size.

## Synthetic Dataset Assumptions

* **Corpus:** 16 documents, 4 topics. Topic vocabularies are disjoint. Filler words are shared.
* **Queries:** 5 words each. 3 words from the primary topic, 2 from a noise topic.
* **Gold Label:** Determined by maximum content-word overlap. This is a proxy for relevance, not a true semantic match.
* **Limitations:**
  * The corpus is very small (16 docs), so TF-IDF statistics are coarse.
  * The gold label is based on word overlap, not semantic similarity.
  * The noise words are from other topics, which may not reflect real-world query noise (e.g., typos, unrelated terms).
  * The adaptive threshold is a simple heuristic and may not be optimal for all query distributions.

## Metrics Direction

* `hit_at_1`: Higher is better.
* `hit_at_k`: Higher is better.
* `avg_context_size`: Lower is better (for context efficiency).

## Reproducibility

The project is deterministic. Given the same `seed` and `n_samples`, `run_experiment` will always produce the same output.

### Setup

Requires Python 3.11-3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Running the Experiment

```bash
python main.py --seed 42 --n-samples 256 --output results.json
```

This will generate `results.json` containing the metrics for both the adaptive and baseline methods.

### Running Tests

```bash
python -m pytest -q
```

The tests check:
1. Dataset generation properties (shape, seed sensitivity, validation).
2. Determinism of `run_experiment`.
3. Baseline context size is always 3.
4. Adaptive context size is less than 3 on average (for the given seed).
5. `n_samples` is respected in the output.

## Limitations

* **Small Corpus:** 16 documents is too small to draw general conclusions about RAG retrieval. Results are specific to this synthetic setup.
* **Heuristic Threshold:** The adaptive threshold (`mean + 0.5 * std`) is a simple heuristic. It may not be optimal for all query distributions.
* **No Downstream Evaluation:** The project only evaluates retrieval, not the quality of the final answer. A better retrieval context does not guarantee a better answer.
* **Synthetic Data:** The dataset is synthetic and may not reflect real-world query and document distributions.

## Files

* `data.py`: Synthetic dataset generation.
* `app.py`: Retrieval and evaluation logic.
* `test_project.py`: Pytest tests.
* `requirements.txt`: Dependencies.
* `main.py`: Entry point (provided by host).
* `example_results.json`: Example output from `run_experiment(42, 256)`.
* `validation_report.json`: Measured results from the validation suite.

## Recorded automated validation

Host contract tests and project tests passed (17 tests, 0 skipped). Demo completed on Python 3.13.15. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
