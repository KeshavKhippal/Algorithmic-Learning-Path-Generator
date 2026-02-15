"""
Canonicalization via agglomerative clustering + difficulty scoring.

Pipeline:
1. Build a unique candidate set across all resources.
2. Embed all unique candidate strings.
3. Compute pairwise cosine similarity.
4. Run AgglomerativeClustering (average linkage) with
   ``distance_threshold = 1 - cluster_threshold``.
5. Choose canonical label per cluster (shortest by token count,
   tie-break by highest global frequency).
6. Score difficulty and assign buckets.
7. Store provenance metadata.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.embeddings.embedder import embed_texts
from src.extractors.spacy_extractor import CandidateResult

logger = logging.getLogger(__name__)


# =========================================================================
# Data structures
# =========================================================================


@dataclass
class CanonicalCluster:
    """One cluster of semantically-equivalent concept strings."""

    canonical_label: str
    members: List[str]
    member_indices: List[int]
    example_sentence: str
    resource_ids: Set[int]
    difficulty_score: float = 0.0
    difficulty_bucket: str = "beginner"


# =========================================================================
# 1. Build unique candidate set
# =========================================================================


@dataclass
class _CandidateIndex:
    """Inverted index from unique candidate text → occurrences."""

    unique_texts: List[str]
    text_to_idx: Dict[str, int]
    # Per unique text: list of (resource_id, sentence)
    occurrences: Dict[str, List[Tuple[int, str]]]
    # Global frequency (count of total occurrences across all resources)
    global_freq: Counter


def _build_candidate_index(results: List[CandidateResult]) -> _CandidateIndex:
    """Build an inverted index from extraction results."""
    occurrences: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    freq: Counter = Counter()

    for cr in results:
        for concept, sentence in cr.candidates:
            occurrences[concept].append((cr.resource_id, sentence))
            freq[concept] += 1

    unique_texts = sorted(occurrences.keys())  # sorted for determinism
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    logger.info(
        "Candidate index: %d unique texts, %d total occurrences.",
        len(unique_texts), sum(freq.values()),
    )

    return _CandidateIndex(
        unique_texts=unique_texts,
        text_to_idx=text_to_idx,
        occurrences=occurrences,
        global_freq=freq,
    )


# =========================================================================
# 2. Cluster
# =========================================================================


def _cluster_candidates(
    embeddings: np.ndarray,
    cluster_threshold: float = 0.85,
) -> np.ndarray:
    """Run agglomerative clustering on embeddings.

    Args:
        embeddings: ``(N, D)`` normalised vectors.
        cluster_threshold: Minimum intra-cluster cosine similarity.

    Returns:
        Cluster label array of shape ``(N,)``.
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    # Cosine similarity (embeddings are L2-normalised)
    sim = embeddings @ embeddings.T
    # Clip for numerical stability
    sim = np.clip(sim, -1.0, 1.0)
    distance = 1.0 - sim

    if n > 10_000:
        logger.warning(
            "⚠ %d candidates — O(n²) clustering will be slow. "
            "Consider approximate NN (FAISS) for future scalability.",
            n,
        )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - cluster_threshold,
    )
    labels = clustering.fit_predict(distance)
    n_clusters = len(set(labels))

    logger.info(
        "Clustered %d candidates into %d canonical groups "
        "(threshold=%.2f).",
        n, n_clusters, cluster_threshold,
    )
    return labels


# =========================================================================
# 3. Choose canonical label
# =========================================================================


def _choose_canonical(
    members: List[str],
    freq: Counter,
) -> str:
    """Pick the canonical label: shortest by token count, tie-break by freq."""
    scored = [
        (len(m.split()), -freq.get(m, 0), m)
        for m in members
    ]
    scored.sort()
    return scored[0][2]


# =========================================================================
# 4. Difficulty scoring
# =========================================================================


def _score_difficulty(
    clusters: List[CanonicalCluster],
    index: _CandidateIndex,
    all_labels: np.ndarray,
    label_per_idx: Dict[int, int],
) -> None:
    """Compute difficulty score and bucket for each cluster *in-place*.

    Features:
    - ``norm_avg_sent_len``: average sentence length (words),
      min-max normalised.
    - ``norm_inv_freq``: ``1 - freq/max_freq`` (rarer → higher).
    - ``cooc_depth``: count of distinct other canonical concepts
      co-occurring in the same sentences, normalised.

    Formula: ``0.35·sent_len + 0.40·inv_freq + 0.25·cooc_depth``
    Buckets: [0, 0.33) → beginner, [0.33, 0.66) → intermediate,
             [0.66, 1.0] → advanced.
    """
    if not clusters:
        return

    max_freq = max(index.global_freq.values()) if index.global_freq else 1

    # Pre-compute: for each sentence, which cluster labels appear
    sent_to_clusters: Dict[str, Set[int]] = defaultdict(set)
    for cluster_idx, cl in enumerate(clusters):
        for member in cl.members:
            for _rid, sent in index.occurrences.get(member, []):
                sent_to_clusters[sent].add(cluster_idx)

    # Compute per-cluster features
    sent_lens: List[float] = []
    inv_freqs: List[float] = []
    cooc_depths: List[float] = []

    for cluster_idx, cl in enumerate(clusters):
        # Average sentence length
        all_sents: List[str] = []
        total_freq = 0
        for member in cl.members:
            for _rid, sent in index.occurrences.get(member, []):
                all_sents.append(sent)
            total_freq += index.global_freq.get(member, 0)

        avg_len = (
            np.mean([len(s.split()) for s in all_sents])
            if all_sents else 0.0
        )
        sent_lens.append(avg_len)

        # Inverse frequency
        inv_freq = 1.0 - (total_freq / max_freq) if max_freq > 0 else 0.0
        inv_freqs.append(inv_freq)

        # Co-occurrence depth
        cooc_labels: Set[int] = set()
        for member in cl.members:
            for _rid, sent in index.occurrences.get(member, []):
                cooc_labels.update(sent_to_clusters.get(sent, set()))
        cooc_labels.discard(cluster_idx)  # don't count self
        cooc_depths.append(float(len(cooc_labels)))

    # Min-max normalise
    def _minmax(vals: List[float]) -> List[float]:
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-12:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    norm_sl = _minmax(sent_lens)
    # inv_freq is already in [0,1] but let's normalise for consistency
    norm_if = _minmax(inv_freqs)
    norm_cd = _minmax(cooc_depths)

    for i, cl in enumerate(clusters):
        score = 0.35 * norm_sl[i] + 0.40 * norm_if[i] + 0.25 * norm_cd[i]
        score = float(np.clip(score, 0.0, 1.0))
        cl.difficulty_score = round(score, 4)

        if score < 0.33:
            cl.difficulty_bucket = "beginner"
        elif score < 0.66:
            cl.difficulty_bucket = "intermediate"
        else:
            cl.difficulty_bucket = "advanced"


# =========================================================================
# Public API
# =========================================================================


def canonicalize(
    extraction_results: List[CandidateResult],
    cluster_threshold: float = 0.85,
    batch_size: int = 64,
) -> Tuple[List[CanonicalCluster], _CandidateIndex, np.ndarray]:
    """Run the full canonicalization pipeline.

    Args:
        extraction_results: Output from ``extract_all()``.
        cluster_threshold: Cosine similarity threshold for merging.
        batch_size: Embedding batch size.

    Returns:
        Tuple of ``(clusters, candidate_index, embeddings)``.
    """
    # 1. Build index
    index = _build_candidate_index(extraction_results)

    if not index.unique_texts:
        logger.warning("No candidate concepts to cluster.")
        return [], index, np.empty((0, 384), dtype=np.float32)

    # 2. Embed
    embeddings = embed_texts(index.unique_texts, batch_size=batch_size)

    # 3. Cluster
    labels = _cluster_candidates(embeddings, cluster_threshold=cluster_threshold)

    # 4. Build cluster objects
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    cluster_member_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_members[int(label)].append(index.unique_texts[idx])
        cluster_member_indices[int(label)].append(idx)

    clusters: List[CanonicalCluster] = []
    for label in sorted(cluster_members.keys()):
        members = cluster_members[label]
        canonical = _choose_canonical(members, index.global_freq)

        # Collect resource IDs and example sentence
        resource_ids: Set[int] = set()
        example_sentence = ""
        for m in members:
            for rid, sent in index.occurrences.get(m, []):
                resource_ids.add(rid)
                if not example_sentence:
                    example_sentence = sent

        clusters.append(CanonicalCluster(
            canonical_label=canonical,
            members=members,
            member_indices=cluster_member_indices[label],
            example_sentence=example_sentence,
            resource_ids=resource_ids,
        ))

    # 5. Difficulty scoring
    label_per_idx = {idx: int(label) for idx, label in enumerate(labels)}
    _score_difficulty(clusters, index, labels, label_per_idx)

    logger.info(
        "Canonicalization complete: %d clusters from %d candidates.",
        len(clusters), len(index.unique_texts),
    )

    return clusters, index, embeddings
