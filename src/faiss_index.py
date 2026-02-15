"""
FAISS index builder and batched top-k query.

Uses ``IndexFlatIP`` (inner-product on L2-normalised vectors = cosine
similarity) with an ``IndexIDMap`` for concept-ID mapping.
Falls back to CPU if GPU is unavailable.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _try_gpu_index(dim: int):
    """Attempt to build a GPU FAISS index. Returns None on failure."""
    try:
        import faiss

        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            logger.info("FAISS: %d GPU(s) detected, using GPU index.", ngpus)
            cpu_index = faiss.IndexFlatIP(dim)
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            return gpu_index
    except Exception as exc:
        logger.debug("GPU FAISS unavailable: %s", exc)
    return None


def build_index(
    embeddings: np.ndarray,
    ids: np.ndarray,
    use_gpu: bool = False,
) -> "faiss.Index":
    """Build a FAISS inner-product index with ID mapping.

    Args:
        embeddings: ``(N, D)`` float32, L2-normalised.
        ids: ``(N,)`` int64 concept IDs.
        use_gpu: If True, attempt GPU acceleration.

    Returns:
        A FAISS index ready for ``search()``.
    """
    import faiss

    n, dim = embeddings.shape
    logger.info("Building FAISS index: N=%d, D=%d.", n, dim)

    if n > 25_000:
        logger.warning(
            "⚠ N=%d > 25k — consider IndexIVF/IndexHNSW for better "
            "scaling.",
            n,
        )

    base_index = None
    if use_gpu:
        base_index = _try_gpu_index(dim)

    if base_index is None:
        logger.info("Using CPU IndexFlatIP.")
        base_index = faiss.IndexFlatIP(dim)

    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(
        np.ascontiguousarray(embeddings, dtype=np.float32),
        np.ascontiguousarray(ids, dtype=np.int64),
    )
    logger.info("FAISS index built: %d vectors indexed.", index.ntotal)
    return index


def query_topk(
    index: "faiss.Index",
    queries: np.ndarray,
    k: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched top-k nearest-neighbor query.

    Args:
        index: Built FAISS index.
        queries: ``(M, D)`` float32 query vectors.
        k: Number of neighbors per query.

    Returns:
        Tuple of:
        - ``similarities``: ``(M, k)`` float32 cosine similarities
        - ``neighbor_ids``: ``(M, k)`` int64 concept IDs
          (``-1`` for padding if fewer than k results)
    """
    actual_k = min(k, index.ntotal)
    similarities, neighbor_ids = index.search(
        np.ascontiguousarray(queries, dtype=np.float32),
        actual_k,
    )
    return similarities.astype(np.float32), neighbor_ids.astype(np.int64)
