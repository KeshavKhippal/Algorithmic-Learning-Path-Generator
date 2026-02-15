"""
Sentence-transformer embedding module.

Uses ``all-MiniLM-L6-v2`` (384-dimensional, normalised) to embed
concept strings.  The model is loaded lazily as a module-level
singleton to avoid repeated loading across calls.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None  # lazy singleton


def _get_model():
    """Load the sentence-transformer model (lazy, one-time)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: all-MiniLM-L6-v2 …")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded.")
    return _MODEL


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    model: Optional[object] = None,
) -> np.ndarray:
    """Embed a list of strings into normalised float32 vectors.

    Args:
        texts: Strings to embed.
        batch_size: Encoding batch size (default 64).
        model: Optional pre-loaded model (for testing). If ``None``,
               the module-level singleton is used.

    Returns:
        ``np.ndarray`` of shape ``(len(texts), 384)`` with dtype
        ``float32`` and unit L2 norms.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    m = model or _get_model()

    embeddings = m.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    arr = np.asarray(embeddings, dtype=np.float32)
    logger.info(
        "Embedded %d texts → shape=%s, dtype=%s",
        len(texts), arr.shape, arr.dtype,
    )
    return arr
