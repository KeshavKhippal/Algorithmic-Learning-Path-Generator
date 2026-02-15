"""
Concept candidate extraction using NLTK.

Uses POS-tagging + RegexpParser noun-phrase chunking to extract
candidate concepts from raw resource text.  NLTK is used because
spaCy is incompatible with Python 3.14 (Cython / pydantic-v1 crash).

The extraction pipeline:
1. Sentence-tokenise the text.
2. POS-tag each sentence.
3. Extract noun phrases via RegexpParser grammar.
4. Normalise: lowercase, strip punctuation, collapse whitespace.
5. Deduplicate per resource, keeping sentence context.
"""

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import nltk
from nltk import RegexpParser, pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Ensure NLTK data is available (idempotent)
# -------------------------------------------------------------------------

_NLTK_READY = False


def _ensure_nltk_data() -> None:
    """Download required NLTK resources if not already present."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)
    _NLTK_READY = True


# -------------------------------------------------------------------------
# Noun-phrase grammar
# -------------------------------------------------------------------------

# Matches sequences of (optional determiners/adjectives) followed by
# one or more nouns — covers the vast majority of technical concept NPs.
_NP_GRAMMAR = r"NP: {<DT|JJ|JJS|JJR>*<NN|NNS|NNP|NNPS>+}"
_NP_PARSER = RegexpParser(_NP_GRAMMAR)

# -------------------------------------------------------------------------
# Normalisation
# -------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_MULTI_SPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().translate(_PUNCT_TABLE)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def _is_valid_candidate(text: str) -> bool:
    """Reject candidates that are too short, purely numeric, or stop-like."""
    if len(text) < 2:
        return False
    if text.isnumeric():
        return False
    # Single-character or single very common word
    if text in {"the", "a", "an", "it", "i", "we", "they", "he", "she",
                "this", "that", "one", "two", "set", "way", "lot", "use"}:
        return False
    return True


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Result of extraction for one resource."""
    resource_id: int
    candidates: List[Tuple[str, str]]  # (normalised_concept, sentence)


def extract_candidates(
    resource_id: int,
    raw_text: str,
    max_text_chars: int = 100_000,
) -> CandidateResult:
    """Extract deduplicated concept candidates from *raw_text*.

    Args:
        resource_id: Database ID of the source resource.
        raw_text: The raw text content (article body / transcript).
        max_text_chars: Truncate input beyond this length to cap cost.

    Returns:
        A ``CandidateResult`` with deduplicated ``(concept, sentence)`` pairs.
    """
    _ensure_nltk_data()

    text = raw_text[:max_text_chars]
    sentences = sent_tokenize(text)

    seen: Dict[str, str] = {}  # normalised_concept -> first sentence

    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        tree = _NP_PARSER.parse(tagged)

        for subtree in tree:
            if not hasattr(subtree, "label"):
                continue
            if subtree.label() != "NP":
                continue

            raw_np = " ".join(word for word, _tag in subtree.leaves())
            normalised = _normalise(raw_np)

            if not _is_valid_candidate(normalised):
                continue

            # Keep first occurrence sentence
            if normalised not in seen:
                seen[normalised] = sent

    candidates = [(concept, sentence) for concept, sentence in seen.items()]
    return CandidateResult(resource_id=resource_id, candidates=candidates)


def extract_all(
    resources: List[Dict],
    max_text_chars: int = 100_000,
) -> List[CandidateResult]:
    """Extract candidates from a list of resource dicts.

    Each dict must have keys ``id`` and ``raw_text``.

    Returns:
        List of ``CandidateResult``, one per resource.
    """
    results: List[CandidateResult] = []
    for res in resources:
        rid = res["id"]
        text = res.get("raw_text") or ""
        if not text.strip():
            logger.warning("Resource %d has empty raw_text, skipping.", rid)
            continue
        cr = extract_candidates(rid, text, max_text_chars=max_text_chars)
        logger.debug(
            "Resource %d: extracted %d candidates.", rid, len(cr.candidates)
        )
        results.append(cr)
    return results
