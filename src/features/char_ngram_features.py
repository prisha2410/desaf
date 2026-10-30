"""
char_ngram_features.py — hashed character-trigram + type-token-ratio features.
"""

import re
import zlib
from typing import Dict

import numpy as np

TRIGRAM_HASH_DIM = 64


def _char_trigrams(text: str):
    t = re.sub(r"\s+", " ", text.lower())
    return [t[i:i + 3] for i in range(len(t) - 2)]


def _tokenize_words(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_char_ngram_features(text: str, dim: int = TRIGRAM_HASH_DIM) -> Dict[str, float]:
    trigrams = _char_trigrams(text)
    counts = np.zeros(dim, dtype=np.float32)
    for tg in trigrams:
        counts[zlib.crc32(tg.encode("utf-8")) % dim] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total

    words = _tokenize_words(text)
    ttr = len(set(words)) / len(words) if words else 0.0

    feats = {f"char3g_{i:02d}": float(counts[i]) for i in range(dim)}
    feats["ttr"] = float(ttr)
    return feats