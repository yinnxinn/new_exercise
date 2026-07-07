from __future__ import annotations

import hashlib
import math
from collections import Counter

from ..nlp.text import tokenize


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        return vec
    return [value / norm for value in vec]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class HashVectorizer:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token, count in Counter(tokenize(text)).items():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        return normalize(vec)
