from __future__ import annotations

import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(
    first_embedding: np.ndarray, second_embedding: np.ndarray
) -> float:
    first = normalize_embedding(first_embedding)
    second = normalize_embedding(second_embedding)
    return float(np.dot(first, second))
