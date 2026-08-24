"""Embedding of raw phenomenological failure descriptions (Section 16-17).

No activation / mechanistic information is used anywhere in this module -
the semantic and mechanistic paths must stay independently frozen
(Section 3).
"""
from __future__ import annotations

import numpy as np


class DescriptionEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, descriptions: list[str]) -> np.ndarray:
        vecs = self._model.encode(descriptions, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs)

    def embed_one(self, description: str) -> np.ndarray:
        return self.embed([description])[0]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """embeddings assumed L2-normalized (as returned by embed())."""
    return embeddings @ embeddings.T
