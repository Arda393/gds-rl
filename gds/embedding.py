"""GDS Bilesen 1 - Embedding tabanli semantik mesafe.

Orijinal gorev (G0) ve modelin o adimda urettigi cikti (output_t)
all-MiniLM-L6-v2 ile vektore cevrilir. Cosine distance drift skorudur:

    GDS_emb = 1 - cosine_similarity(embed(G0), embed(output_t))

Aralik: 0.0 (sifir drift) --- 1.0 (tamamen sapti)
"""

from __future__ import annotations

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model bir kere yuklenir, her cagirida tekrar yuklenmez (doc Adim 1.1 sarti).
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embedding_drift(goal: str, output: str) -> float:
    """Orijinal gorev ile cikti arasindaki semantik drift skoru (0-1)."""
    if not goal.strip() or not output.strip():
        raise ValueError("goal ve output bos olamaz")

    model = _get_model()
    vecs = model.encode([goal, output], normalize_embeddings=True)
    cos_sim = float(np.dot(vecs[0], vecs[1]))

    # cosine_similarity teorik olarak [-1, 1]; drift'i [0, 1]'e sabitle
    return float(np.clip(1.0 - cos_sim, 0.0, 1.0))


def embedding_drift_batch(goal: str, outputs: list[str]) -> list[float]:
    """Ayni gorev icin birden fazla ciktinin drift skorlari (rollout'lar icin)."""
    model = _get_model()
    vecs = model.encode([goal] + list(outputs), normalize_embeddings=True)
    goal_vec = vecs[0]
    return [float(np.clip(1.0 - float(np.dot(goal_vec, v)), 0.0, 1.0)) for v in vecs[1:]]
