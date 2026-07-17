"""GDS Bilesen 2 - NLI entailment skoru.

NLI (Natural Language Inference) iki metin arasindaki mantiksal iliskiyi olcer:
entailment (destekleme), contradiction (celisme), neutral (tarafsizlik).

    Giris   : (orijinal_gorev, modelin_adim_ciktisi)
    GDS_nli = P(contradiction) + 0.5 * P(neutral)

Neden 0.5: tarafsiz aksiyonlar kismi drift sayilir.
"""

from __future__ import annotations

import numpy as np

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

_model = None
_label_index: dict[str, int] | None = None


def _get_model():
    global _model, _label_index
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(NLI_MODEL_NAME)
        # Label sirasini model config'inden oku, hard-code etme
        id2label = _model.model.config.id2label
        _label_index = {label.lower(): idx for idx, label in id2label.items()}
        expected = {"contradiction", "entailment", "neutral"}
        if set(_label_index) != expected:
            raise RuntimeError(f"Beklenmeyen NLI label seti: {_label_index}")
    return _model


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def nli_probs(goal: str, output: str) -> dict[str, float]:
    """(goal, output) cifti icin {entailment, neutral, contradiction} olasiliklari."""
    model = _get_model()
    logits = np.asarray(model.predict([(goal, output)])).reshape(-1)
    probs = _softmax(logits)
    return {label: float(probs[idx]) for label, idx in _label_index.items()}


def nli_drift(goal: str, output: str) -> float:
    """NLI tabanli drift skoru (0-1): P(contradiction) + 0.5 * P(neutral)."""
    if not goal.strip() or not output.strip():
        raise ValueError("goal ve output bos olamaz")

    probs = nli_probs(goal, output)
    score = probs["contradiction"] + 0.5 * probs["neutral"]
    return float(np.clip(score, 0.0, 1.0))
