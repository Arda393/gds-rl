"""GDS (Goal Drift Score) paketi.

Uc bilesenli, LLM-judge gerektirmeyen algoritmik goal drift metrigi:
  - embedding : semantik mesafe (all-MiniLM-L6-v2)
  - nli       : mantiksal celisme skoru (cross-encoder NLI)
  - action    : deterministik aksiyon takibi
  - score     : composite GDS = 0.3*emb + 0.3*nli + 0.4*act
"""

from .embedding import embedding_drift
from .nli import nli_drift
from .action import action_drift, load_taxonomy
from .score import compute_gds, GDSResult

__all__ = [
    "embedding_drift",
    "nli_drift",
    "action_drift",
    "load_taxonomy",
    "compute_gds",
    "GDSResult",
]
