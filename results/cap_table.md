# Yetenek (capability) — baseline vs GDS-RL

IFEval = talimat takibi (prompt-strict) | GSM8K = matematik (flexible) | MMLU = bilgi

| Model | IFEval | GSM8K | MMLU |
|---|---|---|---|
| Qwen2.5-1.5B (full-FT) | 0.392 -> 0.268 (-0.124) | 0.292 -> 0.252 (-0.039) | 0.579 -> 0.581 (+0.002) |
| Qwen2.5-7B (LoRA) | 0.656 -> 0.651 (-0.006) | 0.395 -> 0.409 (+0.014) | 0.665 -> 0.664 (-0.001) |
| Qwen3-4B (LoRA) | 0.826 -> 0.819 (-0.007) | 0.794 -> 0.793 (-0.001) | 0.618 -> 0.618 (+0.000) |
| Gemma-3-1B (LoRA) | (bekliyor) | | |
| Phi-4-mini (LoRA) | 0.704 -> 0.713 (+0.009) | 0.807 -> 0.818 (+0.011) | 0.662 -> 0.663 (+0.001) |
| Llama-3.1-8B (LoRA) | 0.738 -> 0.745 (+0.007) | 0.835 -> 0.837 (+0.002) | 0.631 -> 0.625 (-0.007) |

_Pozitif = GDS-RL sonrasi yukseldi (yetenek korundu/artti); negatif = gerileme._
