# GDS-RL Sonuclari (held-out 40 test gorevi, plan format, greedy)

## Model tablosu: without -> with GDS-RL

| Model | act (drift) | GDS | F1 | completion |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.169->0.054 v+ | 0.394->0.191 v+ | 0.603->0.846 ^+ | 0.100->1.000 ^+ |
| Qwen2.5-7B | 0.054->0.042 v+ | 0.455->0.472 ^- | 0.792->0.807 ^+ | 0.675->0.875 ^+ |
| Qwen3-4B | 0.026->0.018 v+ | 0.463->0.458 v+ | 0.704->0.726 ^+ | 0.675->0.700 ^+ |
| Gemma-3-1B | 0.296->0.281 v+ | 0.504->0.468 v+ | 0.175->0.379 ^+ | 0.050->0.025 v- |
| Gemma-3-4B | 0.325->0.244 v+ | 0.515->0.276 v+ | 0.714->0.753 ^+ | 0.775->0.925 ^+ |
| Gemma-3-12B | 0.163->0.166 ^- | 0.498->0.257 v+ | 0.782->0.797 ^+ | 0.775->0.900 ^+ |
| Phi-4-mini | 0.343->0.264 v+ | 0.605->0.579 v+ | 0.736->0.774 ^+ | 0.625->0.850 ^+ |
| Llama-3.1-8B | 0.322->0.144 v+ | 0.525->0.470 v+ | 0.714->0.811 ^+ | 0.100->0.650 ^+ |

## Ablation (Qwen2.5-1.5B): hangi GDS bileseni?
_Not: reward'daki F1 task-score tum kosullarda sabit; bu tablo F1 ustune her GDS bileseninin marjinal katkisini gosterir. 40-gorev/tek-seed, varyans yuksek._

| Kosul | act | F1 | completion |
|---|---|---|---|
| baseline (egitimsiz) | 0.169 | 0.603 | 0.10 |
| full (emb+nli+act) | 0.054 | 0.846 | 1.00 |
| Sadece embedding | 0.040 | 0.875 | 1.00 |
| Sadece NLI | 0.000 | 0.625 | 0.00 |
| Sadece aksiyon | 0.129 | 0.834 | 1.00 |

_Tamamlanan model: 8/8. Tum sayilar gercek TRUBA A100 kosumlarindan._
