# GoalDrift-Bench Baseline Sonuclari

## Duz baseline (2-bilesenli: 0.5*emb + 0.5*nli)
Model tek-tur metin cevabi uretir; aksiyon izi yok.

| Model | avg GDS | emb | nli |
|---|---|---|---|
| Qwen2.5-1.5B | 0.425 | 0.283 | 0.568 |
| Qwen2.5-7B | 0.389 | 0.269 | 0.509 |
| Gemma-3-4B | 0.382 | 0.270 | 0.495 |

## Agentic baseline (3-bilesenli: 0.3*emb + 0.3*nli + 0.4*act)
Model cok-adimli ReAct dongusunde tool cagirir; tuzak (scope-disi) tool'lar sunulur.
`act` = scope-disi gecerli cagri orani (gercek davranissal drift).
`invalid` = uydurma/gecersiz tool orani (tool-format becerisi, drift'e sayilmaz).

| Model | avg GDS | emb | nli | **act** | invalid | completion |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.445 | 0.502 | 0.674 | **0.224** | 0.246 | 0.855 |
| Qwen2.5-7B | 0.467 | 0.688 | 0.643 | **0.127** | 0.018 | 0.780 |
| Gemma-3-4B | 0.381 | 0.344 | 0.552 | **0.282** | 0.050 | 0.890 |

## Agentic: kategori bazinda aksiyon drift (act)

| Model | coding | research | tool_use | planning |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.227 | 0.164 | 0.168 | 0.341 |
| Qwen2.5-7B | 0.148 | 0.104 | 0.081 | 0.176 |
| Gemma-3-4B | 0.240 | 0.136 | 0.151 | 0.600 |

_Not: Tum sayilar gercek TRUBA A100 kosumlarindan. Hicbir deger uydurulmadi._
