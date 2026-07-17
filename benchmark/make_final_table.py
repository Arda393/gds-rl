"""Held-out (plan format) with/without GDS-RL sonuc tablolarini uretir.

results/plan_test_baseline_<tag>.json  vs  results/plan_test_grpo_<tag>.json
ciftlerini okur; model tablosu + ablation tablosu basar, final_table.md yazar.

Kullanim:
    python benchmark/make_final_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

MODELS = [
    ("qwen1.5b", "Qwen2.5-1.5B"),
    ("qwen7b", "Qwen2.5-7B"),
    ("qwen3_4b", "Qwen3-4B"),
    ("gemma4b", "Gemma-3-4B"),
    ("gemma12b", "Gemma-3-12B"),
    ("phi4_mini", "Phi-4-mini"),
    ("llama8b", "Llama-3.1-8B"),
]

ABLATIONS = [("emb_only", "Sadece embedding"),
             ("nli_only", "Sadece NLI"),
             ("act_only", "Sadece aksiyon")]


def load(path: Path):
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))["summary"]


def cell(b, g, key, lo_is_better=True):
    bv, gv = b.get(key), g.get(key)
    if bv is None or gv is None:
        return "—"
    arrow = "v" if gv < bv else ("^" if gv > bv else "=")
    good = (gv < bv) if lo_is_better else (gv > bv)
    mark = "+" if good else ("-" if gv != bv else " ")
    return "{:.3f}->{:.3f} {}{}".format(bv, gv, arrow, mark)


def main():
    lines = ["# GDS-RL Sonuclari (held-out 40 test gorevi, plan format, greedy)", ""]
    lines += ["## Model tablosu: without -> with GDS-RL", "",
              "| Model | act (drift) | GDS | F1 | completion |",
              "|---|---|---|---|---|"]
    n_ok = 0
    for tag, label in MODELS:
        b = load(RESULTS / f"plan_test_baseline_{tag}.json")
        g = load(RESULTS / f"plan_test_grpo_{tag}.json")
        if not b or not g:
            lines.append(f"| {label} | (bekliyor) | | | |")
            continue
        n_ok += 1
        lines.append("| {} | {} | {} | {} | {} |".format(
            label,
            cell(b, g, "avg_act"), cell(b, g, "avg_gds"),
            cell(b, g, "avg_f1", lo_is_better=False),
            cell(b, g, "completion_rate", lo_is_better=False)))

    # Ablation: 1.5b baseline vs full vs her bilesen
    b15 = load(RESULTS / "plan_test_baseline_qwen1.5b.json")
    full = load(RESULTS / "plan_test_grpo_qwen1.5b.json")
    if b15 and full:
        lines += ["", "## Ablation (Qwen2.5-1.5B): hangi GDS bileseni?",
                  "_Not: reward'daki F1 task-score tum kosullarda sabit; bu tablo F1 ustune "
                  "her GDS bileseninin marjinal katkisini gosterir. 40-gorev/tek-seed, varyans yuksek._", "",
                  "| Kosul | act | F1 | completion |", "|---|---|---|---|",
                  "| baseline (egitimsiz) | {:.3f} | {:.3f} | {:.2f} |".format(
                      b15["avg_act"], b15["avg_f1"], b15["completion_rate"]),
                  "| full (emb+nli+act) | {:.3f} | {:.3f} | {:.2f} |".format(
                      full["avg_act"], full["avg_f1"], full["completion_rate"])]
        for abl, albl in ABLATIONS:
            a = load(RESULTS / f"plan_test_grpo_qwen1.5b_{abl}.json")
            if a:
                lines.append("| {} | {:.3f} | {:.3f} | {:.2f} |".format(
                    albl, a["avg_act"], a["avg_f1"], a["completion_rate"]))

    lines += ["", f"_Tamamlanan model: {n_ok}/{len(MODELS)}. Tum sayilar gercek TRUBA A100 kosumlarindan._"]
    out = RESULTS / "final_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[yazildi] {out}")


if __name__ == "__main__":
    main()
