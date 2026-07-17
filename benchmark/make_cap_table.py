"""Capability (IFEval/GSM8K/MMLU) baseline vs GDS-RL tablosu.

results/capability/<tag>_base/  ve  <tag>_grpo/  altindaki lm-eval ciktilarini
okur, ana metrikleri cikarir, cap_table.md yazar.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "capability"

MODELS = [
    ("qwen1.5b", "Qwen2.5-1.5B (full-FT)"),
    ("qwen7b", "Qwen2.5-7B (LoRA)"),
    ("qwen3_4b", "Qwen3-4B (LoRA)"),
    ("phi4_mini", "Phi-4-mini (LoRA)"),
    ("llama8b", "Llama-3.1-8B (LoRA)"),
]


def read_metrics(tag_dir: str):
    files = glob.glob(str(RESULTS / tag_dir / "**" / "results_*.json"), recursive=True)
    if not files:
        return None
    r = json.load(open(sorted(files)[-1], encoding="utf-8"))["results"]

    def g(task, metric):
        return r.get(task, {}).get(metric)

    return {
        "ifeval": g("ifeval", "prompt_level_strict_acc,none"),
        "ifeval_inst": g("ifeval", "inst_level_loose_acc,none"),
        "gsm8k": g("gsm8k", "exact_match,flexible-extract"),
        "mmlu": g("mmlu", "acc,none"),
    }


def fmt(b, g):
    if b is None or g is None:
        return "—"
    d = g - b
    sign = "+" if d >= 0 else ""
    return "{:.3f} -> {:.3f} ({}{:.3f})".format(b, g, sign, d)


def main():
    lines = ["# Yetenek (capability) — baseline vs GDS-RL", "",
             "IFEval = talimat takibi (prompt-strict) | GSM8K = matematik (flexible) | MMLU = bilgi", "",
             "| Model | IFEval | GSM8K | MMLU |", "|---|---|---|---|"]
    for tag, label in MODELS:
        b = read_metrics(f"{tag}_base")
        g = read_metrics(f"{tag}_grpo")
        if not b or not g:
            lines.append(f"| {label} | (bekliyor) | | |")
            continue
        lines.append("| {} | {} | {} | {} |".format(
            label, fmt(b["ifeval"], g["ifeval"]),
            fmt(b["gsm8k"], g["gsm8k"]), fmt(b["mmlu"], g["mmlu"])))
    lines += ["", "_Pozitif = GDS-RL sonrasi yukseldi (yetenek korundu/artti); negatif = gerileme._"]
    out = RESULTS.parent / "cap_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[yazildi] {out}")


if __name__ == "__main__":
    main()
