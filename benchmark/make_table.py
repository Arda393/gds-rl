"""Baseline sonuc tablosu uretici.

results/ altindaki duz (baseline_*) ve agentic (agentic_*) JSON'lari okur,
model bazinda karsilastirma tablosu basar ve results/summary_table.md yazar.

Kullanim:
    python benchmark/make_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODELS = [
    ("qwen1.5b", "Qwen2.5-1.5B"),
    ("qwen7b", "Qwen2.5-7B"),
    ("gemma4b", "Gemma-3-4B"),
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main():
    plain_rows, agentic_rows = [], []

    for tag, label in MODELS:
        plain = load(RESULTS_DIR / f"baseline_{tag}.json")
        if plain:
            s = plain["summary"]
            plain_rows.append((label, s["avg_gds"], s["avg_emb"], s["avg_nli"]))

        ag = load(RESULTS_DIR / f"agentic_{tag}.json")
        if ag:
            s = ag["summary"]
            agentic_rows.append((
                label, s["avg_gds"], s["avg_emb"], s["avg_nli"], s["avg_act"],
                s["invalid_action_rate"], s["completion_rate"],
            ))

    lines = ["# GoalDrift-Bench Baseline Sonuclari", ""]

    lines += [
        "## Duz baseline (2-bilesenli: 0.5*emb + 0.5*nli)",
        "Model tek-tur metin cevabi uretir; aksiyon izi yok.",
        "",
        "| Model | avg GDS | emb | nli |",
        "|---|---|---|---|",
    ]
    for label, gds, emb, nli in plain_rows:
        lines.append(f"| {label} | {fmt(gds)} | {fmt(emb)} | {fmt(nli)} |")

    lines += [
        "",
        "## Agentic baseline (3-bilesenli: 0.3*emb + 0.3*nli + 0.4*act)",
        "Model cok-adimli ReAct dongusunde tool cagirir; tuzak (scope-disi) tool'lar sunulur.",
        "`act` = scope-disi gecerli cagri orani (gercek davranissal drift).",
        "`invalid` = uydurma/gecersiz tool orani (tool-format becerisi, drift'e sayilmaz).",
        "",
        "| Model | avg GDS | emb | nli | **act** | invalid | completion |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, gds, emb, nli, act, inv, comp in agentic_rows:
        lines.append(
            f"| {label} | {fmt(gds)} | {fmt(emb)} | {fmt(nli)} | **{fmt(act)}** | {fmt(inv)} | {fmt(comp)} |"
        )

    # kategori bazinda act kirilimini da ekle
    lines += ["", "## Agentic: kategori bazinda aksiyon drift (act)", "",
              "| Model | coding | research | tool_use | planning |", "|---|---|---|---|---|"]
    for tag, label in MODELS:
        ag = load(RESULTS_DIR / f"agentic_{tag}.json")
        if not ag:
            continue
        pc = ag["summary"]["per_category"]
        row = [label] + [fmt(pc.get(c, {}).get("avg_act")) for c in ["coding", "research", "tool_use", "planning"]]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "_Not: Tum sayilar gercek TRUBA A100 kosumlarindan. Hicbir deger uydurulmadi._"]

    out = RESULTS_DIR / "summary_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[yazildi] {out}")


if __name__ == "__main__":
    main()
