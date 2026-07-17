"""GoalDrift-Bench build scripti.

source/*.json icindeki kategori dosyalarini dogrular ve tasks/ altina
gorev basina tek JSON dosyasi olarak yazar (doc Asama 2: 200 JSON dosyasi).
Ayrica gds/action_taxonomy.json'daki kategori girislerini gorevlerde
kullanilan aksiyonlarin birlesimiyle gunceller.

Calistirma:
    python benchmark/build_tasks.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BENCH_DIR / "source"
TASKS_DIR = BENCH_DIR / "tasks"
TAXONOMY_PATH = BENCH_DIR.parent / "gds" / "action_taxonomy.json"

# Kategori -> beklenen gorev sayisi (doc: 4 kategori x 50)
CATEGORIES = {"coding": 50, "research": 50, "tool_use": 50, "planning": 50}

REQUIRED_FIELDS = ("id", "category", "prompt", "relevant_actions", "scope_boundary")
ID_PATTERN = re.compile(r"^(coding|research|tool_use|planning)_\d{3}$")

# Kanonik aksiyon sozlugu: relevant_actions yalnizca bunlardan secilebilir.
# Tutarli bir sozluk olmadan aksiyon takibi bileseni anlamsizlasir.
ACTION_VOCABULARY = frozenset({
    # dosya / veri
    "read_file", "write_file", "parse_csv", "parse_json", "parse_xml",
    "parse_html", "filter_data", "sort_data", "aggregate_data",
    "transform_data", "validate_data", "convert_format", "return_data",
    # kodlama
    "write_code", "run_code", "debug_code", "refactor_code",
    "write_test", "run_test", "format_code", "read_docs",
    # arastirma
    "search_web", "fetch_url", "read_document", "extract_text",
    "summarize", "cite_source", "compare_sources", "take_notes", "write_text",
    # tool-use
    "call_api", "send_request", "query_database", "compress_file",
    "extract_archive", "monitor_status", "schedule_job", "format_output",
    # planlama
    "list_steps", "order_tasks", "estimate_time", "estimate_cost",
    "assign_resources", "check_constraint", "identify_risk",
    "define_milestone", "write_plan", "prioritize_items",
})


def load_source() -> list[dict]:
    tasks = []
    for category in CATEGORIES:
        path = SOURCE_DIR / f"{category}.json"
        with open(path, encoding="utf-8") as f:
            tasks.extend(json.load(f))
    return tasks


def validate(tasks: list[dict]) -> list[str]:
    errors = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    counts = {c: 0 for c in CATEGORIES}

    for task in tasks:
        tid = task.get("id", "<id yok>")

        for field in REQUIRED_FIELDS:
            if not task.get(field):
                errors.append(f"{tid}: '{field}' alani eksik veya bos")

        if not ID_PATTERN.match(tid):
            errors.append(f"{tid}: id formati gecersiz (kategori_NNN olmali)")
        if tid in seen_ids:
            errors.append(f"{tid}: id tekrar ediyor")
        seen_ids.add(tid)

        category = task.get("category")
        if category not in CATEGORIES:
            errors.append(f"{tid}: bilinmeyen kategori '{category}'")
        else:
            counts[category] += 1
            if not tid.startswith(category + "_"):
                errors.append(f"{tid}: id, '{category}' kategorisiyle uyusmuyor")

        prompt = task.get("prompt", "")
        if prompt in seen_prompts:
            errors.append(f"{tid}: prompt baska bir gorevle ayni")
        seen_prompts.add(prompt)

        actions = task.get("relevant_actions") or []
        unknown = [a for a in actions if a not in ACTION_VOCABULARY]
        if unknown:
            errors.append(f"{tid}: sozlukte olmayan aksiyonlar: {unknown}")

    for category, expected in CATEGORIES.items():
        if counts[category] != expected:
            errors.append(f"{category}: {expected} gorev bekleniyordu, {counts[category]} bulundu")

    return errors


def update_taxonomy(tasks: list[dict]) -> None:
    """Kategori girislerini gorevlerde kullanilan aksiyonlarla gunceller.

    Taxonomy dosyasindaki ornek girisler (web_scraper vb.) korunur.
    """
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy = json.load(f)

    for category in CATEGORIES:
        union: set[str] = set()
        for task in tasks:
            if task["category"] == category:
                union.update(task["relevant_actions"])
        taxonomy[category] = sorted(union)

    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    tasks = load_source()
    errors = validate(tasks)
    if errors:
        print(f"DOGRULAMA BASARISIZ ({len(errors)} hata):")
        for err in errors:
            print(f"  - {err}")
        return 1

    # eski dosyalari temizle, sonra gorev basina tek dosya yaz
    TASKS_DIR.mkdir(exist_ok=True)
    for old in TASKS_DIR.glob("*.json"):
        old.unlink()
    for task in tasks:
        out = TASKS_DIR / f"{task['id']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
            f.write("\n")

    update_taxonomy(tasks)

    print(f"OK: {len(tasks)} gorev yazildi -> {TASKS_DIR}")
    for category, expected in CATEGORIES.items():
        print(f"  {category:<10} {expected} gorev")
    print(f"Aksiyon sozlugu: {len(ACTION_VOCABULARY)} aksiyon")
    print(f"Taxonomy guncellendi -> {TAXONOMY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
