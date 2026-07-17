"""GoalDrift-Bench train/test ayrimi (kategori-dengeli, seed'li).

Egitim train split'inde yapilir; degerlendirme held-out test split'inde.
Bu sayede GDS-RL'in gercek genelleme kazanci olculur (ezber degil).

    160 train / 40 test  (her kategoriden 40 train / 10 test)

Kullanim:
    python benchmark/split.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
TASKS_DIR = BENCH_DIR / "tasks"
SPLIT_PATH = BENCH_DIR / "splits.json"

CATEGORIES = ["coding", "research", "tool_use", "planning"]
TEST_PER_CATEGORY = 10
SEED = 42


def make_split() -> dict[str, list[str]]:
    rng = random.Random(SEED)
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for path in sorted(TASKS_DIR.glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        by_cat[task["category"]].append(task["id"])

    train, test = [], []
    for cat in CATEGORIES:
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        test.extend(sorted(ids[:TEST_PER_CATEGORY]))
        train.extend(sorted(ids[TEST_PER_CATEGORY:]))

    return {"seed": SEED, "train": sorted(train), "test": sorted(test)}


def main():
    split = make_split()
    SPLIT_PATH.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"train: {len(split['train'])} gorev | test: {len(split['test'])} gorev")
    print(f"[yazildi] {SPLIT_PATH}")


if __name__ == "__main__":
    main()
