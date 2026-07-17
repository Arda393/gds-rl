"""GoalDrift-Bench butunluk testleri.

Doc Asama 2 sartlari:
  - 200 gorev JSON formatinda hazir
  - 4 kategori esit dagilimda (50'ser)
  - Her gorevde relevant_actions listesi var
  - Gorevin siniri (scope_boundary) net olmali
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.build_tasks import ACTION_VOCABULARY, CATEGORIES, ID_PATTERN, REQUIRED_FIELDS

TASKS_DIR = PROJECT_ROOT / "benchmark" / "tasks"


def load_tasks() -> list[dict]:
    tasks = []
    for path in sorted(TASKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            task = json.load(f)
        task["_filename"] = path.stem
        tasks.append(task)
    return tasks


@pytest.fixture(scope="module")
def tasks():
    found = load_tasks()
    assert found, "tasks/ bos — once benchmark/build_tasks.py calistirilmali"
    return found


class TestBenchmarkIntegrity:
    def test_toplam_200_gorev(self, tasks):
        assert len(tasks) == 200, f"200 gorev bekleniyordu, {len(tasks)} bulundu"

    def test_kategori_basina_50(self, tasks):
        for category, expected in CATEGORIES.items():
            count = sum(1 for t in tasks if t["category"] == category)
            assert count == expected, f"{category}: {expected} bekleniyordu, {count} bulundu"

    def test_idler_benzersiz_ve_dosya_adiyla_ayni(self, tasks):
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), "tekrarlanan id var"
        for t in tasks:
            assert t["id"] == t["_filename"], f"{t['_filename']}.json icindeki id farkli: {t['id']}"

    def test_id_formati(self, tasks):
        for t in tasks:
            assert ID_PATTERN.match(t["id"]), f"{t['id']}: id formati gecersiz"

    def test_zorunlu_alanlar_dolu(self, tasks):
        for t in tasks:
            for field in REQUIRED_FIELDS:
                assert t.get(field), f"{t['id']}: '{field}' eksik veya bos"

    def test_aksiyonlar_sozlukten(self, tasks):
        for t in tasks:
            assert t["relevant_actions"], f"{t['id']}: relevant_actions bos"
            unknown = [a for a in t["relevant_actions"] if a not in ACTION_VOCABULARY]
            assert not unknown, f"{t['id']}: sozluk disi aksiyonlar: {unknown}"

    def test_promptlar_benzersiz(self, tasks):
        prompts = [t["prompt"] for t in tasks]
        assert len(prompts) == len(set(prompts)), "ayni prompt birden fazla gorevde var"

    def test_scope_boundary_anlamli(self, tasks):
        for t in tasks:
            assert len(t["scope_boundary"]) >= 15, (
                f"{t['id']}: scope_boundary cok kisa: '{t['scope_boundary']}'"
            )

    def test_promptlar_ingilizce(self, tasks):
        # Benchmark dili ingilizce secildi (NLI modeli ingilizce-only oldugu icin).
        # Kaba kontrol: Turkce'ye ozgu karakterler prompt'ta olmamali.
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        for t in tasks:
            found = turkish_chars & set(t["prompt"])
            assert not found, f"{t['id']}: prompt'ta Turkce karakter var: {found}"


class TestTaxonomySync:
    def test_taxonomy_kategorileri_gorevlerle_uyumlu(self, tasks):
        from gds.action import load_taxonomy

        taxonomy = load_taxonomy()
        for category in CATEGORIES:
            union = set()
            for t in tasks:
                if t["category"] == category:
                    union.update(t["relevant_actions"])
            assert set(taxonomy[category]) == union, (
                f"{category}: taxonomy ile gorevler uyumsuz — build_tasks.py yeniden calistirilmali"
            )
