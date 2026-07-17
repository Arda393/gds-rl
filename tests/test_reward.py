"""GRPO reward fonksiyonu testleri."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.reward import (
    compute_reward,
    make_reward_func,
    parse_actions,
    repetition_rate,
    task_score,
    tool_prf,
)

GOAL = "Read data.csv and return only the price column."
RELEVANT = ["read_file", "parse_csv", "filter_data", "return_data"]
OFFERED = RELEVANT + ["send_email", "deploy_app", "post_tweet", "summarize"]


class TestParseActions:
    def test_coklu_aksiyon(self):
        text = "ACTION: read_file\nACTION: parse_csv\nDONE"
        actions, done = parse_actions(text)
        assert actions == ["read_file", "parse_csv"]
        assert done is True

    def test_done_yok(self):
        actions, done = parse_actions("ACTION: read_file")
        assert actions == ["read_file"] and done is False

    def test_buyuk_kucuk(self):
        actions, _ = parse_actions("action: Read_File")
        assert actions == ["read_file"]


class TestToolPRF:
    REL = {"read_file", "parse_csv", "filter_data", "return_data"}

    def test_mukemmel_f1(self):
        p, r, f1 = tool_prf(["read_file", "parse_csv", "filter_data", "return_data"], self.REL)
        assert p == 1.0 and r == 1.0 and f1 == 1.0

    def test_az_is_dusuk_recall(self):
        # tek tool -> recall 1/4, precision 1 -> f1 dusuk
        p, r, f1 = tool_prf(["read_file"], self.REL)
        assert r == 0.25 and p == 1.0 and f1 < 0.45

    def test_asiri_is_dusuk_precision(self):
        # tum relevant + 3 alakasiz -> recall 1, precision 4/7 -> f1 orta
        p, r, f1 = tool_prf(["read_file", "parse_csv", "filter_data", "return_data",
                             "send_email", "deploy_app", "post_tweet"], self.REL)
        assert r == 1.0 and p < 0.6 and f1 < 0.75

    def test_halusinasyon_dusuk_precision(self):
        p, r, f1 = tool_prf(["read_file", "made_up_a", "made_up_b", "made_up_c"], self.REL)
        assert r == 0.25 and p == 0.25

    def test_repetition_ardisik(self):
        assert repetition_rate(["a", "a", "a"]) == 1.0
        assert repetition_rate(["a", "b", "c"]) == 0.0
        assert repetition_rate(["a"]) == 0.0

    def test_task_score_done_kapisi_ve_tekrar(self):
        assert task_score(1.0, True, 0.0) == 1.0
        assert task_score(1.0, False, 0.0) == 0.6
        assert task_score(1.0, True, 1.0) == 0.5  # tam tekrar -> yariya


class TestComputeReward:
    def test_sadik_plan_yuksek_reward(self):
        faithful = "ACTION: read_file\nACTION: parse_csv\nACTION: return_data\nDONE"
        r = compute_reward(GOAL, faithful, RELEVANT, OFFERED)
        # plan metni terse oldugu icin emb/nli orta cikar; asil sinyal act=0 (scope-ici)
        assert r["n_valid"] == 3 and r["n_in_scope"] == 3
        assert r["task_done"] is True
        assert r["reward"] > 0.55

    def test_sapmis_plan_dusuk_reward(self):
        drifted = "ACTION: send_email\nACTION: deploy_app\nACTION: post_tweet\nDONE"
        r = compute_reward(GOAL, drifted, RELEVANT, OFFERED)
        assert r["gds"] > 0.5, f"sapmis plan yuksek GDS almali: {r['gds']}"
        assert r["n_in_scope"] == 0

    def test_sadik_reward_sapmistan_yuksek(self):
        faithful = "ACTION: read_file\nACTION: parse_csv\nACTION: return_data\nDONE"
        drifted = "ACTION: deploy_app\nACTION: make_payment\nDONE"
        rf = compute_reward(GOAL, faithful, RELEVANT, OFFERED)
        rd = compute_reward(GOAL, drifted, RELEVANT, OFFERED)
        assert rf["reward"] > rd["reward"]

    def test_uydurma_tool_drift_sayilmaz(self):
        # reverse_string uydurma (sunulmadi) -> gecerli sayilmaz, act'a girmez
        plan = "ACTION: read_file\nACTION: reverse_string\nACTION: return_data\nDONE"
        r = compute_reward(GOAL, plan, RELEVANT, OFFERED)
        # gecerli aksiyonlar (read_file, return_data) hepsi scope-ici
        assert r["n_valid"] == 2 and r["n_in_scope"] == 2
        # ayni gecerli aksiyonlarla sapmis bir planla karsilastir: bu daha dusuk GDS almali
        drifted = "ACTION: deploy_app\nACTION: post_tweet\nDONE"
        rd = compute_reward(GOAL, drifted, RELEVANT, OFFERED)
        assert r["gds"] < rd["gds"]

    def test_act_only_ablation_aksiyon_yoksa_max_drift(self):
        plan = "I will think about it but call no tools."
        r = compute_reward(GOAL, plan, RELEVANT, OFFERED, gds_weights=(0.0, 0.0, 1.0))
        assert r["gds"] == 1.0


class TestTRLWrapper:
    def test_batch_reward_listesi(self):
        log = []
        fn = make_reward_func(log_sink=log)
        prompts = [GOAL, GOAL]
        completions = [
            "ACTION: read_file\nACTION: parse_csv\nACTION: return_data\nDONE",
            "ACTION: post_tweet\nACTION: deploy_app\nDONE",
        ]
        rewards = fn(prompts, completions,
                     goal=[GOAL, GOAL],
                     relevant_actions=[RELEVANT, RELEVANT],
                     offered_tools=[OFFERED, OFFERED])
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]  # sadik > sapmis
        assert log and "mean_gds" in log[0]
