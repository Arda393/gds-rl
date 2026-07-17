"""Agentic harness mantik testleri (model gerektirmez)."""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

# evaluate.py'yi import etmeden parse/tool fonksiyonlarini test etmek icin
# agentic_eval'den yalniz saf fonksiyonlari aliyoruz (torch import'u var ama
# test ortaminda torch kurulu degilse bu testler atlanir).
agentic = None
try:
    import agentic_eval as agentic
except Exception:
    agentic = None

import pytest

needs_agentic = pytest.mark.skipif(agentic is None, reason="agentic_eval import edilemedi (torch yok)")


@needs_agentic
class TestParseStep:
    def test_action_satiri(self):
        assert agentic.parse_step("Let me read it.\nACTION: read_file") == ("read_file", False)

    def test_action_parantezli(self):
        assert agentic.parse_step("ACTION: parse_csv(path='x')") == ("parse_csv", False)

    def test_done(self):
        assert agentic.parse_step("The task is finished.\nDONE") == (None, True)

    def test_action_done_dan_oncelikli(self):
        # ayni cevapta hem ACTION hem DONE varsa aksiyon islenir
        assert agentic.parse_step("ACTION: return_data\nDONE") == ("return_data", False)

    def test_buyuk_kucuk_harf(self):
        assert agentic.parse_step("action: Read_File")[0] == "read_file"

    def test_bos(self):
        assert agentic.parse_step("hmm, not sure") == (None, False)


@needs_agentic
class TestBuildToolSet:
    def test_in_scope_her_zaman_var(self):
        all_tools, dangerous = agentic.load_tool_descriptions()
        task = {"relevant_actions": ["read_file", "parse_csv", "return_data"]}
        rng = random.Random(0)
        tools = agentic.build_tool_set(task, all_tools, dangerous, n_offtopic=4, n_dangerous=3, rng=rng)
        names = {n for n, _ in tools}
        for r in task["relevant_actions"]:
            assert r in names, f"{r} in-scope tool eksik"

    def test_tuzak_eklendi(self):
        all_tools, dangerous = agentic.load_tool_descriptions()
        task = {"relevant_actions": ["read_file"]}
        rng = random.Random(1)
        tools = agentic.build_tool_set(task, all_tools, dangerous, n_offtopic=4, n_dangerous=3, rng=rng)
        names = {n for n, _ in tools}
        # 1 in-scope + 4 offtopic + 3 dangerous = 8 (cakisma yoksa)
        assert len(names) >= 7
        assert any(n in dangerous for n in names), "tehlikeli tuzak yok"

    def test_her_tool_aciklamali(self):
        all_tools, dangerous = agentic.load_tool_descriptions()
        task = {"relevant_actions": ["write_plan", "list_steps"]}
        rng = random.Random(2)
        tools = agentic.build_tool_set(task, all_tools, dangerous, 4, 3, rng)
        for name, desc in tools:
            assert desc and len(desc) > 3, f"{name} aciklamasiz"

    def test_descriptions_yuklendi(self):
        all_tools, dangerous = agentic.load_tool_descriptions()
        assert "read_file" in all_tools
        assert "send_email" in dangerous
        assert "_comment" not in all_tools and "_dangerous" not in all_tools


@needs_agentic
def test_system_prompt_tool_listeler():
    tools = [("read_file", "Read a file."), ("send_email", "Send an email.")]
    prompt = agentic.build_system_prompt(tools)
    assert "read_file" in prompt and "send_email" in prompt
    assert "ACTION:" in prompt and "DONE" in prompt


@needs_agentic
class TestPartitionActions:
    OFFERED = {"read_file", "parse_csv", "return_data", "send_email", "deploy_app"}
    RELEVANT = {"read_file", "parse_csv", "return_data"}

    def test_hepsi_in_scope(self):
        valid, invalid, oos = agentic.partition_actions(
            ["read_file", "parse_csv"], self.OFFERED, self.RELEVANT)
        assert valid == ["read_file", "parse_csv"]
        assert invalid == []
        assert oos == []

    def test_sunulan_scope_disi_drift(self):
        # send_email sundugumuz ama scope-disi bir tool -> gercek drift
        valid, invalid, oos = agentic.partition_actions(
            ["read_file", "send_email"], self.OFFERED, self.RELEVANT)
        assert valid == ["read_file", "send_email"]
        assert oos == ["send_email"]
        assert invalid == []

    def test_uydurma_tool_invalid_sayilir(self):
        # reverse_string sunulmadi -> invalid (halusinasyon), drift'e sayilmaz
        valid, invalid, oos = agentic.partition_actions(
            ["read_file", "reverse_string", "un"], self.OFFERED, self.RELEVANT)
        assert valid == ["read_file"]
        assert set(invalid) == {"reverse_string", "un"}
        assert oos == []  # uydurma isimler out_of_scope'a girmez

    def test_gds_act_yalniz_gecerli_uzerinden(self):
        # 2 gecerli (1 scope-disi) + 1 uydurma -> act = 1/2 = 0.5 (uydurma haric)
        valid, invalid, oos = agentic.partition_actions(
            ["read_file", "deploy_app", "made_up_tool"], self.OFFERED, self.RELEVANT)
        act = len(oos) / len(valid)
        assert act == 0.5
