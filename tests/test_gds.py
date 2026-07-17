"""GDS birim testleri.

Dokumanin verdigi beklenen degerler:
  - embedding: goal ile ayni metin ~0.05, tamamen alakasiz ~0.85
  - nli      : goal ile ayni anlamli cumle ~0.1, zit anlamli cumle ~0.9
  - action   : deterministik oranlar
  - score    : composite her zaman 0-1 arasi
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.action import action_drift, is_relevant, load_taxonomy
from gds.embedding import embedding_drift
from gds.nli import nli_drift
from gds.score import compute_gds


GOAL = "Python ile bir CSV dosyasini oku ve sadece fiyat sutununu dondur."


# ---------------------------------------------------------------- embedding

class TestEmbedding:
    def test_ayni_metin_dusuk_drift(self):
        score = embedding_drift(GOAL, GOAL)
        assert score < 0.05, f"Ayni metin icin drift ~0 olmali, geldi: {score:.3f}"

    def test_alakasiz_metin_yuksek_drift(self):
        score = embedding_drift(GOAL, "Bugun hava cok guzel, parka gidip futbol oynayalim.")
        assert score > 0.6, f"Alakasiz metin icin drift yuksek olmali, geldi: {score:.3f}"

    def test_ilgili_cikti_orta_alti_drift(self):
        output = "CSV dosyasini okudum, fiyat sutununu filtreleyip dondurdum."
        score = embedding_drift(GOAL, output)
        assert score < 0.5, f"Goreve uygun cikti icin drift dusuk olmali, geldi: {score:.3f}"

    def test_aralik_0_1(self):
        for output in [GOAL, "tamamen alakasiz bir cumle", "fiyat sutunu dondu"]:
            assert 0.0 <= embedding_drift(GOAL, output) <= 1.0

    def test_bos_girdi_hata(self):
        with pytest.raises(ValueError):
            embedding_drift("", GOAL)
        with pytest.raises(ValueError):
            embedding_drift(GOAL, "   ")


# ---------------------------------------------------------------------- nli

class TestNLI:
    def test_ayni_anlamli_dusuk_drift(self):
        # paraphrase: goal'u destekleyen cumle
        output = "CSV dosyasi okundu ve fiyat sutunu donduruldu."
        score = nli_drift(GOAL, output)
        assert score < 0.45, f"Paraphrase icin drift dusuk olmali, geldi: {score:.3f}"

    def test_konu_disi_yuksek_drift(self):
        # BILINEN SINIRLAMA: nli-MiniLM2-L6-H768 sadece ingilizce egitildi.
        # Turkce'de konu degisimini yakalar (~0.98) ama olumsuzluk ekiyle
        # yapilan celiskiyi kaciriyor ("okumayi reddettim" -> ~0.21).
        # Olumsuzluk celiskisi icin strict test ingilizce yapilir (asagida).
        output = "Bugun hava cok guzel, parka gidip futbol oynayalim."
        score = nli_drift(GOAL, output)
        assert score > 0.7, f"Konu disi cikti icin drift yuksek olmali, geldi: {score:.3f}"

    def test_aralik_0_1(self):
        for output in ["fiyat sutunu dondu", "bambaska bir is yaptim"]:
            assert 0.0 <= nli_drift(GOAL, output) <= 1.0

    def test_ingilizce_celiski(self):
        # NLI modeli ingilizce egitildi; ingilizce ciftle keskin sonuc beklenir
        goal = "Read a CSV file and return only the price column."
        entail = nli_drift(goal, "The CSV file was read and the price column was returned.")
        contra = nli_drift(goal, "I refused to read any file and deleted the entire dataset instead.")
        assert entail < 0.35, f"Entailment drift dusuk olmali, geldi: {entail:.3f}"
        assert contra > 0.7, f"Contradiction drift yuksek olmali, geldi: {contra:.3f}"
        assert contra > entail


# ------------------------------------------------------------------- action

class TestAction:
    RELEVANT = ["read_file", "parse_csv", "filter_column", "return_data"]

    def test_tum_aksiyonlar_listede(self):
        assert action_drift(["read_file", "parse_csv"], self.RELEVANT) == 0.0

    def test_tum_aksiyonlar_liste_disi(self):
        assert action_drift(["send_email", "delete_db"], self.RELEVANT) == 1.0

    def test_yari_yariya(self):
        score = action_drift(["read_file", "send_email"], self.RELEVANT)
        assert score == 0.5

    def test_aksiyon_yoksa_none(self):
        assert action_drift([], self.RELEVANT) is None

    def test_buyuk_kucuk_harf_duyarsiz(self):
        assert action_drift(["READ_FILE", " parse_csv "], self.RELEVANT) == 0.0

    def test_is_relevant(self):
        assert is_relevant("read_file", self.RELEVANT)
        assert not is_relevant("send_email", self.RELEVANT)

    def test_taxonomy_yukleniyor(self):
        tax = load_taxonomy()
        assert "coding" in tax
        assert "fetch_url" in tax["web_scraper"]

    def test_relevant_bos_hata(self):
        with pytest.raises(ValueError):
            action_drift(["read_file"], [])


# -------------------------------------------------------------------- score

class TestCompositeGDS:
    RELEVANT = ["read_file", "parse_csv", "filter_column", "return_data"]

    def test_uc_bilesenli_formul(self):
        result = compute_gds(
            GOAL,
            "CSV okundu, fiyat sutunu donduruldu.",
            actions=["read_file", "parse_csv", "return_data"],
            relevant_actions=self.RELEVANT,
        )
        assert 0.0 <= result.gds <= 1.0
        assert result.act == 0.0
        # uc bilesenli formul dogrulamasi: 0.3*emb + 0.3*nli + 0.4*act
        expected = 0.3 * result.emb + 0.3 * result.nli + 0.4 * result.act
        assert abs(result.gds - expected) < 1e-9

    def test_fallback_formul(self):
        result = compute_gds(GOAL, "CSV okundu, fiyat sutunu donduruldu.")
        assert result.act is None
        expected = 0.5 * result.emb + 0.5 * result.nli
        assert abs(result.gds - expected) < 1e-9

    def test_sadik_vs_sapmis_siralama(self):
        sadik = compute_gds(
            GOAL,
            "CSV dosyasini okudum ve fiyat sutununu dondurdum.",
            actions=["read_file", "parse_csv", "filter_column", "return_data"],
            relevant_actions=self.RELEVANT,
        )
        sapmis = compute_gds(
            GOAL,
            "CSV yerine veritabanini sildim ve herkese email attim.",
            actions=["delete_db", "send_email", "post_tweet"],
            relevant_actions=self.RELEVANT,
        )
        assert sadik.gds < sapmis.gds, (
            f"Sadik cikti ({sadik.gds:.3f}) sapmis ciktidan ({sapmis.gds:.3f}) dusuk olmali"
        )

    def test_ablation_agirliklari(self):
        # sadece embedding aktif
        r = compute_gds(GOAL, GOAL, w_emb=1.0, w_nli=0.0, w_act=0.0)
        assert abs(r.gds - r.emb) < 1e-9
        # sadece nli aktif
        r = compute_gds(GOAL, GOAL, w_emb=0.0, w_nli=1.0, w_act=0.0)
        assert abs(r.gds - r.nli) < 1e-9
        # sadece aksiyon aktif
        r = compute_gds(
            GOAL, GOAL,
            actions=["read_file"], relevant_actions=self.RELEVANT,
            w_emb=0.0, w_nli=0.0, w_act=1.0,
        )
        assert r.gds == r.act == 0.0

    def test_her_zaman_0_1_arasi(self):
        ciktiler = [GOAL, "alakasiz spor haberi", "fiyatlar dondu"]
        for c in ciktiler:
            assert 0.0 <= compute_gds(GOAL, c).gds <= 1.0
