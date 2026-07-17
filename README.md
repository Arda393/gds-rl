# GDS-RL: Goal Drift Score + GRPO Training

Agentic LLM'lerde goal drift'i olcen (GDS) ve RL ile azaltan (GDS-RL) framework.
Detayli rehber: `GoalDrift_Proje_Rehberi.docx`

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanim

```python
from gds import compute_gds

result = compute_gds(
    goal="Python ile bir CSV dosyasini oku ve sadece fiyat sutununu dondur.",
    output="CSV dosyasini okudum, fiyat sutununu dondurdum.",
    actions=["read_file", "parse_csv", "return_data"],
    relevant_actions=["read_file", "parse_csv", "filter_column", "return_data"],
)
print(result)  # GDS=0.123 (emb=..., nli=..., act=...)
```

## Test

```
pytest tests/ -v          # birim testler
python experiments/demo_gds.py   # ornek senaryolarla demo
```

## Benchmark (GoalDrift-Bench)

200 gorev, 4 kategori x 50 (coding, research, tool_use, planning). Gorevler
**ingilizce** yazildi cunku GDS'nin NLI modeli (nli-MiniLM2-L6-H768) sadece
ingilizce egitildi; Turkce'de olumsuzluk celiskilerini kaciriyor.

Kaynak dosyalar `benchmark/source/*.json` icinde tutulur; tek tek gorev
dosyalari bunlardan uretilir:

```
python benchmark/build_tasks.py   # dogrular + tasks/ altina 200 dosya yazar
```

Gorev eklemek/duzenlemek icin source dosyasini degistir, build'i tekrar
calistir. Build ayrica `gds/action_taxonomy.json` kategori girislerini
gunceller.

## Klasor Yapisi

```
goal_drift/
├── gds/               # GDS metrigi (embedding + NLI + aksiyon takibi)
├── training/          # GRPO training pipeline (Asama 4-6)
├── benchmark/tasks/   # GoalDrift-Bench: 200 gorev JSON (Asama 2)
├── experiments/       # Deney scriptleri
├── tests/             # Birim testler
└── results/           # Checkpoint ve grafikler
```

## Durum

- [x] Asama 1: GDS metrigi (embedding.py, nli.py, action.py, score.py)
- [x] Asama 2: 200 gorevlik GoalDrift-Bench (ingilizce)
- [ ] Asama 3: Baseline olcumleri (Qwen2.5-1.5B/7B, Gemma-3-4B)
- [ ] Asama 4-6: reward.py, config.py, grpo_trainer.py
- [ ] Asama 7-9: Deneyler + ablation
- [ ] Asama 10: Sonuclar + paper

## TRUBA / SLURM kurulumu

`experiments/truba/*.slurm` betiklerinde iki yer tutucu vardir; kendi HPC
bilgilerinle degistir:

- `YOURUSER`    -> TRUBA kullanici adin  (dosya yollari: /arf/scratch/YOURUSER/...)
- `YOURACCOUNT` -> SLURM hesap adin      (#SBATCH -A YOURACCOUNT)

Ornek:
```
sed -i 's/YOURUSER/kullaniciadi/g; s/YOURACCOUNT/hesabin/g' experiments/truba/*.slurm
```
