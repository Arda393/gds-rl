"""GRPO egitim hiperparametreleri (Asama 5).

Tum ayarlar burada; hicbir yere hard-code edilmez. Doc bolum 5.5'teki
degerler temel alinmistir; CLI'dan override edilebilir (grpo_trainer.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GRPOTrainingConfig:
    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "results/checkpoints/qwen1.5b_grpo"

    # --- GRPO cekirdek (doc 5.5) ---
    num_rollouts: int = 8          # G: her prompt icin uretilen cevap sayisi
    batch_size: int = 2            # prompt batch (A100'de artirılabilir)
    learning_rate: float = 1e-5    # kucuk LR: kararsiz convergence onlenir
    kl_beta: float = 0.04          # referans modelden uzaklasmayi sinirlar
    epochs: int = 3                # 3 gecis: overfitting onlenir
    max_new_tokens: int = 512      # rollout cevap uzunlugu limiti

    # --- Reward agirliklari (v3) ---
    # reward = w_drift*(1-GDS) + w_task*task_score
    # task_score = F1(tool secimi) * done-kapisi * (1 - 0.5*repetition)
    # F1: hem az-is (recall) hem asiri-is/halusinasyon (precision) hem tekrari kapatir.
    w_drift: float = 0.5
    w_task: float = 0.5

    # --- Sampling ---
    temperature: float = 0.9       # rollout cesitliligi (GRPO icin yuksek)
    top_p: float = 1.0

    # --- Agentic-plan reward formati ---
    # Egitimde model tek turda tum aksiyon planini uretir (GRPO-uyumlu);
    # reward, plandan parse edilen aksiyonlarla 3-bilesenli GDS hesaplar.
    n_offtopic_tools: int = 4      # tuzak: scope-disi tool sayisi
    n_dangerous_tools: int = 3     # tuzak: cazip tehlikeli tool sayisi

    # --- Veri / calistirma ---
    benchmark_limit: int | None = None  # None = tum split; smoke icin 50
    train_split: str = "train"     # egitim sadece train split'inde (held-out test korunur)
    dtype: str = "bfloat16"        # A100/H100/V100 (P100'de fp16 patlar!)
    seed: int = 42

    # --- LoRA (buyuk modeller icin: full-FT 80GB'ye sigmaz) ---
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    grad_checkpointing: bool = False  # aktivasyon OOM'u icin (orn. cok-modlu Gemma)

    # --- Loglama ---
    use_wandb: bool = True
    wandb_project: str = "gds-rl"
    log_every: int = 10            # her N adimda GDS ortalamasini logla
    save_every: int = 100

    def __post_init__(self):
        # LoRA kullanan (buyuk) modeller 8-rollout aktivasyonuyla OOM olur;
        # gradient checkpointing belleği dusurur -> LoRA varsa daima ac.
        if self.use_lora:
            self.grad_checkpointing = True

    def reward_weights(self) -> tuple[float, float]:
        return self.w_drift, self.w_task


# Modele gore hizli on-ayarlar (grpo_trainer.py --preset ile secilir)
PRESETS = {
    "qwen1.5b": dict(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="results/checkpoints/qwen1.5b_grpo",
        batch_size=4,
    ),
    "qwen7b": dict(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="results/checkpoints/qwen7b_grpo",
        batch_size=2,        # 7B daha fazla bellek
        use_lora=True,       # full-FT 80GB'ye sigmaz -> LoRA
    ),
    "gemma4b": dict(
        model_name="google/gemma-3-4b-it",
        output_dir="results/checkpoints/gemma4b_grpo",
        batch_size=2,
        use_lora=True,            # cok-modlu + bellek -> LoRA
        grad_checkpointing=True,  # vision tower + 8 rollout aktivasyonu -> OOM, checkpointing sart
    ),
    # --- Yeni nesil modeller ---
    "qwen3_4b": dict(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        output_dir="results/checkpoints/qwen3_4b_grpo",
        batch_size=2,
        use_lora=True,
    ),
    "llama8b": dict(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="results/checkpoints/llama8b_grpo",
        batch_size=2,
        use_lora=True,
    ),
    "phi4_mini": dict(
        model_name="microsoft/Phi-4-mini-instruct",
        output_dir="results/checkpoints/phi4_mini_grpo",
        batch_size=2,
        use_lora=True,
    ),
    # Gemma boyut serisi (aile-ici olcek ekseni)
    "gemma1b": dict(
        model_name="google/gemma-3-1b-it",   # text-only (Gemma3ForCausalLM)
        output_dir="results/checkpoints/gemma1b_grpo",
        batch_size=2,
        use_lora=True,
    ),
    "gemma12b": dict(
        model_name="google/gemma-3-12b-it",  # cok-modlu, buyuk
        output_dir="results/checkpoints/gemma12b_grpo",
        batch_size=2,
        use_lora=True,
        grad_checkpointing=True,
    ),
}


# Ablation on-ayarlari (Asama 9): GDS bilesenlerinden yalniz biri aktif.
# reward.py bu agirliklari compute_gds'e gecirir.
ABLATIONS = {
    "full":      dict(w_emb=0.3, w_nli=0.3, w_act=0.4),
    "emb_only":  dict(w_emb=1.0, w_nli=0.0, w_act=0.0),
    "nli_only":  dict(w_emb=0.0, w_nli=1.0, w_act=0.0),
    "act_only":  dict(w_emb=0.0, w_nli=0.0, w_act=1.0),
}


def build_config(preset: str = "qwen1.5b", **overrides) -> GRPOTrainingConfig:
    base = dict(PRESETS.get(preset, {}))
    base.update(overrides)
    return GRPOTrainingConfig(**base)
