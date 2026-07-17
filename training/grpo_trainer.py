"""GRPO egitim dongusu (Asama 6) - HuggingFace TRL GRPOTrainer.

Model her prompt'a G=8 cevap (aksiyon plani) uretir, reward.py her cevap icin
GDS tabanli reward hesaplar, GRPO grup-goreli advantage ile politikayi gunceller.
KL penalty (beta) referans modelden cok uzaklasmayi onler.

Kullanim:
    python training/grpo_trainer.py --preset qwen1.5b --limit 50      # smoke
    python training/grpo_trainer.py --preset qwen1.5b                 # tam
    python training/grpo_trainer.py --preset qwen7b --ablation act_only
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

from training.config import ABLATIONS, build_config
from training.reward import make_reward_func

# agentic_eval'den tuzak-tool havuzu ureticisini yeniden kullan
from agentic_eval import build_tool_set, load_tool_descriptions
from evaluate import load_tasks
from plan_prompt import build_plan_messages  # egitim ve eval AYNI prompt


def build_dataset(cfg, seed):
    """Benchmark gorevlerini GRPO dataset'ine cevirir.

    Her satir: prompt (mesaj listesi), goal, relevant_actions, offered_tools.
    """
    from datasets import Dataset

    all_tools, dangerous = load_tool_descriptions()
    tasks = load_tasks(cfg.benchmark_limit, split=cfg.train_split)

    rows = {"prompt": [], "goal": [], "relevant_actions": [], "offered_tools": []}
    for task in tasks:
        # gorev-basina deterministik tuzak seti (eval ile ayni)
        rng = random.Random(f"{seed}:{task['id']}")
        tools = build_tool_set(task, all_tools, dangerous,
                               cfg.n_offtopic_tools, cfg.n_dangerous_tools, rng)
        rows["prompt"].append(build_plan_messages(task["prompt"], tools))
        rows["goal"].append(task["prompt"])
        rows["relevant_actions"].append(task["relevant_actions"])
        rows["offered_tools"].append([n for n, _ in tools])

    return Dataset.from_dict(rows)


def main():
    ap = argparse.ArgumentParser(description="GDS-RL GRPO egitici")
    ap.add_argument("--preset", default="qwen1.5b",
                    choices=["qwen1.5b", "qwen7b", "gemma4b", "qwen3_4b", "llama8b",
                             "phi4_mini", "gemma1b", "gemma12b"])
    ap.add_argument("--ablation", default="full", choices=list(ABLATIONS))
    ap.add_argument("--limit", type=int, default=None, help="gorev limiti (smoke icin)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.limit is not None:
        overrides["benchmark_limit"] = args.limit
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.no_wandb:
        overrides["use_wandb"] = False
    if args.ablation != "full":
        overrides["output_dir"] = f"results/checkpoints/{args.preset}_grpo_{args.ablation}"
    cfg = build_config(args.preset, **overrides)

    import torch
    from datasets import disable_caching
    from trl import GRPOConfig, GRPOTrainer

    disable_caching()
    gds_w = ABLATIONS[args.ablation]
    gds_weights = (gds_w["w_emb"], gds_w["w_nli"], gds_w["w_act"])
    print(f"[cfg] preset={args.preset} ablation={args.ablation} gds_weights={gds_weights}", flush=True)
    print(f"[cfg] model={cfg.model_name} rollouts={cfg.num_rollouts} lr={cfg.learning_rate} kl={cfg.kl_beta}", flush=True)

    dataset = build_dataset(cfg, cfg.seed)
    print(f"[data] {len(dataset)} gorev (split={cfg.train_split})", flush=True)

    gds_log: list = []
    reward_func = make_reward_func(
        weights=cfg.reward_weights(), gds_weights=gds_weights, log_sink=gds_log,
    )

    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.num_rollouts,  # 1 prompt x G generation / device
        gradient_accumulation_steps=cfg.batch_size,
        num_generations=cfg.num_rollouts,
        learning_rate=cfg.learning_rate,
        beta=cfg.kl_beta,
        num_train_epochs=cfg.epochs,
        max_completion_length=cfg.max_new_tokens,
        max_prompt_length=1024,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        logging_steps=cfg.log_every,
        save_steps=cfg.save_every,
        bf16=(cfg.dtype == "bfloat16"),
        fp16=(cfg.dtype == "float16"),
        gradient_checkpointing=cfg.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.grad_checkpointing else None,
        report_to=("wandb" if cfg.use_wandb else "none"),
        run_name=f"{args.preset}_{args.ablation}",
        seed=cfg.seed,
    )

    # LoRA: buyuk modeller full-FT'de 80GB'ye sigmaz; ayrica peft ile GRPO
    # ayri referans model tutmaz (adapter'i kapatip referans logprob alir).
    peft_config = None
    if cfg.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            # all-linear: mimariden bagimsiz (Qwen q_proj, Phi qkv_proj, vs. hepsi)
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        print(f"[lora] aktif: r={cfg.lora_r} alpha={cfg.lora_alpha}", flush=True)

    # Cok-modlu mimari (orn. Gemma-3 4B/12B = Gemma3ForConditionalGeneration);
    # TRL'in AutoModelForCausalLM yolu 'use_cache' ile patlar -> instance verelim.
    # Mimariye gore tespit: gemma-3-1b text-only (Gemma3ForCausalLM) -> normal yol.
    from transformers import AutoConfig

    arch = (AutoConfig.from_pretrained(cfg.model_name).architectures or [""])[0]
    model_arg = cfg.model_name
    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        from transformers import AutoModelForImageTextToText

        m = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name, torch_dtype=getattr(torch, cfg.dtype))
        m.config.use_cache = False
        model_arg = m
        print(f"[model] cok-modlu ({arch}) instance olarak yuklendi", flush=True)

    # Bazi modellerin (orn. Llama-3.1) tokenizer'inda pad_token yok; TRL kendi
    # yukledigi tokenizer'da bunu ayarlamaz -> acikca verip pad'i set edelim.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[tok] pad_token yoktu -> eos_token atandi", flush=True)

    trainer = GRPOTrainer(
        model=model_arg,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_func],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("[train] basliyor...", flush=True)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[train] tamamlandi -> {cfg.output_dir}", flush=True)

    # Egitim egrisini kaydet (Asama 10 training curve grafigi icin)
    import json
    curve_path = Path(cfg.output_dir) / "training_curve.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump({
            "preset": args.preset, "ablation": args.ablation,
            "gds_weights": gds_weights, "reward_weights": cfg.reward_weights(),
            "n_batches": len(gds_log), "curve": gds_log,
        }, f, indent=2)
    print(f"[train] egri kaydedildi -> {curve_path}", flush=True)

    if gds_log:
        first = sum(d["mean_gds"] for d in gds_log[:10]) / min(10, len(gds_log))
        last = sum(d["mean_gds"] for d in gds_log[-10:]) / min(10, len(gds_log))
        print(f"[train] GDS ortalama: ilk~{first:.3f} -> son~{last:.3f} "
              f"({'DUSTU (iyi)' if last < first else 'DUSMEDI'})", flush=True)


if __name__ == "__main__":
    main()
