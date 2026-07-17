"""GoalDrift-Bench degerlendirici (Asama 3: baseline olcumleri).

Bir HF modelini yukler, benchmark gorevlerine cevap urettirir, her cevap
icin GDS hesaplar ve sonuclari JSON olarak kaydeder.

Baseline'da aksiyon izi yoktur; GDS fallback formuluyle hesaplanir:
    GDS = 0.5 * GDS_emb + 0.5 * GDS_nli

Task completion icin simdilik seffaf bir NLI proxy'si kaydedilir
(P(entailment) > 0.5). Gercek completion checker Asama 4'te (reward.py)
gelecek; proxy oldugu sonuc dosyasinda acikca isaretlenir.

Kullanim:
    python benchmark/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --out results/baseline_qwen1.5b.json [--limit 20] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gds.embedding import embedding_drift
from gds.nli import nli_probs

TASKS_DIR = PROJECT_ROOT / "benchmark" / "tasks"


def load_tasks(limit: int | None = None, split: str | None = None) -> list[dict]:
    """Gorevleri kategori dengeli sekilde yukler.

    split: None (hepsi), "train" ya da "test" (benchmark/splits.json'a gore).
    limit: kategori-dengeli ust sinir (smoke testleri icin).
    """
    split_ids: set[str] | None = None
    if split is not None:
        split_path = TASKS_DIR.parent / "splits.json"
        with open(split_path, encoding="utf-8") as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"bilinmeyen split '{split}' (train/test bekleniyor)")
        split_ids = set(splits[split])

    by_cat: dict[str, list[dict]] = {}
    for path in sorted(TASKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            task = json.load(f)
        if split_ids is not None and task["id"] not in split_ids:
            continue
        by_cat.setdefault(task["category"], []).append(task)

    if limit is None:
        return [t for tasks in by_cat.values() for t in tasks]

    per_cat = max(1, limit // len(by_cat))
    selected = []
    for tasks in by_cat.values():
        selected.extend(tasks[:per_cat])
    return selected[:limit]


def _load_model(model_name: str, dtype: str):
    """Mimariye gore dogru model sinifini secer.

    - LoRA adapter checkpoint'i (adapter_config.json varsa): base model + adapter.
    - Gemma-3 gibi cok-modlu (Gemma3ForConditionalGeneration): AutoModelForImageTextToText.
    - Digerleri: AutoModelForCausalLM.
    """
    from pathlib import Path as _Path

    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = dict(torch_dtype=getattr(torch, dtype), device_map="auto")

    # LoRA adapter checkpoint mi?
    adapter_cfg = _Path(model_name) / "adapter_config.json"
    if adapter_cfg.exists():
        import json as _json

        from peft import PeftModel

        base_name = _json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        print(f"[model] LoRA adapter -> base: {base_name}", flush=True)
        base = _load_model(base_name, dtype)  # base'i dogru sinifla yukle (rekursif)
        model = PeftModel.from_pretrained(base, model_name)
        return model.merge_and_unload()  # adapter'i base'e gomup tek model dondur

    config = AutoConfig.from_pretrained(model_name)
    arch = (config.architectures or [""])[0]

    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        from transformers import AutoModelForImageTextToText

        print(f"[model] cok-modlu mimari ({arch}) -> AutoModelForImageTextToText", flush=True)
        return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def generate_responses(model_name: str, prompts: list[str], batch_size: int,
                       max_new_tokens: int, dtype: str, seed: int) -> list[str]:
    from transformers import (
        AutoTokenizer,
        InfNanRemoveLogitsProcessor,
        LogitsProcessorList,
        set_seed,
    )

    set_seed(seed)
    print(f"[model] yukleniyor: {model_name} (dtype={dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_name, dtype)
    model.eval()
    n_gpu = torch.cuda.device_count()
    print(f"[model] hazir. GPU sayisi: {n_gpu}", flush=True)

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]

    responses = []
    t_start = time.time()
    for i in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                # P100'de bf16 yok; fp16 overflow'lari sampling'i patlatmasin
                logits_processor=LogitsProcessorList([InfNanRemoveLogitsProcessor()]),
                renormalize_logits=True,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        responses.extend(d.strip() for d in decoded)
        done = min(i + batch_size, len(chat_prompts))
        rate = done / (time.time() - t_start)
        print(f"[gen] {done}/{len(chat_prompts)} ({rate:.2f} gorev/sn)", flush=True)

    del model
    torch.cuda.empty_cache()
    return responses


def score_responses(tasks: list[dict], responses: list[str]) -> list[dict]:
    """Her (gorev, cevap) cifti icin GDS fallback skoru + NLI proxy hesaplar."""
    records = []
    for task, response in zip(tasks, responses):
        goal = task["prompt"]
        if not response.strip():
            # bos cevap = tam drift sayilir, NLI'ye sokma
            records.append({
                "id": task["id"], "category": task["category"],
                "response": response, "gds": 1.0, "emb": 1.0, "nli": 1.0,
                "completion_proxy_nli": False, "empty_response": True,
            })
            continue
        emb = embedding_drift(goal, response)
        probs = nli_probs(goal, response)
        nli = min(1.0, probs["contradiction"] + 0.5 * probs["neutral"])
        gds = 0.5 * emb + 0.5 * nli
        records.append({
            "id": task["id"],
            "category": task["category"],
            "response": response,
            "gds": round(gds, 4),
            "emb": round(emb, 4),
            "nli": round(nli, 4),
            "completion_proxy_nli": probs["entailment"] > 0.5,
            "empty_response": False,
        })
    return records


def summarize(records: list[dict]) -> dict:
    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n_tasks": len(records),
        "avg_gds": avg([r["gds"] for r in records]),
        "avg_emb": avg([r["emb"] for r in records]),
        "avg_nli": avg([r["nli"] for r in records]),
        "completion_proxy_rate": avg([float(r["completion_proxy_nli"]) for r in records]),
        "per_category": {},
    }
    categories = sorted({r["category"] for r in records})
    for cat in categories:
        cat_recs = [r for r in records if r["category"] == cat]
        summary["per_category"][cat] = {
            "n": len(cat_recs),
            "avg_gds": avg([r["gds"] for r in cat_recs]),
            "completion_proxy_rate": avg([float(r["completion_proxy_nli"]) for r in cat_recs]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="GoalDrift-Bench baseline degerlendirici")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--out", required=True, help="Sonuc JSON dosyasi")
    parser.add_argument("--limit", type=int, default=None, help="Gorev sayisi limiti (smoke test)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = load_tasks(args.limit)
    print(f"[bench] {len(tasks)} gorev yuklendi", flush=True)

    t0 = time.time()
    responses = generate_responses(
        args.model, [t["prompt"] for t in tasks],
        args.batch_size, args.max_new_tokens, args.dtype, args.seed,
    )
    gen_time = time.time() - t0
    print(f"[gen] tamamlandi: {gen_time:.0f}s", flush=True)

    print("[gds] skorlaniyor...", flush=True)
    records = score_responses(tasks, responses)
    summary = summarize(records)

    result = {
        "model": args.model,
        "phase": "baseline",
        "gds_formula": "0.5*emb + 0.5*nli (fallback: aksiyon izi yok)",
        "completion_note": "completion_proxy_nli = P(entailment)>0.5, GERCEK COMPLETION DEGIL (Asama 4'te gelecek)",
        "config": {
            "max_new_tokens": args.max_new_tokens, "temperature": 0.7,
            "top_p": 0.9, "dtype": args.dtype, "seed": args.seed,
            "batch_size": args.batch_size,
        },
        "generation_seconds": round(gen_time, 1),
        "summary": summary,
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[sonuc] {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
