"""GDS tabanli reward fonksiyonu (Asama 4).

GRPO her rollout icin bir reward ister. Model tek turda gorev + tool listesine
karsi bir AKSIYON PLANI uretir (ACTION: ... satirlari + DONE). Reward bu plandan
3-bilesenli GDS hesaplar:

    reward = lambda1 * (1 - GDS) + lambda2 * completion - lambda3 * GDS^2
    (varsayilan: 0.6, 0.3, 0.1 -- doc 5.2)

GDS_act yalniz SUNULAN tool'lar uzerinden olculur; uydurma isimler drift
sayilmaz (agentic_eval ile ayni metodoloji).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gds.score import compute_gds

ACTION_RE = re.compile(r"ACTION:\s*([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
DONE_RE = re.compile(r"\bDONE\b")


def parse_actions(text: str) -> tuple[list[str], bool]:
    """Plandan tum ACTION: tool isimlerini ve DONE isaretini cikarir."""
    actions = [m.lower() for m in ACTION_RE.findall(text)]
    done = bool(DONE_RE.search(text))
    return actions, done


def repetition_rate(actions: list[str]) -> float:
    """Ardisik ayni aksiyon orani (0-1). Dongu/tekrar dejenerasyonunu cezalar."""
    if len(actions) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return repeats / (len(actions) - 1)


def tool_prf(all_actions: list[str], relevant: set[str]) -> tuple[float, float, float]:
    """Tool secimi icin precision / recall / F1.

    recall    = cagrilan farkli relevant / toplam relevant  (az-is -> dusuk)
    precision = cagrilan farkli relevant / toplam farkli cagri  (asiri/uydurma -> dusuk)
    Bu tek F1 terimi hem az-is hem asiri-is hem halusinasyonu birden cezalar.
    """
    distinct = set(all_actions)
    if not relevant:
        return 1.0, 1.0, 1.0
    in_scope = distinct & relevant
    recall = len(in_scope) / len(relevant)
    precision = len(in_scope) / len(distinct) if distinct else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def task_score(f1: float, done: bool, rep: float) -> float:
    """Gercek gorev-basari sinyali: F1, temiz-bitirme kapisi, tekrar cezasi."""
    return f1 * (1.0 if done else 0.6) * (1.0 - 0.5 * rep)


def compute_reward(
    goal: str,
    completion: str,
    relevant_actions: list[str],
    offered_tools: list[str],
    weights: tuple[float, float] = (0.5, 0.5),
    gds_weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> dict:
    """Tek bir rollout icin reward + kirilim dondurur.

    weights = (w_drift, w_task)
        reward = w_drift*(1-GDS) + w_task*task_score
    task_score = F1(tool secimi) * done-kapisi * (1 - 0.5*repetition)
    Bu tasarim gozlemlenen TUM dejenerasyonlari kapatir:
      az-is (dusuk recall), asiri-is/halusinasyon (dusuk precision),
      tekrar (rep cezasi), hic-bitirmeme (done kapisi).
    """
    actions, done = parse_actions(completion)
    offered = {t.lower() for t in offered_tools}
    relevant = {r.lower() for r in relevant_actions}

    valid = [a for a in actions if a in offered]
    in_scope = [a for a in valid if a in relevant]

    w_emb, w_nli, w_act = gds_weights
    try:
        if valid:
            res = compute_gds(goal, completion, actions=valid,
                              relevant_actions=list(relevant),
                              w_emb=w_emb, w_nli=w_nli, w_act=w_act)
            gds = res.gds
        else:
            if w_emb + w_nli == 0:
                gds = 1.0
            else:
                res = compute_gds(goal, completion, w_emb=w_emb, w_nli=w_nli)
                gds = res.gds
    except ValueError:
        gds = 1.0

    precision, recall, f1 = tool_prf(actions, relevant)
    rep = repetition_rate(actions)
    tscore = task_score(f1, done, rep)

    w_drift, w_task = weights
    reward = w_drift * (1.0 - gds) + w_task * tscore

    return {
        "reward": float(reward),
        "gds": float(gds),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "repetition": float(rep),
        "task_score": float(tscore),
        "task_done": bool(done and len(in_scope) >= 1),
        "n_actions": len(actions),
        "n_valid": len(valid),
        "n_in_scope": len(in_scope),
    }


def make_reward_func(weights=(0.5, 0.5), gds_weights=(0.3, 0.3, 0.4), log_sink=None):
    """TRL GRPOTrainer uyumlu reward fonksiyonu uretir.

    TRL cagrisi: reward_func(prompts, completions, **dataset_columns) -> list[float]
    Dataset 'goal', 'relevant_actions', 'offered_tools' sutunlarini tasimali.
    log_sink verilirse her batch'in ortalama GDS/reward/coverage'ini oraya yazar.
    """

    def reward_func(prompts, completions, goal=None, relevant_actions=None,
                    offered_tools=None, **kwargs):
        # completions TRL'de string ya da [{"role","content"}] olabilir
        texts = []
        for c in completions:
            if isinstance(c, list):
                texts.append(c[-1].get("content", ""))
            else:
                texts.append(c)

        goals = goal if goal is not None else prompts
        n = len(texts)
        rels = relevant_actions if relevant_actions is not None else [[]] * n
        offs = offered_tools if offered_tools is not None else [[]] * n

        rewards, gds_vals, f1_vals, done_vals = [], [], [], []
        for i in range(n):
            r = compute_reward(goals[i], texts[i], rels[i], offs[i], weights, gds_weights)
            rewards.append(r["reward"])
            gds_vals.append(r["gds"])
            f1_vals.append(r["f1"])
            done_vals.append(float(r["task_done"]))

        if log_sink is not None and gds_vals:
            log_sink.append({"mean_gds": sum(gds_vals) / n,
                             "mean_reward": sum(rewards) / n,
                             "mean_f1": sum(f1_vals) / n,
                             "mean_done": sum(done_vals) / n})
        return rewards

    return reward_func
