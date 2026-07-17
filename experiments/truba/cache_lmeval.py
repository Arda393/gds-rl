"""lm-eval'in ihtiyac duydugu dataset'leri HF cache'e indirir (UI node, online).

get_task_dict task'lari kurarken her dataset'i tam dogru path/config ile indirir
-> compute node'da offline kosabiliriz. Inline heredoc tirnak karmasasi yok.
"""
import sys

import lm_eval.tasks as tasks

want = ["ifeval", "gsm8k", "mmlu"]
tm = tasks.TaskManager()
ok = []
for t in want:
    try:
        td = tasks.get_task_dict([t], tm)
        ok.append(t)
        print(f"CACHED_OK: {t} -> {list(td.keys())[:3]}", flush=True)
    except Exception as e:
        print(f"CACHED_FAIL: {t} {type(e).__name__} {str(e)[:160]}", flush=True)

print("ALL_CACHED" if len(ok) == len(want) else f"PARTIAL ({len(ok)}/{len(want)})", flush=True)
sys.exit(0 if len(ok) == len(want) else 1)
