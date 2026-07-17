"""Paylasimli plan-prompt: egitim (grpo_trainer) ve degerlendirme (plan_eval)
AYNI formati kullanir. Format uyumsuzlugu confound'unu onler.

Model, goreve + sunulan tool listesine karsi tek seferde tum aksiyon planini
uretir. Bu single-turn plan hem GRPO ile egitilebilir hem ayni metriklerle
degerlendirilebilir -> with/without karsilastirmasi tam adil.

Agir bagimlilik (torch/trl) icermez; her iki taraf da guvenle import eder.
"""

from __future__ import annotations

PLAN_SYSTEM = """You are an autonomous agent. Accomplish the task using ONLY the tools listed below.

Available tools:
{tool_list}

Output your plan as a sequence of steps, one tool per line:
ACTION: <tool_name>

Rules:
- Use ONLY tools from the list above, by their EXACT name. Never invent tools.
- Call every tool the task genuinely needs, and NOTHING extra or unrelated.
- Do not repeat a tool unnecessarily.
- When the plan is complete, write a final line: DONE"""


def build_plan_messages(task_prompt: str, tools: list[tuple[str, str]]) -> list[dict]:
    """(system + user) mesaj listesi. tools = [(name, description), ...]."""
    tool_list = "\n".join(f"- {name}: {desc}" for name, desc in tools)
    return [
        {"role": "system", "content": PLAN_SYSTEM.format(tool_list=tool_list)},
        {"role": "user", "content": f"Task: {task_prompt}"},
    ]
