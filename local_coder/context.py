"""Keep conversations inside a small context window (important on 8 GB laptops)."""
from __future__ import annotations

from .codeutils import estimate_tokens


def fit_messages(messages: list[dict], num_ctx: int, reserve_for_answer: int = 1024) -> tuple[list[dict], int]:
    """Drop the oldest turns until the prompt fits in num_ctx minus the answer reserve.

    The system message and the newest user message are always kept. Returns
    (messages, number_of_dropped_messages).
    """
    budget = max(256, num_ctx - reserve_for_answer)
    if not messages:
        return [], 0
    system = [m for m in messages[:1] if m["role"] == "system"]
    rest = messages[len(system):]
    if not rest:
        return system, 0
    newest, older = rest[-1], rest[:-1]
    used = sum(estimate_tokens(m["content"]) + 4 for m in system + [newest])
    kept: list[dict] = []
    for msg in reversed(older):
        cost = estimate_tokens(msg["content"]) + 4
        if used + cost > budget:
            break
        kept.insert(0, msg)
        used += cost
    # never start the kept history with a lonely assistant reply
    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)
    dropped = len(older) - len(kept)
    return system + kept + [newest], dropped
