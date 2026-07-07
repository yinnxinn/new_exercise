from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..retrieval.recall import HybridCandidate
from .prompt import SAFETY_NOTE


@dataclass(frozen=True)
class AnswerChunk:
    text: str
    done: bool = False


def build_grounded_answer(query: str, candidates: list[HybridCandidate]) -> str:
    if not candidates:
        return f"没有检索到足够可靠的资料来回答：{query}\n\n{SAFETY_NOTE}"

    top = candidates[0].doc
    lines = [
        f"针对“{query}”，当前最相关的资料指向：{top.title}。",
        f"建议优先关注科室：{top.department}。",
    ]

    if top.tags:
        lines.append(f"相关关键词：{', '.join(top.tags[:8])}。")
    if top.content:
        lines.append(f"资料要点：{top.content}")

    if len(candidates) > 1:
        related = "；".join(f"[{idx}] {item.doc.title}/{item.doc.department}" for idx, item in enumerate(candidates[1:4], 2))
        lines.append(f"其他可参考资料：{related}。")

    sources = ", ".join(f"[{idx}] {item.doc.id}-{item.doc.title}" for idx, item in enumerate(candidates[:5], 1))
    lines.append(f"引用来源：{sources}。")
    lines.append(SAFETY_NOTE)
    return "\n".join(lines)


def stream_answer(answer: str, chunk_size: int = 18) -> Iterable[AnswerChunk]:
    for start in range(0, len(answer), chunk_size):
        yield AnswerChunk(answer[start:start + chunk_size], done=False)
    yield AnswerChunk("", done=True)
