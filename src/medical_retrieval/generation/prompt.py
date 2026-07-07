from __future__ import annotations

from ..retrieval.recall import HybridCandidate

SAFETY_NOTE = "本回答仅用于课程演示和医学知识检索，不替代医生诊断。若症状严重或持续，请及时就医。"


def build_context(candidates: list[HybridCandidate], max_chars: int = 1800) -> str:
    blocks: list[str] = []
    used = 0
    for idx, candidate in enumerate(candidates, 1):
        doc = candidate.doc
        block = (
            f"[{idx}] {doc.title}\n"
            f"科室：{doc.department}\n"
            f"标签：{', '.join(doc.tags[:10])}\n"
            f"内容：{doc.content}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_rag_prompt(query: str, candidates: list[HybridCandidate]) -> str:
    context = build_context(candidates)
    return f"""你是医学检索系统的回答模块。请只基于给定资料回答，不要编造诊断或药品剂量。

用户问题：{query}

检索资料：
{context}

回答要求：
1. 先直接回答用户最关心的问题。
2. 给出建议检查或就诊科室时，必须来自检索资料。
3. 列出引用来源编号。
4. 末尾加上安全提示：{SAFETY_NOTE}
"""
