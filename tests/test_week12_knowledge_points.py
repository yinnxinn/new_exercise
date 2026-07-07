"""Week 12 knowledge tests: RAG, source grounding, and streaming output."""

from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.generation.answer import build_grounded_answer, stream_answer
from medical_retrieval.generation.prompt import build_rag_prompt
from medical_retrieval.retrieval.recall import HybridCandidate


def _candidate() -> HybridCandidate:
    return HybridCandidate(
        doc=MedicalDocument("PNE", "肺炎", "呼吸内科", ["咳嗽", "发热", "胸部影像"], "咳嗽发热可完善血常规和胸部影像。"),
        score=1.0,
        sources=["bm25", "vector"],
        matched_entities=["咳嗽", "发热"],
    )


def test_week12_rag_prompt_requires_grounding_and_safety_boundary():
    """Knowledge point: medical RAG prompts must constrain answer scope."""
    prompt = build_rag_prompt("咳嗽发热需要做什么检查", [_candidate()])

    assert "只基于给定资料回答" in prompt
    assert "不要编造诊断或药品剂量" in prompt
    assert "不替代医生诊断" in prompt


def test_week12_answer_contains_source_and_medical_disclaimer():
    """Knowledge point: user-facing medical answers need citations and disclaimers."""
    answer = build_grounded_answer("咳嗽发热需要做什么检查", [_candidate()])

    assert "引用来源" in answer
    assert "PNE-肺炎" in answer
    assert "不替代医生诊断" in answer


def test_week12_streaming_sends_incremental_chunks_then_done_signal():
    """Knowledge point: streaming output is chunked delivery plus an end signal."""
    chunks = list(stream_answer("医学检索系统", chunk_size=2))

    assert [chunk.text for chunk in chunks[:-1]] == ["医学", "检索", "系统"]
    assert chunks[-1].text == ""
    assert chunks[-1].done is True
