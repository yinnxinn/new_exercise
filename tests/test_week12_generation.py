from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.generation.answer import build_grounded_answer, stream_answer
from medical_retrieval.retrieval.recall import HybridCandidate


def test_grounded_answer_contains_source_and_safety_note():
    candidate = HybridCandidate(
        doc=MedicalDocument("D1", "肺炎", "呼吸内科", ["咳嗽", "发热"], "咳嗽发热需要检查血常规和胸部影像。"),
        score=1.0,
        sources=["bm25", "vector"],
        matched_entities=["咳嗽", "发热"],
    )

    answer = build_grounded_answer("咳嗽发热需要做什么检查", [candidate])

    assert "肺炎" in answer
    assert "引用来源" in answer
    assert "不替代医生诊断" in answer


def test_stream_answer_marks_done():
    chunks = list(stream_answer("abcdef", chunk_size=2))

    assert [chunk.text for chunk in chunks[:-1]] == ["ab", "cd", "ef"]
    assert chunks[-1].done is True
