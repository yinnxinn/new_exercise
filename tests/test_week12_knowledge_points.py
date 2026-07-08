"""第 12 周知识点测试：RAG、来源引用、安全边界与流式输出。

本周重点是把检索结果变成可交互产品：
1. RAG 回答必须基于检索上下文，而不是自由编造；
2. 医疗场景必须展示来源和安全提示；
3. 流式输出可以提升交互体验，本质是把完整答案分块返回；
4. 最后一个 chunk 需要 done 信号，前端据此结束 loading 状态。
"""

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


# 知识点 1：RAG prompt 要明确约束模型“只基于资料回答”。
# 医疗系统还要显式禁止编造诊断、药品剂量等高风险内容。
def test_week12_rag_prompt_requires_grounding_and_safety_boundary():
    prompt = build_rag_prompt("咳嗽发热需要做什么检查", [_candidate()])

    assert "只基于给定资料回答" in prompt
    assert "不要编造诊断或药品剂量" in prompt
    assert "不替代医生诊断" in prompt


# 知识点 2：面向用户的医疗回答需要引用来源和免责声明。
# 这样学生能理解“答案生成”不只是拼文本，还要处理可信度和产品边界。
def test_week12_answer_contains_source_and_medical_disclaimer():
    answer = build_grounded_answer("咳嗽发热需要做什么检查", [_candidate()])

    assert "引用来源" in answer
    assert "PNE-肺炎" in answer
    assert "不替代医生诊断" in answer


# 知识点 3：流式输出 = 多个增量 chunk + 一个结束信号。
# 前端收到 done=True 后，才能停止“正在生成”的状态。
def test_week12_streaming_sends_incremental_chunks_then_done_signal():
    chunks = list(stream_answer("医学检索系统", chunk_size=2))

    assert [chunk.text for chunk in chunks[:-1]] == ["医学", "检索", "系统"]
    assert chunks[-1].text == ""
    assert chunks[-1].done is True
