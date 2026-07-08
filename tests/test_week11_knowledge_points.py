"""第 11 周知识点测试：查询理解、多路召回与融合排序。

本周重点是把“一个搜索算法”升级成“召回流程”：
1. 先理解用户问题，包括意图、实体、同义词或医学缩写扩展；
2. 再用 BM25、向量检索、实体匹配等多路召回候选文档；
3. 最后用 RRF 这类方法融合不同召回源的排名；
4. 结果需要说明来自哪些召回通道，便于调试和讲解。
"""

from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.embeddings import build_embedding_model
from medical_retrieval.query.understanding import analyze_query
from medical_retrieval.retrieval.recall import HybridRecallPipeline, RecallResult, reciprocal_rank_fusion


# 知识点 1：召回前先做 Query Understanding。
# “甲亢心慌怕热要查什么”不仅是关键词匹配，还应识别检查意图，并扩展到 TSH/FT3/FT4。
def test_week11_query_understanding_has_intent_entities_and_expansion():
    analysis = analyze_query("甲亢心慌怕热要查什么")

    assert "check" in analysis.intents
    assert any(entity.text == "心慌" for entity in analysis.entities)
    assert {"TSH", "FT3", "FT4"}.issubset(set(analysis.expanded_terms))
    assert "TSH" in analysis.expanded_query


# 知识点 2：RRF 按排名融合，不直接比较 BM25 分数和向量分数。
# 因为不同召回源的原始分数尺度不同，直接相加往往不稳定。
def test_week11_rrf_fuses_rankings_instead_of_raw_scores():
    fused = reciprocal_rank_fusion([
        RecallResult("D1", score=100.0, source="bm25", rank=1),
        RecallResult("D2", score=0.99, source="vector", rank=1),
        RecallResult("D1", score=1.0, source="entity", rank=2),
    ])

    # D1 被两个召回源命中，所以融合后应高于只被一个召回源命中的 D2。
    assert fused["D1"] > fused["D2"]


# 知识点 3：混合召回结果需要保留来源解释。
# 这能帮助学生看到：一个结果可能同时被 BM25、实体匹配和向量召回找到。
def test_week11_hybrid_recall_exposes_which_routes_found_the_doc():
    docs = [
        MedicalDocument("UTI", "尿路感染", "泌尿外科", ["尿频", "尿痛", "尿常规"], "尿频尿痛建议查尿常规。"),
        MedicalDocument("PNE", "肺炎", "呼吸内科", ["咳嗽", "发热"], "咳嗽发热建议查胸部影像。"),
    ]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 64}})
    pipeline = HybridRecallPipeline(docs, model)

    candidates, _ = pipeline.search("尿频尿痛应该挂什么科", top_k=1)

    assert candidates[0].doc.id == "UTI"
    assert {"bm25", "entity", "vector"} & set(candidates[0].sources)
