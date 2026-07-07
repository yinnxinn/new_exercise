"""Week 11 knowledge tests: query understanding and hybrid recall."""

from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.embeddings import build_embedding_model
from medical_retrieval.query.understanding import analyze_query
from medical_retrieval.retrieval.recall import HybridRecallPipeline, RecallResult, reciprocal_rank_fusion


def test_week11_query_understanding_has_intent_entities_and_expansion():
    """Knowledge point: recall starts with understanding the user query."""
    analysis = analyze_query("甲亢心慌怕热要查什么")

    assert "check" in analysis.intents
    assert any(entity.text == "心慌" for entity in analysis.entities)
    assert {"TSH", "FT3", "FT4"}.issubset(set(analysis.expanded_terms))
    assert "TSH" in analysis.expanded_query


def test_week11_rrf_fuses_rankings_instead_of_raw_scores():
    """Knowledge point: RRF combines different recall sources by rank."""
    fused = reciprocal_rank_fusion([
        RecallResult("D1", score=100.0, source="bm25", rank=1),
        RecallResult("D2", score=0.99, source="vector", rank=1),
        RecallResult("D1", score=1.0, source="entity", rank=2),
    ])

    assert fused["D1"] > fused["D2"]


def test_week11_hybrid_recall_exposes_which_routes_found_the_doc():
    """Knowledge point: search results should explain their recall sources."""
    docs = [
        MedicalDocument("UTI", "尿路感染", "泌尿外科", ["尿频", "尿痛", "尿常规"], "尿频尿痛建议查尿常规。"),
        MedicalDocument("PNE", "肺炎", "呼吸内科", ["咳嗽", "发热"], "咳嗽发热建议查胸部影像。"),
    ]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 64}})
    pipeline = HybridRecallPipeline(docs, model)

    candidates, _ = pipeline.search("尿频尿痛应该挂什么科", top_k=1)

    assert candidates[0].doc.id == "UTI"
    assert {"bm25", "entity", "vector"} & set(candidates[0].sources)
