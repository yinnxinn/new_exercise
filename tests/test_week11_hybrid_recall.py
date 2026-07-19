
from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.embeddings import build_embedding_model
from medical_retrieval.query.understanding import analyze_query
from medical_retrieval.retrieval.recall import HybridRecallPipeline


def test_query_understanding_detects_intent_and_expansion():
    analysis = analyze_query("甲亢心慌怕热要查什么")

    assert "check" in analysis.intents
    assert "TSH" in analysis.expanded_terms
    assert "FT3" in analysis.expanded_terms


def test_hybrid_recall_returns_entity_relevant_doc():
    docs = [
        MedicalDocument("D1", "尿路感染", "泌尿外科", ["尿频", "尿痛", "尿常规"], "尿频尿痛需要检查尿常规。"),
        MedicalDocument("D2", "肺炎", "呼吸内科", ["咳嗽", "发热"], "咳嗽发热需要检查胸部影像。"),
    ]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 128}})
    pipeline = HybridRecallPipeline(docs, model)

    candidates, analysis = pipeline.search("尿频尿痛应该挂什么科", top_k=2)

    assert candidates[0].doc.id == "D1"
    assert "department" in analysis.intents
    assert "entity" in candidates[0].sources
