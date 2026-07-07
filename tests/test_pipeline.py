from medical_retrieval.retrieval.pipeline import MedicalSearchEngine


def test_search_returns_relevant_heart_doc():
    engine = MedicalSearchEngine.from_config("configs/default.json")
    results, entities = engine.search("胸闷心慌应该挂什么科", top_k=3)

    assert entities
    assert results[0].doc.id == "D001"
    assert results[0].final_score > 0


def test_search_returns_relevant_urinary_doc():
    engine = MedicalSearchEngine.from_config("configs/default.json")
    results, entities = engine.search("尿频尿痛需要做尿常规吗", top_k=3)

    assert any(entity.text == "尿频" for entity in entities)
    assert results[0].doc.id == "D006"
