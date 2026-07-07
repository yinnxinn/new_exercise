from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.embeddings import build_embedding_model
from medical_retrieval.retrieval.vector_store import InMemoryVectorStore


def test_hash_embedding_and_memory_vector_store_rank_relevant_doc():
    docs = [
        MedicalDocument("D1", "肺炎检查", "呼吸内科", ["咳嗽", "发热", "胸部影像"], "咳嗽发热需要检查血常规和胸部影像。"),
        MedicalDocument("D2", "糖尿病控糖", "内分泌科", ["血糖", "胰岛素"], "糖尿病需要监测空腹血糖和糖化血红蛋白。"),
    ]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 128}})
    vectors = model.encode_batch([doc.full_text for doc in docs])
    store = InMemoryVectorStore()
    store.build(docs, vectors)

    matches = store.search(model.encode("咳嗽发热做什么检查"), top_k=2)

    assert matches[0].doc_id == "D1"
    assert matches[0].score > matches[1].score


def test_memory_vector_store_round_trip(tmp_path):
    docs = [MedicalDocument("D1", "甲亢", "内分泌科", ["心慌"], "怕热多汗心慌需要查甲状腺功能。")]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 64}})
    store = InMemoryVectorStore()
    store.build(docs, model.encode_batch([docs[0].full_text]))
    path = tmp_path / "vectors.json"

    store.save(path)
    loaded = InMemoryVectorStore.load(path)
    matches = loaded.search(model.encode("心慌怕热"), top_k=1)

    assert matches[0].doc_id == "D1"
