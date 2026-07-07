"""Week 10 knowledge tests: embeddings and vector databases.

These tests are intentionally educational: each assertion maps to a course
concept students should understand before moving from BM25 to vector search.
"""

import math

from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.embeddings import build_embedding_model
from medical_retrieval.retrieval.vector_store import InMemoryVectorStore


def test_week10_embedding_turns_text_into_fixed_length_normalized_vector():
    """Knowledge point: an embedding model maps variable-length text to a fixed vector."""
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 32}})

    vector = model.encode("咳嗽发热需要检查血常规")
    norm = math.sqrt(sum(value * value for value in vector))

    assert len(vector) == 32
    assert abs(norm - 1.0) < 1e-6


def test_week10_vector_store_keeps_doc_id_mapping_with_vectors():
    """Knowledge point: a vector database stores vectors plus document ids/metadata."""
    docs = [
        MedicalDocument("PNE", "肺炎", "呼吸内科", ["咳嗽", "发热"], "咳嗽发热可完善胸部影像。"),
        MedicalDocument("DM", "糖尿病", "内分泌科", ["血糖"], "糖尿病需要监测血糖。"),
    ]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 64}})
    store = InMemoryVectorStore()
    store.build(docs, model.encode_batch([doc.full_text for doc in docs]))

    matches = store.search(model.encode("咳嗽发热做什么检查"), top_k=1)

    assert matches[0].doc_id == "PNE"
    assert matches[0].rank == 1
    assert matches[0].score > 0
