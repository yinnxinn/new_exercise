

"""第 10 周知识点测试：Embedding 与向量数据库。

这类测试不是只为了防回归，也可以作为课堂讲义使用。学生读测试时应能看到：
1. embedding 的输入是任意长度文本，输出是固定维度向量；
2. 向量通常会做归一化，便于用 cosine / inner product 计算相似度；
3. 向量数据库保存的不只是向量，还要保存向量和业务文档 id 的映射；
4. 查询文本也要经过同一个 embedding 模型，才能和文档向量做相似度检索。
"""

import math

from medical_retrieval.core.schema import MedicalDocument
from medical_retrieval.embeddings import build_embedding_model
from medical_retrieval.retrieval.vector_store import InMemoryVectorStore


# 知识点 1：embedding 是“文本 -> 固定长度数值向量”的过程。
# 这里用 hash backend 是为了让课堂环境无第三方依赖也能跑通；真实工程可切换到 sentence-transformers。
def test_week10_embedding_turns_text_into_fixed_length_normalized_vector():
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 32}})

    vector = model.encode("咳嗽发热需要检查血常规")
    norm = math.sqrt(sum(value * value for value in vector))

    # 断言 1：不管原始文本多长，输出维度由配置 dim 决定。
    assert len(vector) == 32
    # 断言 2：归一化后向量长度约等于 1，后续可直接用 cosine 相似度。
    assert abs(norm - 1.0) < 1e-6


# 知识点 2：向量库必须维护“向量 -> 文档”的映射，否则只能得到相似向量，无法返回业务结果。
def test_week10_vector_store_keeps_doc_id_mapping_with_vectors():
    docs = [
        MedicalDocument("PNE", "肺炎", "呼吸内科", ["咳嗽", "发热"], "咳嗽发热可完善胸部影像。"),
        MedicalDocument("DM", "糖尿病", "内分泌科", ["血糖"], "糖尿病需要监测血糖。"),
    ]
    model = build_embedding_model({"embedding": {"backend": "hash", "dim": 64}})
    store = InMemoryVectorStore()
    store.build(docs, model.encode_batch([doc.full_text for doc in docs]))

    matches = store.search(model.encode("咳嗽发热做什么检查"), top_k=1)

    # 查询和“肺炎”文档共享更多语义/词项，应该排在第一。
    assert matches[0].doc_id == "PNE"
    assert matches[0].rank == 1
    assert matches[0].score > 0
