# Week 10: Embedding 与向量数据库

本周目标：把医学文档从关键词检索扩展到语义向量检索，并理解 FAISS、Milvus、sentence embeddings 的位置。

## 课程主线

1. 医学知识图谱字段转换成统一文档：`title / department / tags / content`。
2. 文档文本进入 embedding 模型，得到向量。
3. 向量写入向量库，查询也转成向量。
4. 用相似度召回候选医学文档。
5. 与 BM25、NER 召回在后续周融合。

## 当前代码产出

- `medical_retrieval.embeddings`：统一 embedding 接口。
- `HashEmbeddingModel`：无依赖 baseline，适合课堂快速运行。
- `SentenceTransformerEmbeddingModel`：可选真实句向量模型。
- `InMemoryVectorStore`：无依赖向量库，用 JSON 保存。
- `FaissVectorStore`：可选 FAISS 后端，适合本地高性能检索。

## 运行

默认无依赖版本：

```powershell
python -m medical_retrieval.app.build_vector_index --config configs\week10_vector.json
python -m medical_retrieval.app.search_vector --config configs\week10_vector.json --query "咳嗽发热需要做什么检查"
```

FAISS + sentence-transformers 版本需要安装：

```powershell
pip install faiss-cpu sentence-transformers numpy
python -m medical_retrieval.app.build_vector_index --config configs\week10_faiss.example.json
```

## FAISS 与 Milvus 的定位

FAISS 是本地向量索引库，适合课程实验、单机检索和性能演示。Milvus 是服务化向量数据库，适合多用户、持久化、权限、监控和分布式场景。本工程先用统一 `VectorStore` 接口承接两类后端，后续可以继续增加 `MilvusVectorStore`。
