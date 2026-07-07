# Medical Retrieval Plus

这是基于 `week8_medical_retrieval` 做的工程化扩展版。相比原来的单文件实现，这个版本把代码拆成了数据层、NLP 层、检索层和入口层，便于后续继续扩展 CBOW、FastAPI、实验评估和更多医学实体。

> 说明：本项目仅用于课程作业和技术演示，不替代医生诊断。

## 1. 项目目标

- 医学 NER：用词典和规则抽取症状、疾病、科室、检查和药物。
- BM25 检索：处理中文医学短语和英文缩写。
- 向量检索：当前提供无依赖的哈希向量 baseline，后续可替换成 CBOW 或 BERT。
- 融合排序：综合向量分数、关键词命中、字段命中和实体命中。
- 离线评估：输出 `Hit@K` 和 `MRR@K`，方便调参对比。

## 2. 工程目录

```text
medical_retrieval_plus/
├── configs/
│   └── default.json
├── data/
│   └── medical_docs.jsonl
├── docs/
│   ├── architecture-figure.md
│   └── architecture.svg
├── scripts/
│   └── run_demo.ps1
├── src/
│   └── medical_retrieval/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   └── evaluate.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── schema.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── io.py
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── ner.py
│       │   └── text.py
│       └── retrieval/
│           ├── __init__.py
│           ├── bm25.py
│           ├── pipeline.py
│           ├── ranker.py
│           └── vector.py
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

## 3. 运行方式

进入项目目录：

```powershell
cd D:\projects\class2026\medical_retrieval_plus
$env:PYTHONPATH = "src"
```

运行演示：

```powershell
python -m medical_retrieval.app.cli --demo
```

运行单条查询：

```powershell
python -m medical_retrieval.app.cli --query "胸闷心慌应该挂什么科" --top-k 5
```

运行评估：

```powershell
python -m medical_retrieval.app.evaluate
```

运行测试：

```powershell
pytest -q
```

## 4. 数据格式

`data/medical_docs.jsonl` 每行是一条 JSON 文档，必需字段如下：

- `id`
- `title`
- `department`
- `tags`
- `content`

示例：

```json
{"id":"D001","title":"反复胸闷心慌的常见原因","department":"心内科","tags":["胸闷","心慌"],"content":"..."}
```

## 5. 检索流程

1. `data/io.py` 读取 JSONL 文档和配置。
2. `nlp/text.py` 做中文切分、n-gram 和文本归一化。
3. `nlp/ner.py` 抽取医学实体。
4. `retrieval/bm25.py` 计算 BM25 分数。
5. `retrieval/vector.py` 计算向量相似度。
6. `retrieval/ranker.py` 计算字段加权和实体加权。
7. `retrieval/pipeline.py` 汇总得到最终排序。

融合公式：

```text
final_score = alpha * vector_score
            + (1 - alpha) * bm25_score
            + field_bonus
            + entity_bonus
```

## 6. 比原作业的扩展点

- 从单文件脚本拆成了分层工程结构。
- 保留并增强了混合检索主线。
- 入口和评估独立，方便后续接 Web API。
- 目录按职责拆分，后续加模型、实验、接口不会继续堆到一个文件里。
- 架构图已经单独放在 `docs/architecture-figure.md`。

## 7. 后续可以继续补的内容

- CBOW 向量替换哈希向量。
- 同义词归一化层。
- FastAPI 搜索接口。
- 更完整的实验报告和错误案例分析。
- 更大的人工标注测试集。
## 8. 接入中文医学知识图谱数据

当前检索系统最适合优先接入疾病知识图谱类数据，例如
`QASystemOnMedicalGraph` 的 `disease.csv`。它包含疾病、别名、症状、
科室、检查、药品、治疗等字段，可以直接转成检索文档，并同步生成 NER 词典。

下载 `disease.csv` 后运行：

```powershell
python -m medical_retrieval.data.convert_medical_graph --input path\to\disease.csv
```

使用知识图谱配置检索：

```powershell
python -m medical_retrieval.app.cli --config configs\medical_graph.json --query "咳嗽发热需要做什么检查"
```
## 9. 第 10 周：Embedding 与向量数据库

第 10 周在原有 BM25 + 规则 NER 的基础上增加语义向量检索层：

- `embeddings/`：统一 embedding 接口，包含无依赖 `HashEmbeddingModel` 和可选 `SentenceTransformerEmbeddingModel`。
- `retrieval/vector_store.py`：统一向量库接口，包含 `InMemoryVectorStore` 和可选 `FaissVectorStore`。
- `configs/week10_vector.json`：默认无依赖演示配置。
- `configs/week10_faiss.example.json`：FAISS + sentence-transformers 示例配置。
- `docs/weeks/week10_embedding_vector_db.md`：课程说明。

无依赖版本运行：

```powershell
python -m medical_retrieval.app.build_vector_index --config configs\week10_vector.json
python -m medical_retrieval.app.search_vector --config configs\week10_vector.json --query "咳嗽发热需要做什么检查"
```

安装可选向量依赖后，可以切换到 FAISS 和真实 sentence embedding：

```powershell
pip install -e .[vector]
python -m medical_retrieval.app.build_vector_index --config configs\week10_faiss.example.json
```
## 10. 第 11 周：多路召回与多模态概念

第 11 周把单一路径检索升级为多路召回：

- `query/understanding.py`：查询意图识别、实体抽取和简单查询扩展。
- `retrieval/recall.py`：BM25、向量、实体三路召回与 RRF 融合。
- `multimodal/assets.py`：为图片、PDF、表格等多模态资源预留统一接口。
- `configs/week11_hybrid.json`：多路召回演示配置。
- `docs/weeks/week11_hybrid_recall.md`：课程说明。

运行：

```powershell
python -m medical_retrieval.app.hybrid_search --config configs\week11_hybrid.json --query "尿频尿痛还腰痛应该挂什么科"
```
## 11. 第 12 周：Web 交互、RAG 与流式输出

第 12 周把检索 pipeline 包装成可交互系统：

- `generation/`：基于检索结果生成带来源和安全提示的回答。
- `api/server.py`：FastAPI 服务，提供 `/search`、`/answer`、`/stream`。
- `web/`：简单前端页面，支持检索和流式回答。
- `configs/week12_app.json`：Web/RAG 演示配置。
- `docs/weeks/week12_rag_web_streaming.md`：课程说明。

运行：

```powershell
pip install -e .[api]
$env:PYTHONPATH="src"
uvicorn medical_retrieval.api.server:app --reload --port 8000
```

打开 `http://127.0.0.1:8000`。

