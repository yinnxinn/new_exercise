# Week 11: 召回流程与多模态概念

本周目标：把单一路径检索升级成真实搜索系统常见的多路召回流程。

## 召回流程

1. Query Understanding：清洗查询、识别意图、抽取医学实体、做简单查询扩展。
2. BM25 Recall：处理精确词、检查名、药品名、科室名。
3. Vector Recall：处理语义相似表达。
4. Entity Recall：用疾病、症状、检查、药品等结构化实体召回。
5. RRF Fusion：把不同召回源的排名融合成最终候选列表。

## 多模态概念

当前中文医学数据主要是文本、知识图谱和问答。多模态部分先以接口形式预留：

- `MediaAsset(media_type="image")`：医学影像或检查图片。
- `MediaAsset(media_type="pdf")`：指南、手册、论文 PDF。
- `MediaAsset(media_type="table")`：检验指标、药品表格。

后续真正接入多模态模型时，可以把图片 caption 或视觉 embedding 作为另一路召回源。

## 运行

```powershell
python -m medical_retrieval.app.hybrid_search --config configs\week11_hybrid.json --query "尿频尿痛还腰痛应该挂什么科"
```

观察输出中的：

- `意图`
- `实体`
- `扩展`
- `sources=bm25,entity,vector`

这几个字段对应真实检索系统里的查询理解和多路召回解释。
