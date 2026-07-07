# Week 12: 界面、用户交互与流式输出

本周目标：把医学检索 pipeline 包装成可交互的 RAG Demo。

## 系统能力

- `/search`：返回查询理解和检索结果。
- `/answer`：基于检索结果生成带来源的回答。
- `/stream`：用 SSE 格式流式输出回答。
- Web 页面：输入问题、展示检索来源、展示回答。

## 运行

安装 API 依赖：

```powershell
pip install -e .[api]
```

启动服务：

```powershell
$env:PYTHONPATH="src"
uvicorn medical_retrieval.api.server:app --reload --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 教学重点

1. RAG 不是让模型自由回答，而是基于检索结果回答。
2. 医疗回答必须展示来源和安全提示。
3. 流式输出本质是服务端分块返回，前端逐块渲染。
4. 用户交互界面应展示查询理解、召回来源和引用文档。
