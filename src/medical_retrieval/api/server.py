from __future__ import annotations

import json
from pathlib import Path

from ..data.io import load_documents, load_json, load_lexicon
from ..embeddings import build_embedding_model
from ..generation.answer import build_grounded_answer, stream_answer
from ..nlp.ner import RuleMedicalNER, format_entities
from ..retrieval.recall import HybridRecallPipeline

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "week12_app.json"
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def create_pipeline(config_path: str | Path = DEFAULT_CONFIG) -> HybridRecallPipeline:
    config_path = Path(config_path)
    config = load_json(config_path)
    data_path = resolve_project_path(config_path, config["data_path"])
    if config.get("lexicon_path"):
        config["lexicon"] = load_lexicon(resolve_project_path(config_path, config["lexicon_path"]))
    docs = load_documents(data_path)
    ner = RuleMedicalNER(lexicon=config.get("lexicon"))
    embedding_model = build_embedding_model(config)
    return HybridRecallPipeline(docs, embedding_model, ner=ner)


def create_app(config_path: str | Path = DEFAULT_CONFIG):
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FastAPI app requires `pip install -e .[api]`.") from exc

    app = FastAPI(title="Medical Retrieval Plus")
    pipeline = create_pipeline(config_path)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/search")
    def search(q: str = Query(...), top_k: int = 5):
        candidates, analysis = pipeline.search(q, top_k=top_k)
        return {
            "query": q,
            "intents": analysis.intents,
            "entities": format_entities(analysis.entities),
            "expanded_terms": analysis.expanded_terms,
            "results": [
                {
                    "id": item.doc.id,
                    "title": item.doc.title,
                    "department": item.doc.department,
                    "tags": item.doc.tags[:10],
                    "score": item.score,
                    "sources": item.sources,
                    "matched_entities": item.matched_entities,
                    "content": item.doc.content,
                }
                for item in candidates
            ],
        }

    @app.get("/answer")
    def answer(q: str = Query(...), top_k: int = 5):
        candidates, analysis = pipeline.search(q, top_k=top_k)
        return {
            "query": q,
            "intents": analysis.intents,
            "entities": format_entities(analysis.entities),
            "answer": build_grounded_answer(q, candidates),
            "sources": [{"id": item.doc.id, "title": item.doc.title, "department": item.doc.department} for item in candidates],
        }

    @app.get("/stream")
    def stream(q: str = Query(...), top_k: int = 5):
        candidates, _ = pipeline.search(q, top_k=top_k)
        answer_text = build_grounded_answer(q, candidates)

        def event_stream():
            for chunk in stream_answer(answer_text):
                payload = json.dumps({"text": chunk.text, "done": chunk.done}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
