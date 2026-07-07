from __future__ import annotations

import argparse
from pathlib import Path

from ..data.io import load_documents, load_json, load_lexicon
from ..embeddings import build_embedding_model
from ..nlp.ner import RuleMedicalNER, format_entities
from ..retrieval.recall import HybridRecallPipeline

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "week11_hybrid.json"


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Week11 hybrid recall demo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    data_path = resolve_project_path(config_path, config["data_path"])
    lexicon_path = config.get("lexicon_path")
    if lexicon_path:
        config["lexicon"] = load_lexicon(resolve_project_path(config_path, lexicon_path))

    documents = load_documents(data_path)
    ner = RuleMedicalNER(lexicon=config.get("lexicon"))
    embedding_model = build_embedding_model(config)
    pipeline = HybridRecallPipeline(documents, embedding_model, ner=ner)
    candidates, analysis = pipeline.search(args.query, top_k=args.top_k, recall_k=int(config.get("recall_k", 20)))

    print(f"查询：{analysis.raw_query}")
    print(f"意图：{','.join(analysis.intents)}")
    print(f"实体：{format_entities(analysis.entities)}")
    print(f"扩展：{','.join(analysis.expanded_terms) or '无'}")
    for rank, candidate in enumerate(candidates, 1):
        doc = candidate.doc
        print(f"{rank}. [{doc.id}] {doc.title} | {doc.department} | rrf={candidate.score:.4f} | sources={','.join(candidate.sources)}")
        print(f"   matched_entities={','.join(candidate.matched_entities) or '无'}")
        print(f"   tags={','.join(doc.tags[:8])}")


if __name__ == "__main__":
    main()
