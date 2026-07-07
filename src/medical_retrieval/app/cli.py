from __future__ import annotations

import argparse
from pathlib import Path

from ..nlp.ner import format_entities
from ..retrieval.pipeline import MedicalSearchEngine

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.json"

DEMO_QUERIES = [
    "胸闷心慌应该挂什么科",
    "咳嗽发热可能是肺炎吗",
    "饭后胃痛反酸怎么处理",
    "尿频尿痛还腰痛需要做什么检查",
    "关节晨僵是不是类风湿关节炎",
    "怕热出汗心慌要查甲状腺吗",
]


def print_results(query: str, engine: MedicalSearchEngine, top_k: int) -> None:
    results, entities = engine.search(query, top_k=top_k)
    print(f"\n查询：{query}")
    print(f"NER：{format_entities(entities)}")
    for rank, item in enumerate(results, 1):
        doc = item.doc
        print(
            f"{rank}. [{doc.id}] {doc.title} | {doc.department} | "
            f"final={item.final_score:.3f} vec={item.vector_score:.3f} "
            f"bm25={item.bm25_score:.3f} field={item.field_bonus:.2f} "
            f"ner={item.entity_bonus:.2f}"
        )
        print(f"   tags={','.join(doc.tags)}")
        print(f"   matched_ner={','.join(item.matched_entities) or '无'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Retrieval Plus")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--query", help="单条医学查询")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--demo", action="store_true", help="运行内置演示查询")
    args = parser.parse_args()

    engine = MedicalSearchEngine.from_config(args.config)
    if args.demo or not args.query:
        for query in DEMO_QUERIES:
            print_results(query, engine, args.top_k)
    else:
        print_results(args.query, engine, args.top_k)


if __name__ == "__main__":
    main()
