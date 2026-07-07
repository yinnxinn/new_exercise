from __future__ import annotations

import argparse
from pathlib import Path

from ..retrieval.pipeline import MedicalSearchEngine

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.json"

CASES = [
    ("胸闷心慌应该挂什么科", "D001"),
    ("咳嗽发热怀疑肺炎", "D004"),
    ("胃痛反酸需要胃镜吗", "D005"),
    ("尿频尿痛做尿常规", "D006"),
    ("关节晨僵查抗CCP抗体", "D007"),
    ("怕热出汗心慌查TSH", "D010"),
]


def reciprocal_rank(ids: list[str], expected: str) -> float:
    if expected not in ids:
        return 0.0
    return 1.0 / (ids.index(expected) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate demo retrieval cases")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    engine = MedicalSearchEngine.from_config(args.config)
    hits = 0
    mrr = 0.0
    for query, expected in CASES:
        results, _ = engine.search(query, top_k=args.top_k)
        ids = [item.doc.id for item in results]
        hit = expected in ids
        hits += int(hit)
        mrr += reciprocal_rank(ids, expected)
        print(f"{'PASS' if hit else 'FAIL'} query={query} expected={expected} got={ids}")

    total = len(CASES)
    print(f"\nHit@{args.top_k}: {hits / total:.3f}")
    print(f"MRR@{args.top_k}: {mrr / total:.3f}")


if __name__ == "__main__":
    main()
