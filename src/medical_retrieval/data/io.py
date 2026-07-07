from __future__ import annotations

import json
from pathlib import Path

from ..core.schema import MedicalDocument


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_documents(path: str | Path) -> list[MedicalDocument]:
    docs: list[MedicalDocument] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = {"id", "title", "department", "tags", "content"} - set(item)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            docs.append(
                MedicalDocument(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    department=str(item["department"]),
                    tags=[str(tag) for tag in item["tags"]],
                    content=str(item["content"]),
                )
            )
    return docs

def load_lexicon(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    lexicon: dict[str, list[str]] = {}
    for label, terms in data.items():
        if not isinstance(terms, list):
            raise ValueError(f"{path}:{label} must be a list")
        cleaned = sorted({str(term).strip() for term in terms if str(term).strip()}, key=len, reverse=True)
        lexicon[str(label)] = cleaned
    return lexicon



