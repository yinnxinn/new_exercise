from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELD_ALIASES = {
    "name": ["name", "disease"],
    "alias": ["alias"],
    "department": ["department", "cure_department"],
    "symptom": ["symptom"],
    "check": ["check", "checklist", "need_check"],
    "drug": ["drug", "common_drug", "recommand_drug", "recommend_drug"],
    "cause": ["cause"],
    "desc": ["desc", "description"],
    "prevent": ["prevent"],
    "treatment": ["treatment", "cure_way"],
    "complication": ["complication", "acompany", "acompany_with"],
}

LEXICON_FIELDS = {
    "DISEASE": ["name", "alias", "complication"],
    "SYMPTOM": ["symptom"],
    "DEPARTMENT": ["department"],
    "CHECK": ["check"],
    "DRUG": ["drug"],
}


def split_terms(value: str) -> list[str]:
    value = value.replace("[详细]", " ")
    for mark in [",", "，", "、", ";", "；", "|", "\n", "\t"]:
        value = value.replace(mark, " ")
    return [part.strip() for part in value.split() if part.strip()]


def pick(row: dict[str, str], canonical: str) -> str:
    for key in FIELD_ALIASES[canonical]:
        if key in row and row[key].strip():
            return row[key].strip()
    return ""


def convert(input_csv: Path, docs_path: Path, lexicon_path: Path) -> tuple[int, dict[str, int]]:
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon: dict[str, set[str]] = {label: set() for label in LEXICON_FIELDS}
    count = 0

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f, open(
        docs_path, "w", encoding="utf-8", newline="\n"
    ) as out:
        reader = csv.DictReader(f)
        for row in reader:
            name = pick(row, "name")
            if not name:
                continue

            values = {field: pick(row, field) for field in FIELD_ALIASES}
            tags: list[str] = []
            for field in ["alias", "symptom", "check", "drug", "complication"]:
                tags.extend(split_terms(values[field]))
            tags = sorted(set(tags))

            content_parts = [
                ("疾病简介", values["desc"]),
                ("病因", values["cause"]),
                ("预防", values["prevent"]),
                ("症状", values["symptom"]),
                ("检查", values["check"]),
                ("治疗", values["treatment"]),
                ("常用药", values["drug"]),
                ("并发症", values["complication"]),
            ]
            content = "；".join(f"{label}：{text}" for label, text in content_parts if text)
            department = "、".join(split_terms(values["department"]))
            doc = {
                "id": f"KG{count + 1:06d}",
                "title": name,
                "department": department,
                "tags": tags,
                "content": content,
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

            for label, fields in LEXICON_FIELDS.items():
                for field in fields:
                    lexicon[label].update(split_terms(values[field]))

    serializable = {label: sorted(terms, key=len, reverse=True) for label, terms in lexicon.items()}
    with open(lexicon_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return count, {label: len(terms) for label, terms in serializable.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert medical graph disease.csv to Medical Retrieval Plus JSONL")
    parser.add_argument("--input", required=True, help="QASystemOnMedicalGraph disease.csv path")
    parser.add_argument("--docs-output", default="data/medical_graph_docs.jsonl")
    parser.add_argument("--lexicon-output", default="data/medical_graph_lexicon.json")
    args = parser.parse_args()

    count, lexicon_counts = convert(Path(args.input), Path(args.docs_output), Path(args.lexicon_output))
    print(f"wrote {count} documents to {args.docs_output}")
    print(f"wrote lexicon to {args.lexicon_output}: {lexicon_counts}")


if __name__ == "__main__":
    main()
