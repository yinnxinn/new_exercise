from __future__ import annotations

import re

STOPWORDS = {
    "的", "了", "和", "与", "或", "是", "有", "在", "我", "想", "请问",
    "应该", "怎么", "如何", "什么", "可能", "需要", "可以", "是否",
}


def normalize_text(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    tokens: list[str] = []

    for part in parts:
        if re.fullmatch(r"[a-z0-9]+", part):
            if part not in STOPWORDS:
                tokens.append(part)
            continue

        chars = [ch for ch in part if ch not in STOPWORDS]
        tokens.extend(chars)
        for i in range(len(part) - 1):
            bigram = part[i : i + 2]
            if bigram not in STOPWORDS:
                tokens.append(bigram)
        for i in range(len(part) - 2):
            trigram = part[i : i + 3]
            if trigram not in STOPWORDS:
                tokens.append(trigram)

    return tokens
