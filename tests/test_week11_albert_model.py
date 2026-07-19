"""第 11 周实践：使用 ALBERT 生成句向量。

运行前安装可选依赖：

    pip install -e .[vector]

默认模型是 ``sentence-transformers/paraphrase-albert-small-v2``。测试只读取
本地 Hugging Face 缓存，避免在 pytest 过程中隐式下载大模型。可以使用
``WEEK11_ALBERT_MODEL`` 环境变量切换到其他已下载的 ALBERT 模型。
"""

from __future__ import annotations

import math
import os

import pytest

from medical_retrieval.embeddings import build_embedding_model


MODEL_NAME = os.getenv(
    "WEEK11_ALBERT_MODEL",
    "sentence-transformers/paraphrase-albert-small-v2",
)


@pytest.fixture(scope="module")
def albert_model():
    """从本地缓存加载 ALBERT；没有模型文件时给出可理解的跳过原因。"""

    pytest.importorskip(
        "sentence_transformers",
        reason="运行 ALBERT 示例需要安装 sentence-transformers",
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    try:
        try:
            model = build_embedding_model(
                {
                    "embedding": {
                        "backend": "sentence-transformer",
                        "model_name": MODEL_NAME,
                        "normalize": True,
                    }
                }
            )
        except OSError as exc:
            pytest.skip(
                f"本地尚未缓存 {MODEL_NAME}；请先下载模型后重新运行。原始错误：{exc}"
            )
        yield model
    finally:
        monkeypatch.undo()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def test_week11_albert_outputs_fixed_length_normalized_vectors(albert_model):
    vectors = albert_model.encode_batch(
        [
            "The patient has a fever and a cough.",
            "A person is suffering from cough and high temperature.",
        ]
    )

    assert albert_model.model_name == MODEL_NAME
    assert albert_model.dim > 0
    assert len(vectors) == 2
    assert all(len(vector) == albert_model.dim for vector in vectors)
    assert all(
        math.isclose(
            math.sqrt(sum(value * value for value in vector)),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        for vector in vectors
    )


def test_week11_albert_places_related_sentences_closer(albert_model):
    query, related, unrelated = albert_model.encode_batch(
        [
            "The patient has a fever and a cough.",
            "A person is suffering from cough and high temperature.",
            "The database stores product order records.",
        ]
    )

    related_score = cosine_similarity(query, related)
    unrelated_score = cosine_similarity(query, unrelated)

    assert related_score > unrelated_score
