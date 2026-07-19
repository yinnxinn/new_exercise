"""第 11 周实践：Milvus Collection 的创建、写入、过滤和向量检索。

启动项目自带的 Milvus 后运行：

    docker compose up -d etcd minio milvus
    pytest -q tests/test_week11_milvus_usage.py -s

默认连接 ``127.0.0.1:19530``，也可以通过 ``MILVUS_HOST`` 和
``MILVUS_PORT`` 环境变量修改。测试使用唯一的临时 Collection，并在测试
结束后自动删除，不会修改项目正式 Collection。
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from medical_retrieval.retrieval.milvus_schema import (
    build_filter_expr,
    create_collection,
)


def connect_to_milvus():
    """连接课堂 Milvus；服务没有启动时跳过集成测试。"""

    pymilvus = pytest.importorskip(
        "pymilvus",
        reason="运行 Milvus 示例需要安装 pymilvus",
    )
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")

    try:
        pymilvus.connections.connect(
            alias="default",
            host=host,
            port=port,
            timeout=3,
        )
        pymilvus.utility.list_collections(timeout=3)
    except Exception as exc:
        pytest.skip(f"Milvus 服务不可用（{host}:{port}）：{exc}")

    return pymilvus


def test_week11_milvus_insert_filter_and_vector_search():
    pymilvus = connect_to_milvus()
    collection_name = f"week11_usage_{uuid4().hex[:12]}"
    collection = None

    try:
        # 为了突出 Milvus 的使用流程，这里使用容易观察的 4 维教学向量。
        collection = create_collection(
            name=collection_name,
            embedding_dim=4,
            drop_existing=True,
        )
        collection.insert(
            [
                {
                    "document_id": 101,
                    "document_external_id": "W11-PNE",
                    "dialogue_id": 1001,
                    "department": "呼吸内科",
                    "department_tag": "dept:呼吸内科",
                    "source_category": "内科",
                    "category_tag": "category:内科",
                    "tag_codes": ["symptom:咳嗽", "symptom:发热"],
                    "content_hash": "pneumonia",
                    "embedding_model": "week11-demo-4d",
                    "is_active": True,
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "document_id": 102,
                    "document_external_id": "W11-DM",
                    "dialogue_id": 1002,
                    "department": "内分泌科",
                    "department_tag": "dept:内分泌科",
                    "source_category": "内科",
                    "category_tag": "category:内科",
                    "tag_codes": ["disease:糖尿病", "check:血糖"],
                    "content_hash": "diabetes",
                    "embedding_model": "week11-demo-4d",
                    "is_active": True,
                    "embedding": [0.0, 1.0, 0.0, 0.0],
                },
            ]
        )
        collection.flush()
        collection.load()

        filter_expr = build_filter_expr(
            department_tag="dept:呼吸内科",
            required_tag_codes=["symptom:咳嗽"],
            embedding_model="week11-demo-4d",
        )
        results = collection.search(
            data=[[0.95, 0.05, 0.0, 0.0]],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 32}},
            limit=2,
            expr=filter_expr,
            output_fields=[
                "document_id",
                "document_external_id",
                "department",
                "tag_codes",
            ],
        )

        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].entity.get("document_id") == 101
        assert results[0][0].entity.get("department") == "呼吸内科"
        assert "symptom:咳嗽" in results[0][0].entity.get("tag_codes")
        assert results[0][0].score > 0.9
    finally:
        if collection is not None:
            collection.release()
        if pymilvus.connections.has_connection("default"):
            if pymilvus.utility.has_collection(collection_name):
                pymilvus.utility.drop_collection(collection_name)
            pymilvus.connections.disconnect("default")
