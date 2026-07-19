import json

from medical_retrieval.data.convert_dialogue_dataset import convert
from medical_retrieval.data.io import load_documents
from medical_retrieval.data.tags import make_tag_code
from medical_retrieval.retrieval.milvus_schema import build_filter_expr


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_convert_dialogue_dataset_outputs_mysql_and_milvus_records(tmp_path):
    source_dir = tmp_path / "Chinese-medical-dialogue-data" / "IM_内科"
    source_dir.mkdir(parents=True)
    csv_path = source_dir / "heart.csv"
    csv_path.write_text(
        "department,title,question,answer\n"
        "心血管内科,高血压用药,高血压可以吃党参吗,高血压患者用药和补品需要咨询医生。\n"
        "心血管内科,高血压用药,高血压可以吃党参吗,高血压患者用药和补品需要咨询医生。\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    stats = convert(source_dir.parent, output_dir)

    assert stats["documents"] == 1
    assert stats["duplicate_rows"] == 1

    docs = load_documents(output_dir / "medical_dialogue_docs.jsonl")
    assert docs[0].question == "高血压可以吃党参吗"
    assert make_tag_code("department", "心血管内科") in docs[0].tag_codes
    assert make_tag_code("category", "IM_内科") in docs[0].tag_codes

    mysql_tags = read_jsonl(output_dir / "mysql_tags.jsonl")
    assert any(item["code"] == "dept:心血管内科" for item in mysql_tags)

    document_tags = read_jsonl(output_dir / "mysql_document_tags.jsonl")
    assert any(item["tag_code"] == "cat:im_内科" for item in document_tags)

    milvus_rows = read_jsonl(output_dir / "milvus_documents.jsonl")
    assert milvus_rows[0]["department_tag"] == "dept:心血管内科"
    assert "kw:高血压" in milvus_rows[0]["tag_codes"]


def test_build_milvus_filter_expr_uses_tag_array_filter():
    expr = build_filter_expr(
        department_tag="dept:心血管内科",
        required_tag_codes=["kw:高血压", "cat:im_内科"],
        embedding_model="bge-small-zh-v1.5",
    )

    assert 'department_tag == "dept:心血管内科"' in expr
    assert 'ARRAY_CONTAINS(tag_codes, "kw:高血压")' in expr
    assert 'embedding_model == "bge-small-zh-v1.5"' in expr
