from medical_retrieval.data.convert_dialogue_dataset import convert
from medical_retrieval.data.mysql_loader import build_milvus_payload, load_import_plan, summarize_plan


def test_mysql_loader_plan_and_milvus_payload_use_database_ids(tmp_path):
    source_dir = tmp_path / "dataset" / "IM_内科"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.csv").write_text(
        "department,title,question,answer\n"
        "心血管内科,高血压,高血压头晕怎么办,建议监测血压并咨询医生。\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "dialogue"
    convert(source_dir.parent, output_dir)

    plan = load_import_plan(output_dir)
    summary = summarize_plan(plan)

    assert summary["dialogues"] == 1
    assert summary["documents"] == 1
    assert summary["tags"] >= 3
    assert summary["milvus"] == 1

    payload = build_milvus_payload(
        plan["milvus"],
        document_ids={"CMD00000001": 101},
        dialogue_ids={"CMD00000001": 201},
    )

    assert payload[0]["document_id"] == 101
    assert payload[0]["dialogue_id"] == 201
    assert payload[0]["department_tag"] == "dept:心血管内科"
