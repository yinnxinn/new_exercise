import json

from medical_retrieval.data.convert_medical_graph import convert
from medical_retrieval.retrieval.pipeline import MedicalSearchEngine


def test_convert_medical_graph_builds_docs_and_lexicon(tmp_path):
    source = tmp_path / "disease.csv"
    source.write_text(
        "name,alias,department,symptom,check,drug,desc,cause,prevent,treatment,complication\n"
        "肺炎,肺部感染,呼吸内科,咳嗽 发热,血常规 胸部影像,抗感染治疗,肺部炎症,感染,注意休息,药物治疗,胸腔积液\n",
        encoding="utf-8",
    )
    docs_path = tmp_path / "docs.jsonl"
    lexicon_path = tmp_path / "lexicon.json"

    count, lexicon_counts = convert(source, docs_path, lexicon_path)

    assert count == 1
    assert lexicon_counts["DISEASE"] >= 2
    doc = json.loads(docs_path.read_text(encoding="utf-8").splitlines()[0])
    assert doc["title"] == "肺炎"
    assert doc["department"] == "呼吸内科"
    assert "咳嗽" in doc["tags"]

    config = {
        "data_path": str(docs_path),
        "lexicon_path": str(lexicon_path),
        "ranking": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    engine = MedicalSearchEngine.from_config(config_path)
    results, entities = engine.search("咳嗽发热是不是肺炎", top_k=1)

    assert results[0].doc.title == "肺炎"
    assert {entity.text for entity in entities} >= {"咳嗽", "发热", "肺炎"}
