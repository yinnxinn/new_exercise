from __future__ import annotations

import json
import re
from typing import Optional

from ..core.schema import Entity

ENTITY_LEXICON: dict[str, list[str]] = {
    "SYMPTOM": [
        "胸闷", "心慌", "胸痛", "出汗", "呼吸困难", "咳嗽", "发热", "咳痰", "胃痛",
        "反酸", "烧心", "头痛", "头晕", "尿频", "尿急", "尿痛", "腰痛", "腿麻",
        "关节痛", "晨僵", "皮疹", "瘙痒", "乏力", "失眠", "怕热", "怕冷",
    ],
    "DISEASE": [
        "心律失常", "冠心病", "糖尿病", "高血压", "肺炎", "支气管炎", "胃炎",
        "胃食管反流", "偏头痛", "脑血管疾病", "尿路感染", "类风湿关节炎", "湿疹",
        "荨麻疹", "甲亢", "甲减", "腰椎间盘突出", "坐骨神经痛",
    ],
    "DEPARTMENT": [
        "心内科", "内分泌科", "呼吸内科", "消化内科", "神经内科", "儿科",
        "风湿免疫科", "皮肤科", "泌尿外科", "骨科", "急诊",
    ],
    "CHECK": [
        "心电图", "心肌酶", "动态心电图", "血常规", "胸部影像", "胃镜", "尿常规",
        "尿培养", "TSH", "FT3", "FT4", "抗CCP抗体", "类风湿因子", "血沉", "C反应蛋白",
    ],
    "DRUG": ["降压药", "胰岛素", "退烧药", "抗感染治疗"],
}

ENTITY_WEIGHTS = {
    "SYMPTOM": 0.12,
    "DISEASE": 0.16,
    "DEPARTMENT": 0.18,
    "CHECK": 0.10,
    "DRUG": 0.08,
}


class _RuleMedicalNER:
    def __init__(self, lexicon: dict[str, list[str]] | None = None):
        self.lexicon = lexicon or ENTITY_LEXICON

    def extract(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        lowered = text.lower()
        for label, terms in self.lexicon.items():
            for term in terms:
                start = lowered.find(term.lower())
                while start >= 0:
                    found.append(Entity(term, label, start, start + len(term)))
                    start = lowered.find(term.lower(), start + 1)

        found.sort(key=lambda item: (item.start, -(item.end - item.start)))
        kept: list[Entity] = []
        occupied: set[int] = set()
        for item in found:
            span = set(range(item.start, item.end))
            if span & occupied:
                continue
            kept.append(item)
            occupied.update(span)
        return kept


try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


class LLMMedicalNER:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-v4-pro",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rule_ner = _RuleMedicalNER()
        self.client = None
        if OpenAI is not None and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.entity_types = {
            "sym": "临床表现/症状",
            "dis": "疾病/诊断",
            "dru": "药物/治疗方案",
            "equ": "医疗设备",
            "pro": "医疗程序/操作",
            "bod": "身体部位/器官",
            "ite": "医学检验项目/检查",
            "mic": "微生物类",
            "dep": "科室",
        }
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        entity_desc = "\n".join([f"  - {code}: {desc}" for code, desc in self.entity_types.items()])
        return f"""你是一个专业的医学文本信息抽取助手。请从中文医学文本中抽取实体。

支持的实体类型：
{entity_desc}

输出要求：
- 仅返回 JSON 数组
- 每个实体包含 entity, start_idx, end_idx, type
- 如果没有实体，返回 []
"""

    def _build_user_prompt(self, text: str) -> str:
        return f"原文：{text}\n\n请抽取原文中的所有医学实体。"

    def _parse_response(self, response: str) -> list[dict]:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                response = match.group()
            data = json.loads(response)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        if not entities:
            return []
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        kept: list[Entity] = []
        occupied: set[int] = set()
        for item in entities:
            span = set(range(item.start, item.end))
            if span & occupied:
                continue
            kept.append(item)
            occupied.update(span)
        return kept

    def extract(self, text: str) -> list[Entity]:
        if not text or not text.strip():
            return []
        if self.client is None:
            return self.rule_ner.extract(text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._build_user_prompt(text)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_entities = self._parse_response(response.choices[0].message.content or "[]")
        except Exception:
            return self.rule_ner.extract(text)

        entities: list[Entity] = []
        for item in raw_entities:
            try:
                entity_text = item.get("entity", "")
                entity_type = item.get("type", "").upper()
                start_idx = int(item.get("start_idx", 0))
                end_idx = int(item.get("end_idx", len(entity_text)))
                if entity_text and text[start_idx:end_idx] == entity_text:
                    entities.append(Entity(text=entity_text, type=entity_type, start=start_idx, end=end_idx))
            except Exception:
                continue
        return self._deduplicate_entities(entities) or self.rule_ner.extract(text)


class RuleMedicalNER:
    def __init__(self, use_llm: bool = False, **kwargs):
        self.use_llm = use_llm
        self.rule_ner = _RuleMedicalNER(kwargs.get("lexicon"))
        self.llm_ner = LLMMedicalNER(**kwargs) if use_llm else None

    def extract(self, text: str) -> list[Entity]:
        if self.use_llm and self.llm_ner is not None:
            return self.llm_ner.extract(text)
        return self.rule_ner.extract(text)


def format_entities(entities: list[Entity]) -> str:
    if not entities:
        return "无"
    return ", ".join(f"{entity.text}/{entity.type}" for entity in entities)
