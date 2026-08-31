"""
발화(STT 텍스트)가 이미지에서 뽑은 요소들과 얼마나 일치하는지 판단.


핵심(주체/행동) 요소 중 하나라도 맞으면 관련성 게이트 통과(is_relevant=True).
부가(옷/배경 등) 요소는 게이트 판단엔 영향 안 주고, 참고용으로만 같이 반환함.
"""

import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

SCORE_TABLE = {
    "동의어_일치": 1.0,
    "상위개념_일치": 0.5,
    "불일치": 0.0,
}


class ConceptMatch(BaseModel):
    label: str = Field(description="판단 대상 요소의 label (입력받은 label을 그대로 반환)")
    match_type: Literal["동의어_일치", "상위개념_일치", "불일치"] = Field(
        description=(
            "발화가 이 요소의 synonyms 중 하나를 정확히 언급했으면 '동의어_일치', "
            "synonyms는 없지만 hypernyms 중 하나를 언급했으면 '상위개념_일치', "
            "둘 다 언급 안 했으면 '불일치'"
        )
    )


class MatchJudgment(BaseModel):
    matches: List[ConceptMatch]


def judge_matches(transcript: str, concepts: dict) -> MatchJudgment:
    load_dotenv()

    api_key = os.environ.get("OLLAMA_API_KEY")
    llm = ChatOllama(
        model="gemma4:cloud",
        base_url="https://ollama.com",
        temperature=0,
        format=MatchJudgment.model_json_schema(),
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {},
    )
    structured_llm = llm.with_structured_output(MatchJudgment)

    concepts_text = "\n".join(
        f"- {c['label']}: 동의어={c['synonyms']}, 상위개념={c['hypernyms']}"
        for c in concepts["concepts"]
    )

    example_output = {
        "matches": [
            {"label": "남자", "match_type": "동의어_일치"},
            {"label": "뛰다", "match_type": "상위개념_일치"},
        ]
    }
    example_json_text = json.dumps(example_output, ensure_ascii=False, indent=2)

    prompt = (
        f'발화: "{transcript}"\n\n'
        "아래는 그림에서 뽑은 요소 목록이야. 각 요소마다, 위 발화가 그 요소의 "
        "동의어를 정확히 언급했는지, 상위개념만 언급했는지, 아예 언급 안 했는지 "
        "판단해줘.\n\n"
        f"{concepts_text}\n\n"
        "각 요소의 label을 그대로 써서, match_type을 '동의어_일치' / "
        "'상위개념_일치' / '불일치' 중 하나로만 답해.\n\n"
        "다른 설명이나 마크다운 없이, 반드시 순수 JSON 형식으로만 답해.\n"
        "JSON 형식 예시는 다음과 같다:\n"
        f"{example_json_text}"
    )

    return structured_llm.invoke(prompt)


def score_matches(judgment: MatchJudgment, concepts: dict) -> dict:
    """match_type(LLM 판단)을 점수로 환산하고, 핵심/부가로 나눠서 게이트를 판단."""
    category_by_label = {c["label"]: c["category"] for c in concepts["concepts"]}

    results = [
        {
            "label": m.label,
            "category": category_by_label.get(m.label, "부가"),
            "match_type": m.match_type,
            "score": SCORE_TABLE[m.match_type],
        }
        for m in judgment.matches
    ]

    core_results = [r for r in results if r["category"] == "핵심"]
    detail_results = [r for r in results if r["category"] == "부가"]

    # 핵심 요소 중 하나라도 (동의어_일치/상위개념_일치) 맞으면 관련성 게이트 통과
    is_relevant = any(r["match_type"] != "불일치" for r in core_results)

    return {
        "is_relevant": is_relevant,
        "core_matches": core_results,
        "detail_matches": detail_results,
    }