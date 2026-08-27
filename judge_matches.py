"""
2단계 — 발화(STT 텍스트)가 1단계에서 뽑은 요소들과 얼마나 일치하는지 판단.
실제 사용자가 응답할 때마다 실행되는 부분. 텍스트 전용 LLM만 씀 (이미지 다시 안 봄).

핵심: 이 단계에서 모델이 내놓을 수 있는 답을 Literal로 딱 세 가지로만
강제한다 — "동의어_일치" | "상위개념_일치" | "불일치". 지난번 겪었던
"role 필드에 아무 문자열이나 나오던" 문제를 여기서 원천 차단하는 것.

동의어_일치/상위개념_일치/불일치라는 "분류"는 LLM이 판단하고,
그 분류를 실제 몇 점으로 환산할지(SYNONYM_SCORE 등)는 코드가 결정론적으로
계산한다 (LLM은 산수 안 함 — scoring.py 때부터 지켜온 원칙 그대로).

사전 준비:
    ollama pull qwen3:1.7b

실행:
    python3 judge_matches.py
"""

import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# match_type별 점수. 실제 테스트해보면서 조정하세요.
SCORE_TABLE = {
    "동의어_일치": 1.0,
    "상위개념_일치": 0.5,
    "불일치": 0.0,
}


class ConceptMatch(BaseModel):
    """요소 하나에 대해, 발화가 이걸 얼마나 정확히 언급했는지."""

    label: str = Field(description="어떤 요소에 대한 판단인지 (입력받은 label을 그대로 반환)")
    match_type: Literal["동의어_일치", "상위개념_일치", "불일치"] = Field(
        description=(
            "발화가 이 요소의 synonyms 중 하나를 정확히 언급했으면 '동의어_일치', "
            "synonyms는 없지만 hypernyms 중 하나를 언급했으면 '상위개념_일치', "
            "둘 다 언급 안 했으면 '불일치'"
        )
    )


class MatchJudgment(BaseModel):
    matches: List[ConceptMatch]


def load_concepts(concepts_path: str) -> dict:
    with open(concepts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_matches(transcript: str, concepts: dict) -> MatchJudgment:
    
    load_dotenv()
    
    api_key = os.environ.get("OLLAMA_API_KEY")
    llm = ChatOllama(
        model="gemma4:cloud",
         base_url="https://ollama.com",
        temperature=0,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {},
    )
    structured_llm = llm.with_structured_output(MatchJudgment)

    concepts_text = "\n".join(
        f"- {c['label']}: 동의어={c['synonyms']}, 상위개념={c['hypernyms']}"
        for c in concepts["concepts"]
    )

    prompt = (
        f'발화: "{transcript}"\n\n'
        "아래는 그림에서 뽑은 요소 목록이야. 각 요소마다, 위 발화가 그 요소의 "
        "동의어를 정확히 언급했는지, 상위개념만 언급했는지, 아예 언급 안 했는지 "
        "판단해줘.\n\n"
        f"{concepts_text}\n\n"
        "각 요소의 label을 그대로 써서, match_type을 '동의어_일치' / "
        "'상위개념_일치' / '불일치' 중 하나로만 답해."
    )

    return structured_llm.invoke(prompt)


def score_matches(judgment: MatchJudgment) -> dict:
    """match_type(LLM 판단)을 실제 점수(코드 계산)로 환산."""
    results = [
        {"label": m.label, "match_type": m.match_type, "score": SCORE_TABLE[m.match_type]}
        for m in judgment.matches
    ]
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"matches": results, "overall_relevance": overall}


if __name__ == "__main__":
    concepts = load_concepts("walking_tags.json")   # extract_tags.py의 OUTPUT_PATH와 동일하게 맞춤

    # 테스트하고 싶은 발화로 바꿔보세요 (STT로 변환됐다고 가정)
    transcript = "어...남자...남자가....음...뛰...뛰어가고...있는...것..같아.."

    judgment = judge_matches(transcript, concepts)
    result = score_matches(judgment)

    for m in result["matches"]:
        print(f"- {m['label']}: {m['match_type']} → {m['score']}점")
    print(f"\n전체 관련도 점수: {result['overall_relevance']:.2f}")