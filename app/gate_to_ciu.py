"""
관련성 게이트(gate_pass.py) + CIU 채점(app/langgraph_judge.py)을 잇는 파이프라인.

흐름:
    1. concepts(dict)를 받음 (이미 준비된 상태로 들어온다고 가정 — DB 조회는 이 파일 책임 아님)
    2. 발화와 비교해서 관련성 게이트 판단 (gate_pass.py)
    3. 게이트 통과하면 → 기존 CIU 채점 파이프라인(langgraph_judge.py)으로 넘김
       게이트 통과 못하면 → CIU 채점 생략, 점수 0으로 채움
"""

from .gate_pass import judge_matches, score_matches
from .ciu_judge import score_transcript

ZERO_SCORE = {
    "raw_ciu": 0,
    "duplicate_ciu": 0,
    "net_ciu": 0,
    "total_words_excl_disfluency": 0,
    "total_words_raw": 0,
    "aq_term1_excl_disfluency": 0.0,
    "aq_term1_raw": 0.0,
}

REQUIRED_CONCEPT_KEYS = {"label", "category", "synonyms", "hypernyms"}
VALID_CATEGORIES = {"핵심", "부가"}

def validate_concepts(concepts: dict) -> None:
    """concepts json이 게이트/CIU 채점이 기대하는 스키마를 만족하는지 검사한다.
 
    스키마가 어긋나면(최상위 키 이름이 다르거나, category에 오타가 있거나,
    synonyms/hypernyms가 빠졌거나) 파이프라인 깊은 곳에서 알아채기 힘든
    KeyError나 '조용한 오동작'(예: category 오타 → 게이트가 항상 막힘)으로
    이어지는 대신, 여기서 바로 명확한 ValueError로 실패시킨다.
    DB에서 image_id로 가져온 concepts json을 이 함수에 넣기만 하면 검증된다.
    """
    if not isinstance(concepts, dict) or "concepts" not in concepts:
        raise ValueError("concepts json에 최상위 'concepts' 키가 없습니다.")
 
    items = concepts["concepts"]
    if not isinstance(items, list) or not items:
        raise ValueError("concepts['concepts']는 비어있지 않은 리스트여야 합니다.")
 
    labels = []
    for i, c in enumerate(items):
        if not isinstance(c, dict):
            raise ValueError(f"concepts['concepts'][{i}]가 dict가 아닙니다: {c!r}")
 
        missing = REQUIRED_CONCEPT_KEYS - c.keys()
        if missing:
            raise ValueError(
                f"concepts['concepts'][{i}] (label={c.get('label')!r})에 "
                f"필수 키가 없습니다: {sorted(missing)}"
            )
 
        if c["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"concepts['concepts'][{i}] (label={c['label']!r})의 category가 "
                f"'핵심'/'부가'가 아닙니다: {c['category']!r}"
            )
 
        if not isinstance(c["synonyms"], list) or not isinstance(c["hypernyms"], list):
            raise ValueError(
                f"concepts['concepts'][{i}] (label={c['label']!r})의 "
                "synonyms/hypernyms는 리스트여야 합니다."
            )
 
        labels.append(c["label"])
 
    if len(labels) != len(set(labels)):
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        raise ValueError(f"concepts['concepts']에 label 중복이 있습니다: {dupes}")
 
    if not any(c["category"] == "핵심" for c in items):
        raise ValueError(
            "concepts['concepts']에 category='핵심'인 요소가 하나도 없습니다 — "
            "이 상태로는 관련성 게이트를 절대 통과할 수 없습니다."
        )


def process_response(concepts: dict, transcript: str) -> dict:
    judgment = judge_matches(transcript, concepts)
    relevance_result = score_matches(judgment, concepts)

    if not relevance_result["is_relevant"]:
        return {
            "status": "irrelevant",
            "relevance": relevance_result,
            "ciu": ZERO_SCORE,
        }

    ciu_result = score_transcript(transcript, concepts)  # 같은 concepts 재사용

    return {
        "status": "scored",
        "relevance": relevance_result,
        "ciu": ciu_result["summary"],
    }