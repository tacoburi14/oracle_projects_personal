"""
관련성 게이트(gate_pass.py) + CIU 채점(app/langgraph_judge.py)을 잇는 파이프라인.

흐름:
    1. concepts(dict)를 받음 (이미 준비된 상태로 들어온다고 가정 — DB 조회는 이 파일 책임 아님)
    2. 발화와 비교해서 관련성 게이트 판단 (gate_pass.py)
    3. 게이트 통과하면 → 기존 CIU 채점 파이프라인(langgraph_judge.py)으로 넘김
       게이트 통과 못하면 → CIU 채점 생략, 점수 0으로 채움
"""

from app.gate_pass import judge_matches, score_matches
from app.ciu_judge import score_transcript

ZERO_SCORE = {
    "raw_ciu": 0,
    "duplicate_ciu": 0,
    "net_ciu": 0,
    "total_words_excl_disfluency": 0,
    "total_words_raw": 0,
    "aq_term1_excl_disfluency": 0.0,
    "aq_term1_raw": 0.0,
}


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