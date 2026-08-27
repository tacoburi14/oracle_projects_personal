"""
관련성 점수 계산 스크립트 — 실제 /score 요청마다 도는 부분.

extract_tags.py가 미리 만들어둔 JSON 태그 파일과 사용자 발화(전사문)를
텍스트로만 비교한다. 이미지도, LLM 호출도 필요 없어서 거의 즉시 끝난다.

채점 규칙:
    - 동의어를 정확히 언급 → SYNONYM_SCORE (높은 점수)
    - 상위개념만 언급 (예: '남자' 대신 '사람'만 말함) → HYPERNYM_SCORE (낮은 점수)
    - 둘 다 언급 안 함 → NO_MATCH_SCORE (0점)

점수 자체는 코드가 결정론적으로 계산한다 — LLM은 여기 전혀 관여하지 않는다
(태그의 synonyms/hypernyms 목록을 뽑아내는 extract_tags.py 단계에서만
LLM을 썼고, 그 결과를 여기서는 그냥 텍스트로 비교만 함).

실행:
    python3 score_relevance.py
"""

import json

# 매칭 방식에 따른 점수. 지난번 얘기하신 대로, 정확한 숫자는 실제 테스트해보면서
# 조정하세요 — 일단 "정확히 맞음 > 상위개념만 맞음 > 틀림" 순서만 지키면 됩니다.
SYNONYM_SCORE = 1.0
HYPERNYM_SCORE = 0.5
NO_MATCH_SCORE = 0.0


def load_tags(tags_path: str) -> dict:
    with open(tags_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_relevance(transcript: str, tags: dict) -> dict:
    concept_results = []

    for concept in tags["concepts"]:
        # 한국어는 조사가 단어 뒤에 바로 붙으니까("남자아이가"), 부분 문자열
        # 검사(in)만으로도 웬만한 경우엔 잘 걸려요. 완벽하진 않지만 첫 버전으로 충분해요.
        matched_synonym = any(syn in transcript for syn in concept["synonyms"])
        matched_hypernym = any(hyp in transcript for hyp in concept["hypernyms"])

        if matched_synonym:
            score, match_type = SYNONYM_SCORE, "synonym"
        elif matched_hypernym:
            score, match_type = HYPERNYM_SCORE, "hypernym"
        else:
            score, match_type = NO_MATCH_SCORE, "none"

        concept_results.append(
            {"label": concept["label"], "score": score, "match_type": match_type}
        )

    overall = (
        sum(r["score"] for r in concept_results) / len(concept_results)
        if concept_results
        else 0.0
    )

    return {"concepts": concept_results, "overall_relevance": overall}


if __name__ == "__main__":
    tags = load_tags("walking_tags.json")

    # 테스트하고 싶은 발화로 바꿔보세요
    transcript = (
        "어...남자...남자아이가....그.....어.....뛰어 가는 음 중이에요......."
    )

    result = score_relevance(transcript, tags)

    for c in result["concepts"]:
        print(f"- {c['label']}: {c['match_type']} → {c['score']}점")
    print(f"\n전체 관련도 점수: {result['overall_relevance']:.2f}")