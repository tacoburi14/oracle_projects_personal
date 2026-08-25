"""
CIU 최종 점수 집계 — 순수 결정론적 계산.

판정(토큰별 accurate/relevant/not_redundant)은 LLM(Qwen)이 하고, 총 단어 수·
CIU 개수·AQ 점수 같은 산수는 여기서 항상 파이썬 코드로만 계산한다.
LLM에게 숫자 계산을 맡기지 않는 이유: LLM 산수는 가끔 틀린다 — 판단은
LLM, 집계는 코드로 역할을 분리해야 결과를 신뢰할 수 있다.

이 함수는 surface/disfluency/accurate/relevant/not_redundant를 갖고 있고
counted/raw_ciu_eligible/is_duplicate/is_ciu 프로퍼티를 제공하는 어떤
객체 리스트에도 동작한다 (지금은 langgraph_judge.TokenJudgment만 넘김).
"""


def summarize(tokens: list) -> dict:
    counted = [t for t in tokens if t.counted]
    raw_ciu = sum(1 for t in tokens if t.raw_ciu_eligible)
    duplicate_ciu = sum(1 for t in tokens if t.is_duplicate)
    net_ciu = raw_ciu - duplicate_ciu
    total_excl = len(counted)
    total_raw = len(tokens)

    def term1(total):
        return round(net_ciu / total * 20, 2) if total else 0.0

    return {
        "raw_ciu": raw_ciu,
        "duplicate_ciu": duplicate_ciu,
        "net_ciu": net_ciu,
        "total_words_excl_disfluency": total_excl,
        "total_words_raw": total_raw,
        "aq_term1_excl_disfluency": term1(total_excl),
        "aq_term1_raw": term1(total_raw),
    }
