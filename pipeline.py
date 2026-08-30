"""
관련성 게이트 + CIU 채점을 잇는 파이프라인.

흐름:
    1. 미리 저장된 concepts JSON(1단계 결과)을 읽음
    2. 발화와 비교해서 관련성 게이트 판단 (judge_matches.py)
    3. 게이트 통과하면 → 기존 CIU 채점 파이프라인(langgraph_judge.py)으로 넘김
       게이트 통과 못하면 → CIU 채점 생략, 관련 없다는 결과만 반환

실행 (하드코딩 없이 매번 다른 값으로 테스트 가능):
    python3 pipeline.py --concepts walking_tags.json --transcript "여자가 자전거를 타고 있어"
"""

import argparse
import json
import os

from judge_matches import load_concepts, judge_matches, score_matches
from app.langgraph_judge import score_transcript


def process_response(concepts_path: str, transcript: str) -> dict:
    concepts = load_concepts(concepts_path)

    judgment = judge_matches(transcript, concepts)
    relevance_result = score_matches(judgment, concepts)

    if not relevance_result["is_relevant"]:
        return {
            "status": "irrelevant",
            "relevance": relevance_result,
            "ciu": None,
        }

    ciu_result = score_transcript(transcript)

    return {
        "status": "scored",
        "relevance": relevance_result,
        "ciu": ciu_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="그림 파일 경로 (이미 extract_tags.py로 등록된 그림)")
    parser.add_argument("--transcript", required=True, help="STT로 변환된 발화 텍스트")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    concepts_path = f"{base_name}_tags.json"

    if not os.path.exists(concepts_path):
        raise FileNotFoundError(
            f"{concepts_path}가 없어요. 먼저 이 명령어로 그림을 등록하세요:\n"
            f"  python3 extract_tags.py --image {args.image}"
        )

    result = process_response(concepts_path, args.transcript)
    print(json.dumps(result, ensure_ascii=False, indent=2))