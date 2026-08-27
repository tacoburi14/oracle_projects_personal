"""
1단계 — 비전 LLM으로 그림 속 요소(사람/사물/배경/행동) 목록 추출.
그림을 등록할 때 딱 한 번만 실행하면 됨.

사람, 사물, 배경뿐 아니라 행동(동사)까지 포함해서, 각 요소마다
동의어 묶음 + 상위개념 묶음을 한국어로 자유롭게 생성한다.
(이 단계는 아직 Literal 아님 — 다음 단계에서 쓸 "후보 목록" 자체를
만드는 단계라서, 여기서는 모델이 자유롭게 뽑아내야 함.)

사전 준비:
    ollama pull qwen3-vl

실행:
    python3 extract_concepts.py
"""

import base64
import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class TagConcept(BaseModel):
    """그림 속 요소 하나(사람/사물/배경/행동)에 대한 동의어·상위개념 묶음."""

    label: str = Field(description="이 요소를 대표하는 이름 (한국어). 예: '남자아이', '날리다'")
    synonyms: List[str] = Field(
        description="이 요소를 정확히 가리키는 동의어들 (한국어). 예: ['남자', '남성', '소년']"
    )
    hypernyms: List[str] = Field(
        description="이 요소를 포함하는 더 포괄적인 상위 개념 단어들 (한국어). 예: ['사람', '인물']"
    )


class ImageConcepts(BaseModel):
    concepts: List[TagConcept]


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_concepts(image_path: str) -> ImageConcepts:
    image_b64 = encode_image(image_path)

    load_dotenv()
    api_key = os.environ.get("OLLAMA_API_KEY")

    llm = ChatOllama(
        model="gemma4:cloud",
        base_url="https://ollama.com",
        temperature=0,
        format=ImageConcepts.model_json_schema(),
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {},
    )
    structured_llm = llm.with_structured_output(ImageConcepts)

    # 예시 JSON을 파이썬 딕셔너리로 만들고 json.dumps()로 문자열화.
    # (프롬프트 문자열 안에 "를 직접 섞어 쓰면 파이썬 문자열이 중간에 끊겨서
    # 문법 에러가 나니까, 이렇게 딕셔너리 → json.dumps()로 만드는 게 안전함)
    example_output = {
        "concepts": [
            {
                "label": "남자",
                "synonyms": ["남자", "남성", "소년"],
                "hypernyms": ["사람", "인물"],
            },
            {
                "label": "뛰다",
                "synonyms": ["뛰다", "달리다"],
                "hypernyms": ["움직이다", "행동하다"],
            },
        ]
    }
    example_json_text = json.dumps(example_output, ensure_ascii=False, indent=2)

    prompt_text = (
        "이 그림에 등장하는 사람, 사물, 배경뿐 아니라 행동(동사)까지 "
        "전부 찾아서 각각에 대해 다음 두 가지를 한국어로 정리해줘.\n\n"
        "1. synonyms: 이 요소를 정확히 가리키는 동의어들\n"
        "   (예: 남자아이라면 '남자', '남성', '소년' / 날리는 행동이라면 "
        "'날리다', '띄우다')\n"
        "2. hypernyms: 이 요소를 포함하는 더 포괄적인 상위 개념 단어들\n"
        "   (예: 남자아이라면 '사람', '인물' / 날리는 행동이라면 '움직이다', "
        "'하다')\n\n"
        "행동(동사)도 반드시 최소 하나 이상 포함시켜줘.\n"
        "다른 설명이나 마크다운 없이, 반드시 순수 JSON 형식으로만 답해.\n"
        "JSON 형식 예시는 다음과 같다:\n"
        f"{example_json_text}\n\n"
        "모든 답은 한국어로만."
    )

    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            {"type": "text", "text": prompt_text},
        ]
    )

    return structured_llm.invoke([message])


if __name__ == "__main__":
    IMAGE_PATH = "walking_image.jpg"
    OUTPUT_PATH = "walking_tags.json"

    concepts = extract_concepts(IMAGE_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(concepts.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"요소 {len(concepts.concepts)}개 추출 완료 → {OUTPUT_PATH}에 저장됨\n")
    print(json.dumps(concepts.model_dump(), ensure_ascii=False, indent=2))