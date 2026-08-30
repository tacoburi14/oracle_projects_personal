"""
그림 태그 추출 스크립트 — 그림을 새로 등록할 때 딱 한 번만 실행.
"""

import base64
import json
import os
import argparse
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class TagConcept(BaseModel):
    """그림 속 요소 하나(사람/사물/배경/행동)에 대한 분류·동의어·상위개념 묶음."""

    label: str = Field(description="이 요소를 대표하는 이름 (한국어). 예: '남자아이', '날리다'")
    category: Literal["핵심", "부가"] = Field(
        description=(
            "이 요소가 그림의 핵심 내용(주체가 누구인지, 무엇을 하는지)이면 '핵심', "
            "옷/배경/소품 등 부수적인 디테일이면 '부가'"
        )
    )
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

    example_output = {
        "concepts": [
            {
                "label": "남자",
                "category": "핵심",
                "synonyms": ["남자", "남성", "소년"],
                "hypernyms": ["사람", "인물"],
            },
            {
                "label": "뛰다",
                "category": "핵심",
                "synonyms": ["뛰다", "달리다"],
                "hypernyms": ["움직이다", "행동하다"],
            },
            {
                "label": "티셔츠",
                "category": "부가",
                "synonyms": ["티셔츠", "셔츠"],
                "hypernyms": ["옷", "의류"],
            },
        ]
    }
    example_json_text = json.dumps(example_output, ensure_ascii=False, indent=2)

    
    prompt_text = (
        "이 그림에 등장하는 사람, 사물, 배경뿐 아니라 행동(동사)까지 "
        "전부 찾아서 각각에 대해 다음 세 가지를 한국어로 정리해줘.\n\n"
        "1. category: 다음 기준으로 '핵심' 또는 '부가'로 분류해줘.\n"
        "   - 핵심: '누가 무엇을 하는가'라는 문장이 성립하는 데 반드시 필요한 요소.\n"
        "     · 주체 (사람/동물 등 행동의 주체)\n"
        "     · 핵심 행동 (동사)\n"
        "     · 그 행동에 반드시 딸린 대상 (예: '자전거를 타다'의 '자전거', "
        "'연을 날리다'의 '연' — 이게 없으면 그 행동 자체가 말이 안 되는 것들)\n"
        "   - 부가: 문장이 이미 성립한 상태에서 덧붙는 디테일.\n"
        "     · 외형/의상 (옷, 색깔, 헬멧 등 착용물)\n"
        "     · 배경/장소 (공원, 하늘, 나무 등 주변 환경)\n"
        "     · 그 외 부수적인 소품\n"
        "   판단 기준: 이 요소를 빼도 '누가 무엇을 하다'라는 문장이 여전히 "
        "말이 되면 '부가', 말이 안 되면 '핵심'.\n\n"
        "2. synonyms: 이 요소를 정확히 가리키는 동의어들\n"
        "3. hypernyms: 이 요소를 포함하는 더 포괄적인 상위 개념 단어들\n\n"
        "행동(동사)도 반드시 최소 하나 이상 포함시켜줘.\n\n"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="그림 파일 경로")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_path = f"{base_name}_tags.json"   # 이미지 이름에서 자동 생성

    concepts = extract_concepts(args.image)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concepts.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"요소 {len(concepts.concepts)}개 추출 완료 → {output_path}에 저장됨\n")
    print(json.dumps(concepts.model_dump(), ensure_ascii=False, indent=2))