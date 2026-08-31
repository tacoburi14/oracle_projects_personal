"""
LangChain + LangGraph + Pydantic로 짠 Ollama(gemma4) 기반 CIU 채점기.

그래프 구조:
    prepare (프롬프트 조립) → judge (Ollama 구조화 출력 호출) → summarize (결정론적 집계) → END

Pydantic(JudgmentOutput)이 gemma4 출력의 "형식"을 강제한다 — 프롬프트로
"이런 JSON을 뱉어라"라고 텍스트로만 부탁하는 게 아니라, LangChain의
with_structured_output()이 그 형식을 실제로 강제(내부적으로 보정 시도 포함)한다.

산수(총 단어 수·CIU 개수·AQ 점수)는 gemma4에게 안 맡긴다 — scoring.summarize()가
항상 결정론적으로 계산한다.
"""

import os
from typing import List, Optional, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .prompts import PROMPT_TEMPLATE
from .scoring import summarize


# ── Pydantic: Gemma에게 강제할 구조화 출력 스키마 ──────────────────────────
class TokenJudgment(BaseModel):
    """어절 하나하나의 판정 결과"""

    surface: str = Field(description="어절 원문")
    stem: str = Field(default="", description="어간 (선택, LLM이 안 채워도 무방)")
    disfluency: Optional[str] = Field(default=None, description='"filler" | "false_start" | null')
    category: Optional[str] = Field(
        default=None, description="person|object|action|location|aspect|modality|bound_noun|null"
    )
    role: Optional[str] = Field(
        default=None, description="Agent|Theme|Predicate|Location|Aspect|Modality|null"
    )
    accurate: bool = Field(default=False, description="장면 정보와 사실적으로 일치하는가")
    relevant: bool = Field(default=False, description="과제(그림 설명)와 관련이 있는가")
    not_redundant: bool = Field(default=True, description="이미 나온 정보의 재지칭이 아닌가")
    note: str = Field(default="", description="Qwen의 판정 근거 한 줄")

    @property
    def counted(self) -> bool:
        return self.disfluency is None

    @property
    def raw_ciu_eligible(self) -> bool:
        return self.counted and self.accurate and self.relevant

    @property
    def is_duplicate(self) -> bool:
        return self.raw_ciu_eligible and not self.not_redundant

    @property
    def is_ciu(self) -> bool:
        return self.raw_ciu_eligible and self.not_redundant


class JudgmentOutput(BaseModel):
    """어절들의 리스트 전체. 
        Qwen 호출 하나의 최종 출력 스키마. 
        with_structured_output()이 이 모양을 보장한다."""

    tokens: List[TokenJudgment]


# ── LangGraph: 그래프 상태 ────────────────────────────────────────────────
class CIUState(TypedDict, total=False):
    transcript: str
    concepts: dict          # scene_description 대신 이제 concepts JSON이 정답 기준
    prompt: str
    tokens: List[TokenJudgment]
    summary: dict


def prepare_node(state: CIUState) -> dict:
    concepts_text = "\n".join(
        f"- {c['label']}: 동의어={c['synonyms']}, 상위개념={c['hypernyms']}"
        for c in state["concepts"]["concepts"]
    )
    prompt = PROMPT_TEMPLATE.format(concepts_text=concepts_text, transcript=state["transcript"])
    return {"prompt": prompt}


def judge_node(state: CIUState) -> dict:

    load_dotenv()

    api_key = os.environ.get("OLLAMA_API_KEY")
    llm = ChatOllama(
        model="gemma4:cloud",
        base_url="https://ollama.com",
        temperature=0,
        format=JudgmentOutput.model_json_schema(),
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {},
    )

    structured_llm = llm.with_structured_output(JudgmentOutput)
    result: JudgmentOutput = structured_llm.invoke(state["prompt"])
    return {"tokens": result.tokens}


def summarize_node(state: CIUState) -> dict:
    return {"summary": summarize(state["tokens"])}


def build_graph():
    graph = StateGraph(CIUState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("judge", judge_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "judge")
    graph.add_edge("judge", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    """그래프 컴파일은 한 번만 하고 재사용 (매 요청마다 다시 만들 필요 없음)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def score_transcript(transcript: str, concepts: dict) -> dict:
    """이 모듈의 공개 진입점.
    concepts는 extract_tags.py가 만든 JSON(dict)을 그대로 받는다 —
    이게 CIU 채점의 정답 기준이다."""
    return get_compiled_graph().invoke({"transcript": transcript, "concepts": concepts})