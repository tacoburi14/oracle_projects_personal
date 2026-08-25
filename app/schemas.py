from typing import List, Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    transcript: str = Field(..., description="STT 등으로 얻은 발화 전사문 (필러/멈춤 표시 포함 가능)")
    scene_id: str = Field("kite", description="채점 기준이 될 장면(그림) id")


class TokenOut(BaseModel):
    surface: str
    stem: str = ""
    disfluency: Optional[str] = None
    category: Optional[str] = None
    role: Optional[str] = None
    counted: bool
    raw_ciu_eligible: bool
    is_duplicate: bool
    is_ciu: bool
    note: str


class SummaryOut(BaseModel):
    raw_ciu: int
    duplicate_ciu: int
    net_ciu: int
    total_words_excl_disfluency: int
    total_words_raw: int
    aq_term1_excl_disfluency: float
    aq_term1_raw: float


class ScoreResponse(BaseModel):
    scene_id: str
    tokens: List[TokenOut]
    summary: SummaryOut
