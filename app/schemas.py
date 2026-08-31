from typing import List, Optional
from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    image_id: str
    transcript: str


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
