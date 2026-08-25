import os

from fastapi import FastAPI, HTTPException

from .langgraph_judge import score_transcript
from .scenes import SCENES
from .schemas import ScoreRequest, ScoreResponse, SummaryOut, TokenOut

app = FastAPI(
    title="CIU Scoring Service",
    description="발화 전사문을 Ollama(Qwen) + LangGraph로 CIU(Correct Information Unit) 채점하는 API.",
    version="0.3.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.environ.get("OLLAMA_MODEL", "qwen3:4b"),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    }


@app.get("/scenes")
def list_scenes():
    return SCENES


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if req.scene_id not in SCENES:
        raise HTTPException(status_code=400, detail=f"알 수 없는 scene_id: {req.scene_id}")

    try:
        result = score_transcript(req.transcript, req.scene_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Ollama 서버 연결 실패 등 — 500으로 애매하게 죽는 대신 502로 명확히 알려주기.
        raise HTTPException(status_code=502, detail=f"Ollama 호출 실패: {e}")

    tokens = result["tokens"]
    token_out = [
        TokenOut(
            surface=t.surface,
            stem=t.stem,
            disfluency=t.disfluency,
            category=t.category,
            role=t.role,
            counted=t.counted,
            raw_ciu_eligible=t.raw_ciu_eligible,
            is_duplicate=t.is_duplicate,
            is_ciu=t.is_ciu,
            note=t.note,
        )
        for t in tokens
    ]
    summary = SummaryOut(**result["summary"])
    return ScoreResponse(scene_id=req.scene_id, tokens=token_out, summary=summary)
