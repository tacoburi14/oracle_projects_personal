import os

from fastapi import FastAPI, HTTPException

from .schemas import ScoreRequest
from .gate_to_ciu import process_response


app = FastAPI(
    title="CIU Scoring Service",
    description="발화 전사문을 Ollama(gemma) + LangGraph로 CIU(Correct Information Unit) 채점하는 API.",
    version="0.3.0",
)

def get_concepts_by_image_id(image_id: str) -> dict:
    """image_id로 DB에서 해당 이미지의 태그(concepts) json을 조회해서 돌려준다.
 
    TODO: 실제 DB 연동 코드로 교체할 것. 지금은 자리만 잡아둔 스텁이라
    호출하면 NotImplementedError가 난다.
    예: SELECT tags_json FROM images WHERE id = :image_id
    이미지를 못 찾으면 여기서 HTTPException(status_code=404, ...)를 던지도록 구현하는 걸 추천.
    """
    raise NotImplementedError(
        "get_concepts_by_image_id: image_id로 DB에서 concepts json 가져오는 로직을 구현하세요."
    )
 
 
@app.post("/score")
def score_endpoint(request: ScoreRequest):
    try:
        concepts = get_concepts_by_image_id(request.image_id)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    try:
        return process_response(concepts, request.transcript)
    except Exception as e:
        # Ollama 서버 연결 실패 등 — 500으로 애매하게 죽는 대신 502로 명확히 알려주기.
        raise HTTPException(status_code=502, detail=f"채점 파이프라인 실패: {e}")
 
 
@app.get("/health")
def health():
    return {
        "status": "ok",
        # 실제 ciu_judge.py / gate_pass.py에 하드코딩된 값과 맞춤.
        # 참고: OLLAMA_HOST 환경변수는 실제 호출 코드에서 안 쓰이고 있음(base_url 하드코딩됨) — 죽은 설정.
        "model": os.environ.get("OLLAMA_MODEL", "gemma4:cloud"),
        "ollama_host": os.environ.get("OLLAMA_HOST", "https://ollama.com"),
    }
 
