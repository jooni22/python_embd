from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Optional, Union
import uvicorn, time, setproctitle, torch, logging, http, json
from starlette.requests import Request

# Wyłączenie domyślnego loggera dostępu Uvicorn
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.getLevelName(logging.DEBUG))

# Ustawienie nazwy procesu
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
setproctitle.setproctitle(MODEL_NAME)

app = FastAPI()

# Dodanie TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Inicjalizacja modelu
rerank_module = SentenceTransformer(MODEL_NAME)
rerank_module.to('cuda' if torch.cuda.is_available() else 'cpu')
rerank_module.encode([" "])
BATCH_SIZE = 32

class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    truncate: bool

class RerankResponse(BaseModel):
    index: int
    score: float
    text: Optional[str]

@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    url = f"{request.url.path}?{request.query_params}" if request.query_params else request.url.path
    start_time = time.perf_counter()
    
    body = await request.body()
    request_copy = Request(request.scope, receive=request._receive)
    response = await call_next(request_copy)
    
    process_time = (time.perf_counter() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    host = getattr(getattr(request, "client", None), "host", None)
    port = getattr(getattr(request, "client", None), "port", None)
    try:
        status_phrase = http.HTTPStatus(response.status_code).phrase
    except ValueError:
        status_phrase = ""
    
    try:
        json_body = json.loads(body)
        query = json_body.get('query', '')
        texts = json_body.get('texts', [])
        input_info = f"Query: {query[:30]}..., Texts: {len(texts)}"
    except json.JSONDecodeError:
        input_info = 'invalid_json'
    
    logger.info(f'{host}:{port} - "{request.method} {url}" {response.status_code} {status_phrase} {formatted_process_time}ms - Input: {input_info}')
    return response

@app.post("/rerank", response_model=List[RerankResponse])
async def get_rerank_embeddings(request: RerankRequest):
    query = request.query
    texts = request.texts
    truncate = request.truncate

    # Przetwarzanie w batchach
    all_embeddings = []
    for i in range(0, len(texts) + 1, BATCH_SIZE):
        batch = [query] + texts[i:i+BATCH_SIZE-1] if i == 0 else texts[i:i+BATCH_SIZE]
        batch_embeddings = rerank_module.encode(batch, convert_to_tensor=True)
        all_embeddings.extend(batch_embeddings)

    query_embedding = all_embeddings[0]
    text_embeddings = all_embeddings[1:]

    similarities = cos_sim(query_embedding, torch.stack(text_embeddings)).flatten()

    reranked_results = []
    for i, score in enumerate(similarities):
        reranked_result = RerankResponse(index=i, score=float(score), text=texts[i] if not truncate else None)
        reranked_results.append(reranked_result)

    reranked_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)

    return reranked_results

if __name__ == "__main__":
    uvicorn.run("reranking-service:app", host="0.0.0.0", port=8000, reload=True)

# Komentarz: Zoptymalizowano przetwarzanie batchowe dla lepszej wydajności
# Komentarz: Dodano szczegółowe logowanie z czasami wykonania
# Komentarz: Przeniesiono model na GPU, jeśli jest dostępne