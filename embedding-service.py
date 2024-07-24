from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
from urllib.parse import urlparse, parse_qs
import uvicorn, time, setproctitle, torch, logging, http, json, sys, os
from uvicorn.config import LOGGING_CONFIG
from typing import Any, Dict
from starlette.requests import Request
import aiofiles
from datetime import datetime

# Disable uvicorn access logger
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

# Konfiguracja loguru
logger.remove()  # Usuń domyślny handler
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

# Ścieżka do pliku logów
log_file = "logs/embedding_service.log"

# Upewnij się, że katalog logów istnieje
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Asynchroniczna funkcja do zapisywania logów
async def async_log_to_file(message: str):
    async with aiofiles.open(log_file, mode='a') as f:
        await f.write(f"{datetime.now().isoformat()} - {message}\n")

# Ustawienie nazwy procesu
MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
setproctitle.setproctitle(MODEL_NAME)

app = FastAPI()

# Dodanie TrustedHostMiddleware (opcjonalne, ale zalecane dla bezpieczeństwa)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Inicjalizacja modelu
model = SentenceTransformer(MODEL_NAME)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.encode([" "])
BATCH_SIZE = 32

class Item(BaseModel):
    input: Union[List[str], str]
    model: str

class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int    

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0}

@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    url = f"{request.url.path}?{request.query_params}" if request.query_params else request.url.path
    start_time = time.perf_counter()
    
    # Odczytaj body żądania
    body = await request.body()
    
    # Log request body
    log_message = f"REQUEST: {request.method} {url}\nBody: {body.decode()}"
    await async_log_to_file(log_message)
    
    # Utwórz nowy obiekt Request z oryginalnym body
    request_copy = Request(request.scope, receive=request._receive)
    
    # Przetwórz żądanie
    response = await call_next(request_copy)
    
    process_time = (time.perf_counter() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    host = getattr(getattr(request, "client", None), "host", None)
    port = getattr(getattr(request, "client", None), "port", None)
    try:
        status_phrase = http.HTTPStatus(response.status_code).phrase
    except ValueError:
        status_phrase = ""
    
    # Parsuj body jako JSON
    try:
        json_body = json.loads(body)
        input_data = json_body.get('input', 'input_list')
    except json.JSONDecodeError:
        input_data = 'invalid_json'
    
    # Przygotuj informację o inputie
    if isinstance(input_data, list):
        input_count = len(input_data)
        input_sample = input_data[0] if input_data else ''
        input_info = f"List[{input_count}]: {input_sample}"
    else:
        input_info = str(input_data)
    
    # Skrócenie input_info do maksymalnie 150 znaków
    input_info = input_info[:150] + '...' if len(input_info) > 150 else input_info
    
    # Log info
    log_message = f"INFO: {host}:{port} - \"{request.method} {url}\" {response.status_code} {status_phrase} {formatted_process_time}ms - Input: {input_info}"
    logger.info(log_message)
    await async_log_to_file(log_message)
    
    return response

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: Request, item: Item):
    query_params = parse_qs(urlparse(str(request.url)).query)
    api_version = query_params.get("api-version")
    if api_version:
        print(f"API version: {api_version}")
        
    if isinstance(item.input, str):
        input_list = [item.input]
    else:
        input_list = item.input

    embeddings = []
    total_tokens = 0
    start_time = time.time()
    # Przetwarzanie w batchach
    for i in range(0, len(input_list), BATCH_SIZE):
        batch = input_list[i:i+BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        
        for j, embedding in enumerate(batch_embeddings):
            embeddings.append(Embedding(
                object="embedding",
                embedding=embedding.cpu().tolist(),
                index=i+j
            ))
            total_tokens += len(batch[j].split())
    
    response = EmbeddingResponse(
        object="list",
        data=embeddings,
        model=item.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )
    torch.cuda.empty_cache()
    return response

if __name__ == "__main__":
    uvicorn.run("embedding-service:app", host="0.0.0.0", port=6000, reload=True)

# Komentarz: Zoptymalizowano przetwarzanie batchowe dla lepszej wydajności
# Komentarz: Użyto time.perf_counter() dla bardziej precyzyjnego pomiaru czasu
# Komentarz: Dodano asynchroniczne logowanie do pliku z użyciem aiofiles
# Komentarz: Logi INFO i request body są zapisywane do jednego pliku