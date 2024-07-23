from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, RootModel
from typing import List, Union
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import uvicorn, time, setproctitle, logging, http, json, sys, os
from starlette.requests import Request
from loguru import logger
import aiofiles
from datetime import datetime

# Wyłączenie domyślnego loggera dostępu Uvicorn
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

# Konfiguracja loguru
logger.remove()  # Usuń domyślny handler

# Dodaj logger dla stdout, aby widzieć logi w konsoli
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

# Ścieżka do pliku logów
log_file = "logs/splade_query_service.log"

# Upewnij się, że katalog logów istnieje
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Asynchroniczna funkcja do zapisywania logów
async def async_log_to_file(message: str):
    async with aiofiles.open(log_file, mode='a') as f:
        await f.write(f"{datetime.now().isoformat()} - {message}\n")

# Ustawienie nazwy procesu
MODEL_NAME = 'naver/efficient-splade-VI-BT-large-query'
setproctitle.setproctitle(MODEL_NAME)

app = FastAPI()

# Dodanie TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Sprawdzenie dostępności CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Załadowanie modelu SPLADE i tokenizera
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Wstępne załadowanie modelu
logger.info("Initializing model...")
dummy_input = "a"
tokens = tokenizer([dummy_input], return_tensors='pt', padding=True, truncation=True, max_length=512)
tokens = {k: v.to(device) for k, v in tokens.items()}
with torch.no_grad():
    _ = model(**tokens)
logger.info("Model initialization completed.")

BATCH_SIZE = 32

class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]

class SparseValue(BaseModel):
    index: int
    value: float

class EmbedSparseResponse(RootModel):
    root: List[SparseValue]

@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    url = f"{request.url.path}?{request.query_params}" if request.query_params else request.url.path
    start_time = time.perf_counter()
    
    body = await request.body()
    request_copy = Request(request.scope, receive=request._receive)
    
    # Log request body
    log_message = f"REQUEST: {request.method} {url}\nBody: {body.decode()}"
    await async_log_to_file(log_message)
    
    response = await call_next(request_copy)
    
    process_time = (time.perf_counter() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    host = getattr(getattr(request, "client", None), "host", None)
    port = getattr(getattr(request, "client", None), "port", None)
    
    # Log info
    log_message = f"INFO: {host}:{port} - \"{request.method} {url}\" {response.status_code} {formatted_process_time}ms"
    logger.info(log_message)
    await async_log_to_file(log_message)
    
    return response

@app.post("/embed_sparse", response_model=List[EmbedSparseResponse])
async def get_sparse_embedding_query(request: EmbedRequest):
    inputs = request.inputs
    if isinstance(inputs, str):
        inputs = [inputs]

    all_sparse_values = []
    
    for i in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            output = model(**tokens)

        sparse_vec = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens['attention_mask'].unsqueeze(-1), dim=1)[0]

        for j in range(len(batch)):
            non_zero_mask = sparse_vec[j] > 0
            non_zero_indices = non_zero_mask.nonzero().squeeze(dim=1)
            non_zero_values = sparse_vec[j][non_zero_mask]

            sorted_indices = non_zero_values.argsort(descending=True)
            top_k = min(100, len(sorted_indices))
            top_indices = non_zero_indices[sorted_indices[:top_k]]
            top_values = non_zero_values[sorted_indices[:top_k]]

            top_indices = top_indices.cpu().tolist()
            top_values = top_values.cpu().tolist()

            sparse_values = [SparseValue(index=int(idx), value=float(val)) for idx, val in zip(top_indices, top_values)]
            all_sparse_values.append(EmbedSparseResponse(root=sparse_values))

    return all_sparse_values

if __name__ == "__main__":
    uvicorn.run("splade-query-service:app", host="0.0.0.0", port=5000, reload=True)

# Komentarz: Zoptymalizowano przetwarzanie batchowe dla lepszej wydajności
# Komentarz: Dodano asynchroniczne logowanie do pliku z użyciem aiofiles
# Komentarz: Logi INFO i request body są zapisywane do jednego pliku
