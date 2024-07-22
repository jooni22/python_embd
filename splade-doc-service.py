from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, RootModel
from typing import List, Union
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import uvicorn, time, setproctitle, logging, http, json
from starlette.requests import Request

# Wyłączenie domyślnego loggera dostępu Uvicorn
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.getLevelName(logging.DEBUG))

# Ustawienie nazwy procesu
MODEL_NAME = 'naver/efficient-splade-VI-BT-large-doc'
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
model.eval()  # Ustawienie modelu w tryb ewaluacji

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
        input_data = json_body.get('inputs', 'input_list')
    except json.JSONDecodeError:
        input_data = 'invalid_json'
    
    if isinstance(input_data, list):
        input_count = len(input_data)
        input_sample = input_data[0] if input_data else ''
        input_info = f"List[{input_count}]: {input_sample}"
    else:
        input_info = str(input_data)
    
    input_info = input_info[:150] + '...' if len(input_info) > 150 else input_info
    
    logger.info(f'{host}:{port} - "{request.method} {url}" {response.status_code} {status_phrase} {formatted_process_time}ms - Input: {input_info}')
    return response

@app.post("/embed_sparse", response_model=List[EmbedSparseResponse])
async def get_sparse_embedding(request: EmbedRequest):
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
    uvicorn.run("splade-doc-service:app", host="0.0.0.0", port=4000, reload=True)

# Komentarz: Zoptymalizowano przetwarzanie batchowe dla lepszej wydajności
# Komentarz: Dodano szczegółowe logowanie z czasami wykonania
# Komentarz: Przeniesiono model na GPU, jeśli jest dostępne
