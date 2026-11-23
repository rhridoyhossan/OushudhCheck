from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.app.api.routes import get_scanner, router as api_router
except ImportError:
    from app.api.routes import get_scanner, router as api_router  # type: ignore


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await run_in_threadpool(get_scanner)
    yield


app = FastAPI(
    title="OushudhCheck FastAPI Backend",
    description="Read-only SQLite medicines.db API & Multi-Strip Safety Scanner Backend",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router, prefix="/api")


@app.get("/health")
def health_check():
    scanner = get_scanner()
    return {
        "status": "ok",
        "service": "oushudhcheck-fastapi",
        "detector": scanner.detector.get_model_info(),
    }
