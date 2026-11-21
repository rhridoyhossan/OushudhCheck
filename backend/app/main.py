from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.app.api.routes import router as api_router
except ImportError:
    from app.api.routes import router as api_router  # type: ignore

app = FastAPI(
    title="OushudhCheck FastAPI Backend",
    description="Read-only SQLite medicines.db API & Multi-Strip Safety Scanner Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "oushudhcheck-fastapi"}
