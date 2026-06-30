from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import supplement, pill

app = FastAPI(title="Click AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(supplement.router, prefix="/api/v1", tags=["supplement"])
app.include_router(pill.router, prefix="/api/v1", tags=["pill"])


@app.get("/health")
def health():
    return {"status": "ok"}
