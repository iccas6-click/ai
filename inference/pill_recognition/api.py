from __future__ import annotations

import argparse
import io
from functools import lru_cache
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .pipeline import PillRecognitionPipeline
from .settings import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_pipeline() -> PillRecognitionPipeline:
    return PillRecognitionPipeline(get_settings())


def create_app(
    pipeline_factory: Callable[[], PillRecognitionPipeline] = get_pipeline,
) -> FastAPI:
    app = FastAPI(
        title="CLICK Pill Recognition API",
        version="0.1.0",
        description="RTMDet + AIHub retrieval pill recognition API.",
    )

    @app.get("/health")
    def health():
        settings = get_settings()
        return {
            "status": "ok",
            "recognizer": settings.recognizer,
            "top_k": settings.top_k,
        }

    @app.post("/recognize")
    async def recognize(file: UploadFile = File(...)):
        image_rgb = await read_upload_image(file)
        result = pipeline_factory().recognize(image_rgb)
        return result.to_dict()

    return app


async def read_upload_image(file: UploadFile) -> np.ndarray:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from error
    return np.asarray(image)


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLICK pill recognition API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "pill_recognition.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
