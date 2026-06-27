from __future__ import annotations

import argparse
import io
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from time import perf_counter
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .pipeline import PillRecognitionPipeline
from .product_db import (
    ProductSearchQuery,
    load_product_index,
    product_reference_image_url,
    score_product,
    search_products,
)
from .settings import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_pipeline() -> PillRecognitionPipeline:
    return PillRecognitionPipeline(get_settings())


@lru_cache(maxsize=1)
def get_product_index() -> dict:
    return load_product_index(get_settings().aihub_mapping)


def create_app(
    pipeline_factory: Callable[[], PillRecognitionPipeline] = get_pipeline,
    product_index_factory: Callable[[], dict] = get_product_index,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        warmup_pipeline(app, pipeline_factory)
        yield

    app = FastAPI(
        title="CLICK Pill Recognition API",
        version="0.1.0",
        description="RTMDet + AIHub retrieval pill recognition API.",
        lifespan=lifespan,
    )
    app.state.warmup = initial_warmup_state()

    @app.get("/health")
    def health():
        settings = get_settings()
        return {
            "status": "ok",
            "recognizer": settings.recognizer,
            "top_k": settings.top_k,
            "max_batch_crops": settings.max_batch_crops,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_image_pixels": settings.max_image_pixels,
            "retrieval_query_preprocess": settings.retrieval_query_preprocess,
            "warmup": app.state.warmup,
        }

    @app.post("/recognize")
    async def recognize(file: UploadFile = File(...)):
        request_start = perf_counter()
        decode_start = perf_counter()
        image_rgb = await read_upload_image(file)
        decode_ms = elapsed_ms(decode_start)
        pipeline_get_start = perf_counter()
        pipeline = pipeline_factory()
        pipeline_get_ms = elapsed_ms(pipeline_get_start)
        pipeline_call_start = perf_counter()
        result = pipeline.recognize(image_rgb)
        attach_api_timings(
            result,
            request_start=request_start,
            upload_decode_ms=decode_ms,
            pipeline_get_ms=pipeline_get_ms,
            pipeline_call_ms=elapsed_ms(pipeline_call_start),
        )
        return result.to_dict()

    @app.post("/crops/recognize")
    async def recognize_crop(file: UploadFile = File(...)):
        request_start = perf_counter()
        decode_start = perf_counter()
        crop_rgb = await read_upload_image(file)
        decode_ms = elapsed_ms(decode_start)
        pipeline_get_start = perf_counter()
        pipeline = pipeline_factory()
        pipeline_get_ms = elapsed_ms(pipeline_get_start)
        pipeline_call_start = perf_counter()
        result = pipeline.recognize_crop(crop_rgb)
        attach_api_timings(
            result,
            request_start=request_start,
            upload_decode_ms=decode_ms,
            pipeline_get_ms=pipeline_get_ms,
            pipeline_call_ms=elapsed_ms(pipeline_call_start),
        )
        return result.to_dict()

    @app.post("/crops/recognize-batch")
    async def recognize_crops_batch(files: list[UploadFile] = File(...)):
        request_start = perf_counter()
        settings = get_settings()
        validate_crop_batch_size(files, settings.max_batch_crops)
        decode_start = perf_counter()
        crops_rgb = [await read_upload_image(file) for file in files]
        decode_ms = elapsed_ms(decode_start)
        pipeline_get_start = perf_counter()
        pipeline = pipeline_factory()
        pipeline_get_ms = elapsed_ms(pipeline_get_start)
        pipeline_call_start = perf_counter()
        result = pipeline.recognize_crops_batch(crops_rgb)
        attach_api_timings(
            result,
            request_start=request_start,
            upload_decode_ms=decode_ms,
            pipeline_get_ms=pipeline_get_ms,
            pipeline_call_ms=elapsed_ms(pipeline_call_start),
        )
        return result.to_dict()

    @app.get("/products/search")
    def search_product_db(
        imprint: str = "",
        shape: str = "",
        color: str = "",
        text: str = "",
        limit: int = 20,
    ):
        if not any(value.strip() for value in (imprint, shape, color, text)):
            raise HTTPException(
                status_code=400,
                detail="At least one of imprint, shape, color, or text is required.",
            )
        product_index = product_index_factory()
        if not product_index:
            raise HTTPException(
                status_code=503,
                detail="AI Hub product metadata is unavailable.",
            )
        query = ProductSearchQuery(
            imprint=imprint.strip(),
            shape=shape.strip(),
            color=color.strip(),
            text=text.strip(),
            limit=clamp_limit(limit),
        )
        results = search_products(product_index, query)
        return {
            "query": {
                "imprint": query.imprint,
                "shape": query.shape,
                "color": query.color,
                "text": query.text,
                "limit": query.limit,
            },
            "count": len(results),
            "results": results,
        }

    @app.get("/products/{pill_id}/reference-image")
    def get_product_reference_image(pill_id: str):
        image_path = find_reference_image(get_settings().aihub_mapping, pill_id)
        if image_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Reference image not found for {pill_id}.",
            )
        return FileResponse(
            image_path,
            media_type="image/png",
            filename=image_path.name,
        )

    @app.post("/products/refine")
    def refine_product_candidates(request: ProductRefineRequest):
        if not request.candidates and not request.has_query():
            raise HTTPException(
                status_code=400,
                detail="Candidates or at least one query field is required.",
            )
        product_index = product_index_factory()
        if not product_index:
            raise HTTPException(
                status_code=503,
                detail="AI Hub product metadata is unavailable.",
            )
        query = request.to_query()
        results = refine_candidates(product_index, request.candidates, query)
        limit = clamp_limit(request.limit)
        limited_results = results[:limit]
        settings = get_settings()
        status, status_reason = determine_refine_status(
            limited_results,
            min_score=settings.candidate_min_score,
            ambiguity_margin=settings.candidate_ambiguity_margin,
        )
        return {
            "query": {
                "imprint": query.imprint,
                "shape": query.shape,
                "color": query.color,
                "text": query.text,
                "limit": limit,
            },
            "count": len(limited_results),
            "status": status,
            "status_reason": status_reason,
            "results": limited_results,
        }

    return app


class ProductCandidateInput(BaseModel):
    pill_id: str
    score: float = 0.0
    source: str | None = None
    view: str | None = None


class ProductRefineRequest(BaseModel):
    candidates: list[ProductCandidateInput] = Field(default_factory=list)
    imprint: str = ""
    shape: str = ""
    color: str = ""
    text: str = ""
    limit: int = 3

    def has_query(self) -> bool:
        return any(
            value.strip()
            for value in (self.imprint, self.shape, self.color, self.text)
        )

    def to_query(self) -> ProductSearchQuery:
        return ProductSearchQuery(
            imprint=self.imprint.strip(),
            shape=self.shape.strip(),
            color=self.color.strip(),
            text=self.text.strip(),
            limit=max(clamp_limit(self.limit) * 8, 20),
        )


def clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), 100))


def initial_warmup_state() -> dict:
    enabled = get_settings().warmup_on_startup
    return {
        "enabled": enabled,
        "status": "pending" if enabled else "disabled",
        "duration_ms": None,
        "error": None,
    }


def warmup_pipeline(
    app: FastAPI,
    pipeline_factory: Callable[[], PillRecognitionPipeline],
) -> None:
    if not get_settings().warmup_on_startup:
        app.state.warmup = {
            "enabled": False,
            "status": "disabled",
            "duration_ms": None,
            "error": None,
        }
        return
    start = perf_counter()
    try:
        pipeline = pipeline_factory()
        if hasattr(pipeline, "warmup"):
            pipeline.warmup(load_detector=True)
        app.state.warmup = {
            "enabled": True,
            "status": "ok",
            "duration_ms": elapsed_ms(start),
            "error": None,
        }
    except Exception as error:
        app.state.warmup = {
            "enabled": True,
            "status": "failed",
            "duration_ms": elapsed_ms(start),
            "error": f"{type(error).__name__}: {error}",
        }


def attach_api_timings(
    result,
    request_start: float,
    upload_decode_ms: float,
    pipeline_get_ms: float,
    pipeline_call_ms: float,
) -> None:
    result.timings_ms.update(
        {
            "upload_decode": upload_decode_ms,
            "pipeline_get": pipeline_get_ms,
            "pipeline_call": pipeline_call_ms,
            "api_total": elapsed_ms(request_start),
        }
    )


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


def validate_crop_batch_size(files: list[UploadFile], max_batch_crops: int) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="At least one crop image is required.")
    if len(files) > max_batch_crops:
        raise HTTPException(
            status_code=400,
            detail=f"Too many crop images: got {len(files)}, max {max_batch_crops}.",
        )


def refine_candidates(
    product_index: dict,
    candidates: list[ProductCandidateInput],
    query: ProductSearchQuery,
) -> list[dict]:
    merged: dict[str, dict] = {}

    for pill_id, evidence in aggregate_candidate_evidence(candidates).items():
        product = product_index.get(pill_id)
        if product is None:
            continue
        metadata_score, reasons = score_product(product, query)
        image_score = aggregate_image_score(evidence["scores"])
        row = asdict(product)
        row.update(
            {
                "score": round(image_score + metadata_score, 2),
                "image_score": round(image_score, 2),
                "image_score_max": round(max(evidence["scores"]), 2),
                "image_evidence_count": len(evidence["scores"]),
                "views": sorted(evidence["views"]),
                "candidate_sources": sorted(evidence["sources"]),
                "metadata_score": metadata_score,
                "matched": combined_match_reason(
                    image_match_reason(len(evidence["scores"])),
                    ", ".join(reasons),
                ),
                "source": "recognition_candidates",
                "reference_image_url": product_reference_image_url(pill_id),
            }
        )
        merged[pill_id] = row

    for row in search_products(product_index, query):
        pill_id = row["pill_id"]
        current = merged.get(pill_id)
        if current is None:
            row = dict(row)
            metadata_score = float(row.get("score", 0))
            row.update(
                {
                    "score": metadata_score,
                    "image_score": 0.0,
                    "image_score_max": 0.0,
                    "image_evidence_count": 0,
                    "views": [],
                    "candidate_sources": [],
                    "metadata_score": metadata_score,
                    "source": "aihub_metadata_search",
                    "reference_image_url": product_reference_image_url(pill_id),
                }
            )
            merged[pill_id] = row
            continue
        metadata_score = float(row.get("score", 0))
        current["score"] = round(float(current["image_score"]) + metadata_score, 2)
        current["metadata_score"] = metadata_score
        current["matched"] = combined_match_reason(
            image_match_reason(int(current.get("image_evidence_count") or 1)),
            row.get("matched"),
        )

    results = list(merged.values())
    results.sort(
        key=lambda row: (
            -float(row.get("score", 0)),
            row.get("product_name") or "",
            row.get("pill_id") or "",
        )
    )
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return results


def aggregate_candidate_evidence(
    candidates: list[ProductCandidateInput],
) -> dict[str, dict]:
    evidence_by_id: dict[str, dict] = {}
    for candidate in candidates:
        pill_id = candidate.pill_id.strip()
        if not pill_id:
            continue
        evidence = evidence_by_id.setdefault(
            pill_id,
            {"scores": [], "sources": set(), "views": set()},
        )
        evidence["scores"].append(clamp_score(candidate.score))
        source = candidate.source.strip() if candidate.source else ""
        view = candidate.view.strip() if candidate.view else ""
        if source:
            evidence["sources"].add(source)
        if view:
            evidence["views"].add(view)
    return evidence_by_id


def aggregate_image_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    duplicate_bonus = min(max(len(scores) - 1, 0) * 5.0, 10.0)
    return min(max(scores) + duplicate_bonus, 100.0)


def clamp_score(score: float) -> float:
    return max(0.0, min(float(score), 100.0))


def image_match_reason(evidence_count: int) -> str:
    if evidence_count <= 1:
        return "image candidate"
    return f"image candidate x{evidence_count}"


def determine_refine_status(
    rows: list[dict],
    min_score: float,
    ambiguity_margin: float,
) -> tuple[str, str]:
    if not rows:
        return "no_candidate", "No product candidate remained after refinement."

    top_score = float(rows[0].get("score") or 0.0)
    if top_score < min_score:
        return (
            "low_confidence",
            f"Top refined score {top_score:.2f} is below the review threshold {min_score:.2f}.",
        )

    if len(rows) >= 2:
        second_score = float(rows[1].get("score") or 0.0)
        margin = top_score - second_score
        if margin < ambiguity_margin:
            return (
                "ambiguous",
                f"Top-2 refined score margin {margin:.2f} is below {ambiguity_margin:.2f}.",
            )

    return (
        "needs_confirmation",
        "Refined candidate is available, but final medication identity must be confirmed by the user.",
    )


def combined_match_reason(*parts: str | None) -> str:
    return " + ".join(part for part in parts if part)


def find_reference_image(mapping_path, pill_id: str):
    if mapping_path is None:
        return None
    if product_reference_image_url(pill_id) is None:
        return None
    product_dir = mapping_path.parent / str(pill_id).strip()
    if not product_dir.is_dir():
        return None
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = sorted(path for path in product_dir.glob(suffix) if path.is_file())
        if matches:
            return matches[0]
    return None


async def read_upload_image(file: UploadFile) -> np.ndarray:
    settings = get_settings()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded image is too large: got {len(payload)} bytes, "
                f"max {settings.max_upload_bytes}."
            ),
        )
    try:
        image = Image.open(io.BytesIO(payload))
        width, height = image.size
        pixels = width * height
        if pixels > settings.max_image_pixels:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded image has too many pixels: got {pixels}, "
                    f"max {settings.max_image_pixels}."
                ),
            )
        image = image.convert("RGB")
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
