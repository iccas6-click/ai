from __future__ import annotations

import argparse
import io
from dataclasses import asdict
from functools import lru_cache
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .pipeline import PillRecognitionPipeline
from .product_db import (
    ProductSearchQuery,
    load_product_index,
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

    @app.post("/crops/recognize")
    async def recognize_crop(file: UploadFile = File(...)):
        crop_rgb = await read_upload_image(file)
        result = pipeline_factory().recognize_crop(crop_rgb)
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
