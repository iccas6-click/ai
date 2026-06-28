from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from pill_recognition_legacy.aihub_classifier import (
    AIHubProductInfo,
    load_aihub_class_names,
    load_aihub_product_master,
    rotate_crop,
)

from .query_preprocess import preprocess_query_crop
from .schemas import ProductCandidate
from .product_db import product_reference_image_url
from .settings import Settings
from .visual_features import CropVisualFeatures, estimate_crop_visual_features


@dataclass(frozen=True)
class RetrievalAssetPaths:
    weights: Path
    mapping: Path
    index: Path


class AIHubResNetRetriever:
    def __init__(
        self,
        weights_path: Path,
        mapping_path: Path,
        index_path: Path,
        device: str,
        rotation_tta: bool = True,
        metadata_rerank: bool = False,
        query_preprocess: str = "none",
    ) -> None:
        if not weights_path.exists() or not mapping_path.exists():
            raise FileNotFoundError("AI Hub weights or mapping does not exist")
        if not index_path.exists():
            raise FileNotFoundError(
                f"Retrieval index does not exist: {index_path}. "
                "Run build_retrieval_index.py first."
            )
        self.device = torch.device(device)
        self.rotation_tta = rotation_tta
        self.metadata_rerank = metadata_rerank
        self.query_preprocess = query_preprocess
        self.class_names = load_aihub_class_names(mapping_path)
        self.product_master = load_aihub_product_master(
            mapping_path.parent,
            set(self.class_names.values()),
        )
        self.encoder = load_aihub_resnet_encoder(weights_path).to(self.device).eval()
        self.transform = retrieval_transform()

        payload = torch.load(index_path, map_location="cpu", weights_only=True)
        pill_ids = payload.get("pill_ids")
        embeddings = payload.get("embeddings")
        if not isinstance(pill_ids, list) or not torch.is_tensor(embeddings):
            raise ValueError("Retrieval index has invalid format")
        self.pill_ids = [str(pill_id) for pill_id in pill_ids]
        self.index_positions_by_pill_id = build_index_positions(self.pill_ids)
        self.index_mode = str(payload.get("index_mode", "prototype"))
        self.embeddings = torch.nn.functional.normalize(
            embeddings.float(),
            dim=1,
        ).to(self.device)
        self.model_version = (
            f"aihub-resnet152-retrieval:{self.index_mode}:{index_path.name}"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "AIHubResNetRetriever":
        if settings.aihub_weights is None or settings.aihub_mapping is None:
            raise FileNotFoundError("AI Hub retrieval assets are not configured")
        return cls(
            settings.aihub_weights,
            settings.aihub_mapping,
            settings.retrieval_index,
            settings.device,
            metadata_rerank=settings.retrieval_metadata_rerank,
            query_preprocess=settings.retrieval_query_preprocess,
        )

    def predict_batch(
        self,
        crops_rgb: list[np.ndarray],
        top_k: int,
        allowed_pill_ids: set[str] | None = None,
    ) -> list[list[ProductCandidate]]:
        if not crops_rgb:
            return []
        selected_index_positions = self._selected_index_positions(allowed_pill_ids)
        if allowed_pill_ids and not selected_index_positions:
            return [[] for _ in crops_rgb]
        query_embeddings = self.embed_crops(crops_rgb)
        scores = query_embeddings @ self.embeddings.T
        variant_count = infer_variant_count(
            query_embeddings.shape[0],
            crop_count=len(crops_rgb),
        )
        if variant_count > 1:
            scores = scores.reshape(len(crops_rgb), variant_count, -1).max(dim=1).values
        if selected_index_positions:
            selected_tensor = torch.tensor(
                selected_index_positions,
                device=scores.device,
                dtype=torch.long,
            )
            scores = scores.index_select(1, selected_tensor)
        search_k = min(max(top_k * 24, 64), scores.shape[1])
        values, indices = torch.topk(scores, search_k, dim=1)
        crop_features = (
            [estimate_crop_visual_features(crop) for crop in crops_rgb]
            if self.metadata_rerank
            else [CropVisualFeatures() for _ in crops_rgb]
        )
        results = []
        for row_values, row_indices, features in zip(
            values.tolist(),
            indices.tolist(),
            crop_features,
        ):
            best_by_pill_id: dict[str, float] = {}
            for score, index in zip(row_values, row_indices):
                source_index = (
                    selected_index_positions[index]
                    if selected_index_positions
                    else index
                )
                pill_id = self.pill_ids[source_index]
                product = self.product_master.get(pill_id)
                reranked_score = (
                    apply_metadata_rerank(float(score), product, features)
                    if self.metadata_rerank
                    else float(score)
                )
                if (
                    pill_id not in best_by_pill_id
                    or reranked_score > best_by_pill_id[pill_id]
                ):
                    best_by_pill_id[pill_id] = reranked_score
            ranked = sorted(
                best_by_pill_id.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            row_candidates = []
            for rank, (pill_id, score) in enumerate(ranked, start=1):
                product = self.product_master.get(pill_id)
                row_candidates.append(
                    product_candidate_from_aihub_product(
                        rank,
                        pill_id,
                        score,
                        product,
                        features if self.metadata_rerank else None,
                    )
                )
            results.append(row_candidates)
        return results

    def _selected_index_positions(self, allowed_pill_ids: set[str] | None) -> list[int]:
        if not allowed_pill_ids:
            return []
        positions = []
        for pill_id in sorted(allowed_pill_ids):
            positions.extend(self.index_positions_by_pill_id.get(pill_id, []))
        return positions

    def embed_crops(self, crops_rgb: list[np.ndarray]) -> torch.Tensor:
        rotations = (0, 1, 2, 3) if self.rotation_tta else (0,)
        preprocess_modes = query_preprocess_modes(self.query_preprocess)
        preprocessed_crops = []
        for crop in crops_rgb:
            preprocessed_crops.extend(
                preprocess_query_crop(crop, mode) for mode in preprocess_modes
            )
        embedding_sum = None
        for rotation in rotations:
            batch = torch.stack(
                [
                    self.transform(
                        Image.fromarray(rotate_crop(crop, rotation)).convert("RGB")
                    )
                    for crop in preprocessed_crops
                ]
            ).to(self.device)
            with torch.inference_mode():
                embeddings = self.encoder(batch).flatten(1)
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            embedding_sum = embeddings if embedding_sum is None else embedding_sum + embeddings
        return torch.nn.functional.normalize(embedding_sum / len(rotations), dim=1)


def query_preprocess_modes(mode: str) -> list[str]:
    normalized = str(mode or "none").strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return ["none"]
    aliases = {
        "multi": ["none", "foreground"],
        "multi_foreground": ["none", "foreground"],
        "multi_foreground_dark": ["none", "foreground", "foreground_dark"],
        "multi_grabcut": ["none", "grabcut_dark"],
        "multi_grabcut_dark": ["none", "grabcut_dark"],
    }
    if normalized in aliases:
        return aliases[normalized]
    if "+" in normalized:
        modes = [part.strip() for part in normalized.split("+") if part.strip()]
        if modes:
            return dedupe_preserving_order(modes)
    return [normalized]


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def infer_variant_count(embedding_count: int, crop_count: int) -> int:
    if crop_count <= 0:
        return 1
    if embedding_count == crop_count:
        return 1
    if embedding_count % crop_count != 0:
        raise ValueError(
            "Query embedding count must be equal to crop count or a fixed "
            "multiple of crop count."
        )
    return max(1, embedding_count // crop_count)


def product_candidate_from_aihub_product(
    rank: int,
    pill_id: str,
    score: float,
    product: AIHubProductInfo | None,
    features: CropVisualFeatures | None = None,
) -> ProductCandidate:
    matched = "AIHub ResNet embedding similarity"
    if features and (features.color or features.shape):
        details = []
        if features.color:
            details.append(f"color={features.color}")
        if features.shape:
            details.append(f"shape={features.shape}")
        matched = f"{matched} + metadata rerank ({', '.join(details)})"
    return ProductCandidate(
        rank=rank,
        pill_id=pill_id,
        score=round(min(float(score), 1.0) * 100, 2),
        source="aihub_resnet_retrieval",
        product_name=product.product_name if product else None,
        ingredient=product.ingredient if product else None,
        company=product.company if product else None,
        item_seq=product.item_seq if product else None,
        etc_otc_code=product.etc_otc_code if product else None,
        print_front=product.print_front if product else None,
        print_back=product.print_back if product else None,
        drug_shape=product.drug_shape if product else None,
        color_class1=product.color_class1 if product else None,
        color_class2=product.color_class2 if product else None,
        matched=matched,
        reference_image_url=product_reference_image_url(pill_id),
    )


def apply_metadata_rerank(
    score: float,
    product: AIHubProductInfo | None,
    features: CropVisualFeatures,
) -> float:
    if product is None:
        return score

    bonus = 0.0
    product_colors = {
        normalize_metadata_text(product.color_class1),
        normalize_metadata_text(product.color_class2),
    }
    if features.color and normalize_metadata_text(features.color) in product_colors:
        bonus += 0.006

    product_shape = normalize_metadata_text(product.drug_shape)
    feature_shape = normalize_metadata_text(features.shape)
    if feature_shape and product_shape:
        if feature_shape == product_shape:
            bonus += 0.003
        elif {feature_shape, product_shape} <= {"타원형", "장방형"}:
            bonus += 0.0015

    return score + bonus


def normalize_metadata_text(value: str | None) -> str:
    return str(value or "").strip()


def build_index_positions(pill_ids: list[str]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for index, pill_id in enumerate(pill_ids):
        positions.setdefault(pill_id, []).append(index)
    return positions


def load_aihub_resnet_encoder(weights_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("AI Hub checkpoint does not contain a model state dict")
    output_classes = int(state_dict["fc.weight"].shape[0])
    model = models.resnet152(weights=None, num_classes=output_classes)
    model.load_state_dict(state_dict)
    model.fc = torch.nn.Identity()
    return model


def retrieval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
