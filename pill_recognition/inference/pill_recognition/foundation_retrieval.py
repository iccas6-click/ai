from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pill_recognition_legacy.aihub_classifier import (
    AIHubProductInfo,
    load_aihub_class_names,
    load_aihub_product_master,
    rotate_crop,
)

from .schemas import ProductCandidate


DEFAULT_TORCHHUB_REPO = "facebookresearch/dinov2"
DEFAULT_TORCHHUB_MODEL = "dinov2_vits14"


@dataclass(frozen=True)
class FoundationEncoderConfig:
    torchhub_repo: str = DEFAULT_TORCHHUB_REPO
    torchhub_model: str = DEFAULT_TORCHHUB_MODEL
    image_size: int = 224

    @property
    def name(self) -> str:
        return f"{self.torchhub_repo}:{self.torchhub_model}:{self.image_size}"


class FoundationImageRetriever:
    def __init__(
        self,
        mapping_path: Path,
        index_path: Path,
        device: str,
        encoder_config: FoundationEncoderConfig | None = None,
        rotation_tta: bool = True,
    ) -> None:
        if not mapping_path.exists():
            raise FileNotFoundError("AI Hub mapping does not exist")
        if not index_path.exists():
            raise FileNotFoundError(
                f"Foundation retrieval index does not exist: {index_path}."
            )

        self.device = torch.device(device)
        self.encoder_config = encoder_config or FoundationEncoderConfig()
        self.rotation_tta = rotation_tta
        self.class_names = load_aihub_class_names(mapping_path)
        self.product_master = load_aihub_product_master(
            mapping_path.parent,
            set(self.class_names.values()),
        )
        self.encoder = load_torchhub_encoder(
            self.encoder_config.torchhub_repo,
            self.encoder_config.torchhub_model,
        ).to(self.device).eval()
        self.transform = foundation_transform(self.encoder_config.image_size)

        payload = torch.load(index_path, map_location="cpu", weights_only=True)
        pill_ids = payload.get("pill_ids")
        embeddings = payload.get("embeddings")
        if not isinstance(pill_ids, list) or not torch.is_tensor(embeddings):
            raise ValueError("Foundation retrieval index has invalid format")
        self.pill_ids = [str(pill_id) for pill_id in pill_ids]
        self.index_mode = str(payload.get("index_mode", "prototype"))
        self.embeddings = torch.nn.functional.normalize(
            embeddings.float(),
            dim=1,
        ).to(self.device)
        self.model_version = (
            f"foundation-retrieval:{payload.get('encoder', self.encoder_config.name)}:"
            f"{self.index_mode}:{index_path.name}"
        )

    def predict_batch(
        self,
        crops_rgb: list[np.ndarray],
        top_k: int,
    ) -> list[list[ProductCandidate]]:
        if not crops_rgb:
            return []
        query_embeddings = self.embed_crops(crops_rgb)
        scores = query_embeddings @ self.embeddings.T
        search_k = min(max(top_k * 24, 64), scores.shape[1])
        values, indices = torch.topk(scores, search_k, dim=1)

        results = []
        for row_values, row_indices in zip(values.tolist(), indices.tolist()):
            best_by_pill_id: dict[str, float] = {}
            for score, index in zip(row_values, row_indices):
                pill_id = self.pill_ids[index]
                if pill_id not in best_by_pill_id or score > best_by_pill_id[pill_id]:
                    best_by_pill_id[pill_id] = float(score)
            ranked = sorted(
                best_by_pill_id.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            row_candidates = []
            for rank, (pill_id, score) in enumerate(ranked, start=1):
                product = self.product_master.get(pill_id)
                row_candidates.append(
                    product_candidate_from_foundation_product(
                        rank,
                        pill_id,
                        score,
                        product,
                    )
                )
            results.append(row_candidates)
        return results

    def embed_crops(self, crops_rgb: list[np.ndarray]) -> torch.Tensor:
        rotations = (0, 1, 2, 3) if self.rotation_tta else (0,)
        embedding_sum = None
        for rotation in rotations:
            batch = torch.stack(
                [
                    self.transform(
                        Image.fromarray(rotate_crop(crop, rotation)).convert("RGB")
                    )
                    for crop in crops_rgb
                ]
            ).to(self.device)
            with torch.inference_mode():
                embeddings = encode_images(self.encoder, batch)
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            embedding_sum = embeddings if embedding_sum is None else embedding_sum + embeddings
        return torch.nn.functional.normalize(embedding_sum / len(rotations), dim=1)


def product_candidate_from_foundation_product(
    rank: int,
    pill_id: str,
    score: float,
    product: AIHubProductInfo | None,
) -> ProductCandidate:
    return ProductCandidate(
        rank=rank,
        pill_id=pill_id,
        score=round(min(float(score), 1.0) * 100, 2),
        source="foundation_image_retrieval",
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
        matched="Foundation image embedding similarity",
    )


def load_torchhub_encoder(repo: str, model: str) -> torch.nn.Module:
    return torch.hub.load(repo, model)


def encode_images(encoder: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    output = encoder(batch)
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if torch.is_tensor(value):
                output = value[:, 0] if key == "last_hidden_state" else value
                break
    if not torch.is_tensor(output):
        raise TypeError("Foundation encoder output must be a tensor or supported dict")
    return output.flatten(1)


def foundation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
