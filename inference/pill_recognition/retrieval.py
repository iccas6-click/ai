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

from .schemas import ProductCandidate
from .settings import Settings


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
        self.embeddings = torch.nn.functional.normalize(
            embeddings.float(),
            dim=1,
        ).to(self.device)
        self.model_version = f"aihub-resnet152-retrieval:{index_path.name}"

    @classmethod
    def from_settings(cls, settings: Settings) -> "AIHubResNetRetriever":
        if settings.aihub_weights is None or settings.aihub_mapping is None:
            raise FileNotFoundError("AI Hub retrieval assets are not configured")
        return cls(
            settings.aihub_weights,
            settings.aihub_mapping,
            settings.retrieval_index,
            settings.device,
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
        values, indices = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        results = []
        for row_values, row_indices in zip(values.tolist(), indices.tolist()):
            row_candidates = []
            for rank, (score, index) in enumerate(zip(row_values, row_indices), start=1):
                pill_id = self.pill_ids[index]
                product = self.product_master.get(pill_id)
                row_candidates.append(
                    product_candidate_from_aihub_product(
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
                embeddings = self.encoder(batch).flatten(1)
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            embedding_sum = embeddings if embedding_sum is None else embedding_sum + embeddings
        return torch.nn.functional.normalize(embedding_sum / len(rotations), dim=1)


def product_candidate_from_aihub_product(
    rank: int,
    pill_id: str,
    score: float,
    product: AIHubProductInfo | None,
) -> ProductCandidate:
    return ProductCandidate(
        rank=rank,
        pill_id=pill_id,
        score=round(float(score) * 100, 2),
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
        matched="AIHub ResNet embedding similarity",
    )


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

