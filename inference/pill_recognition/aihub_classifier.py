from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .schemas import Candidate


@dataclass(frozen=True)
class AIHubProductInfo:
    pill_id: str
    product_name: str | None = None
    company: str | None = None
    item_seq: str | None = None
    etc_otc_code: str | None = None
    ingredient: str | None = None
    chart: str | None = None
    image_url: str | None = None


class AIHubPillClassifier:
    def __init__(self, weights_path: Path, mapping_path: Path, device: str) -> None:
        if not weights_path.exists() or not mapping_path.exists():
            raise FileNotFoundError("AI Hub weights or label mapping does not exist")

        self.device = torch.device(device)
        self.class_names = load_aihub_class_names(mapping_path)
        self.product_master = load_aihub_product_master(
            mapping_path.parent,
            set(self.class_names.values()),
        )
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("model")
        if not isinstance(state_dict, dict):
            raise ValueError("AI Hub checkpoint does not contain a model state dict")

        output_classes = int(state_dict["fc.weight"].shape[0])
        if output_classes != len(self.class_names):
            raise ValueError(
                "AI Hub checkpoint and label mapping class counts differ: "
                f"{output_classes} != {len(self.class_names)}"
            )

        self.model = models.resnet152(weights=None, num_classes=output_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.model_version = f"aihub-resnet152-class01-epoch{checkpoint.get('epoch', 'unknown')}"
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_batch(
        self,
        crops_rgb: list[np.ndarray],
        top_k: int = 3,
    ) -> list[list[Candidate]]:
        if not crops_rgb:
            return []

        batch = torch.stack(
            [self.transform(Image.fromarray(crop).convert("RGB")) for crop in crops_rgb]
        ).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1)
            values, indices = torch.topk(probabilities, min(top_k, probabilities.shape[1]), dim=1)

        predictions = []
        for row_indices, row_values in zip(indices.tolist(), values.tolist()):
            row_candidates = []
            for rank, (class_id, confidence) in enumerate(
                zip(row_indices, row_values),
                start=1,
            ):
                pill_id = self.class_names[class_id]
                product = self.product_master.get(pill_id)
                row_candidates.append(
                    Candidate(
                        rank=rank,
                        class_id=class_id,
                        class_name=pill_id,
                        confidence=round(float(confidence), 4),
                        source="aihub-resnet152",
                        product_name=product.product_name if product else None,
                        company=product.company if product else None,
                        item_seq=product.item_seq if product else None,
                        etc_otc_code=product.etc_otc_code if product else None,
                        ingredient=product.ingredient if product else None,
                        chart=product.chart if product else None,
                        image_url=product.image_url if product else None,
                    )
                )
            predictions.append(row_candidates)
        return predictions


def load_aihub_product_master(
    crop_root: Path,
    pill_ids: set[str] | None = None,
) -> dict[str, AIHubProductInfo]:
    if not crop_root.exists():
        return {}

    target_ids = pill_ids or {
        path.name for path in crop_root.iterdir() if path.is_dir() and path.name.startswith("K-")
    }
    products = {}
    for pill_id in sorted(target_ids):
        sample_json = next((crop_root / pill_id).glob("*.json"), None)
        if sample_json is None:
            continue
        try:
            payload = json.loads(sample_json.read_text(encoding="utf-8"))
            image_rows = payload.get("images", [])
            if not image_rows:
                continue
            row = image_rows[0]
        except (OSError, json.JSONDecodeError):
            continue
        products[pill_id] = AIHubProductInfo(
            pill_id=pill_id,
            product_name=clean_text(row.get("dl_name")),
            company=clean_text(row.get("dl_company")),
            item_seq=clean_text(row.get("item_seq")),
            etc_otc_code=clean_text(row.get("di_etc_otc_code")),
            ingredient=clean_text(row.get("dl_material")),
            chart=clean_text(row.get("chart")),
            image_url=clean_text(row.get("img_key")),
        )
    return products


def clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_aihub_class_names(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows = payload.get("pill_label_path_sharp_score")
    if not isinstance(rows, list):
        raise ValueError("AI Hub mapping is missing pill_label_path_sharp_score")

    mapping: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError("AI Hub mapping contains an invalid label row")
        class_id, pill_id = row[:2]
        mapping[int(class_id)] = str(pill_id)

    expected_ids = set(range(len(mapping)))
    if set(mapping) != expected_ids:
        raise ValueError("AI Hub class IDs must be contiguous and start at zero")
    return mapping
