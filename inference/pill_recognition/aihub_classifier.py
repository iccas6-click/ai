from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .schemas import Candidate


class AIHubPillClassifier:
    def __init__(self, weights_path: Path, mapping_path: Path, device: str) -> None:
        if not weights_path.exists() or not mapping_path.exists():
            raise FileNotFoundError("AI Hub weights or label mapping does not exist")

        self.device = torch.device(device)
        self.class_names = load_aihub_class_names(mapping_path)
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
            predictions.append(
                [
                    Candidate(
                        rank=rank,
                        class_id=class_id,
                        class_name=self.class_names[class_id],
                        confidence=round(float(confidence), 4),
                        source="aihub-resnet152",
                    )
                    for rank, (class_id, confidence) in enumerate(
                        zip(row_indices, row_values),
                        start=1,
                    )
                ]
            )
        return predictions


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
