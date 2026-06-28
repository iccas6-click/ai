from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .schemas import Candidate


class EfficientNetPillClassifier:
    NON_PILL_CLASS_ID = 118

    def __init__(self, weights_path: Path, mapping_path: Path, device: str) -> None:
        if not weights_path.exists() or not mapping_path.exists():
            raise FileNotFoundError("CNN weights or class mapping file does not exist")

        self.device = torch.device(device)
        self.class_names = load_cnn_class_names(mapping_path)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            119,
        )
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
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

    def predict(self, crop_rgb: np.ndarray, top_k: int = 3) -> list[Candidate]:
        image = Image.fromarray(crop_rgb).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            values, indices = torch.topk(probabilities, top_k + 1)

        candidates = []
        for class_id, confidence in zip(indices.tolist(), values.tolist()):
            if class_id == self.NON_PILL_CLASS_ID:
                continue
            candidates.append(
                    Candidate(
                        rank=len(candidates) + 1,
                        class_id=class_id,
                        class_name=self.class_names.get(class_id, f"Unknown ({class_id})"),
                        confidence=round(float(confidence), 4),
                        source="github-efficientnet",
                    )
            )
            if len(candidates) == top_k:
                break
        return candidates


def load_cnn_class_names(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"final_class_idx_merged", "dl_name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CNN mapping must contain columns: {sorted(required)}")
        for row in reader:
            raw_id = row.get("final_class_idx_merged", "").strip()
            name = row.get("dl_name", "").strip()
            if raw_id and name:
                mapping[int(float(raw_id))] = name
    return mapping
