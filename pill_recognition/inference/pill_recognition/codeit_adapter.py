from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import models, transforms

from .schemas import PillDetection, ProductCandidate, RecognitionResult, VisionObservation
from .settings import Settings
from .visual_features import estimate_crop_visual_features


@dataclass(frozen=True)
class CodeitProductInfo:
    class_id: int
    product_name: str
    company: str | None = None
    item_seq: str | None = None
    ingredient: str | None = None
    etc_otc_code: str | None = None
    print_front: str | None = None
    print_back: str | None = None
    drug_shape: str | None = None
    color_class1: str | None = None
    color_class2: str | None = None


class CodeitPillRecognizer:
    """Adapter for ZerofZero/codeit10_pj1 RTMDet + EfficientNet pill recognizer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.project_dir = settings.codeit_project_dir
        self.app_dir = self.project_dir / "web" / "rtmdet_cnn_server"
        self.model_dir = self.project_dir / "models"
        self.rtmdet_dir = self.model_dir / "RTMDet"
        self.cnn_dir = self.model_dir / "cnn"
        self.rtmdet_config = self.rtmdet_dir / "config.py"
        self.rtmdet_checkpoint = self.rtmdet_dir / "best_coco_bbox_mAP_epoch_15.pth"
        self.pill_yaml = self.rtmdet_dir / "pill.yaml"
        self.cnn_weights = self.cnn_dir / "cls119_classifier_v4.pt"
        self.class_mapping_csv = self.app_dir / "data" / "class_mapping.csv"
        self.pill_info_json = self.app_dir / "data" / "pill_info_master.json"
        self.device = settings.device if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device if self.device.startswith("cuda") else "cpu")
        self.raw_score_thr = float(os.getenv("CODEIT_RAW_SCORE_THR", "0.2"))
        self.group_iou_thr = float(os.getenv("CODEIT_GROUP_IOU_THR", "0.5"))
        self.display_top1_thr = float(os.getenv("CODEIT_DISPLAY_TOP1_THR", "0.3"))
        self.max_detections = int(os.getenv("CODEIT_MAX_DETECTIONS", "10"))
        self.rtmdet_class_mapping = self._load_rtmdet_names()
        self.products_by_class_id, self.products_by_name = self._load_products()
        self.detector = None
        self.cnn_model = None
        self.cnn_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @property
    def model_version(self) -> str:
        return "codeit10_pj1:rtmdet-tiny+efficientnet-b0-cls119"

    def recognize(self, image_rgb: np.ndarray, top_k: int = 3) -> RecognitionResult:
        total_start = perf_counter()
        image_rgb = ensure_rgb_uint8(image_rgb)
        height, width = image_rgb.shape[:2]
        detector_start = perf_counter()
        raw_result = self._predict_rtmdet(image_rgb)
        grouped = self._group_rtmdet_predictions(raw_result, image_rgb)
        detector_ms = elapsed_ms(detector_start)

        recognition_start = perf_counter()
        detections: list[PillDetection] = []
        for pill_index, det in enumerate(grouped, start=1):
            x1, y1, x2, y2 = det["bbox"]
            crop = image_rgb[y1:y2, x1:x2].copy()
            candidates = self._recognize_crop_candidates(
                crop,
                top_k=top_k,
                rtmdet_top=det["rtmdet_top"],
            )
            features = estimate_crop_visual_features(crop)
            vision = VisionObservation(
                shape=features.shape,
                color=features.color,
                confidence=0.55 if candidates else 0.0,
                notes="codeit10_pj1 RTMDet crop with EfficientNet-B0 product candidates.",
                raw={
                    "rtmdet_top": det["rtmdet_top"],
                    "crop_visual_features": {"shape": features.shape, "color": features.color},
                },
            )
            status, status_reason = determine_status(candidates)
            detections.append(
                PillDetection(
                    pill_id=pill_index,
                    bbox=(x1, y1, x2, y2),
                    crop_bbox=(x1, y1, x2, y2),
                    detector_confidence=round(float(det["confidence"]), 4),
                    vision=vision,
                    candidates=candidates,
                    status=status,
                    status_reason=status_reason,
                )
            )
        recognition_ms = elapsed_ms(recognition_start)

        warnings = []
        if not detections:
            warnings.append("No pill was detected by codeit RTMDet. Retake the photo with separated pills.")
        return RecognitionResult(
            image_width=width,
            image_height=height,
            pill_count=len(detections),
            model_version=self.model_version,
            detections=detections,
            warnings=warnings,
            timings_ms={
                "detector": detector_ms,
                "recognition": recognition_ms,
                "total": elapsed_ms(total_start),
            },
            candidate_scope={"recognizer": "codeit", "project_dir": str(self.project_dir)},
        )

    def recognize_crops_batch(self, crops_rgb: list[np.ndarray], top_k: int = 3) -> RecognitionResult:
        total_start = perf_counter()
        detections: list[PillDetection] = []
        max_width = 0
        max_height = 0
        for pill_index, crop in enumerate(crops_rgb, start=1):
            crop = ensure_rgb_uint8(crop)
            height, width = crop.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            candidates = self._recognize_crop_candidates(crop, top_k=top_k, rtmdet_top=[])
            features = estimate_crop_visual_features(crop)
            vision = VisionObservation(
                shape=features.shape,
                color=features.color,
                confidence=0.5 if candidates else 0.0,
                notes="codeit10_pj1 EfficientNet-B0 crop classification.",
            )
            status, status_reason = determine_status(candidates)
            detections.append(
                PillDetection(
                    pill_id=pill_index,
                    bbox=(0, 0, width, height),
                    crop_bbox=(0, 0, width, height),
                    detector_confidence=1.0,
                    vision=vision,
                    candidates=candidates,
                    status=status,
                    status_reason=status_reason,
                )
            )
        return RecognitionResult(
            image_width=max_width,
            image_height=max_height,
            pill_count=len(detections),
            model_version=f"single-crop+{self.model_version}",
            detections=detections,
            warnings=[] if detections else ["No crop was provided."],
            timings_ms={"total": elapsed_ms(total_start)},
            candidate_scope={"recognizer": "codeit", "project_dir": str(self.project_dir)},
        )

    def warmup(self) -> None:
        self._load_detector()
        self._load_cnn_model()
        dummy = np.full((128, 128, 3), 240, dtype=np.uint8)
        self._recognize_crop_candidates(dummy, top_k=1, rtmdet_top=[])

    def _load_rtmdet_names(self) -> dict[int, str]:
        self._require_file(self.pill_yaml)
        with self.pill_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", data)
        if isinstance(names, dict):
            return {int(k): str(v).strip() for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name).strip() for idx, name in enumerate(names)}
        raise ValueError("Unsupported codeit pill.yaml names format")

    def _load_products(self) -> tuple[dict[int, CodeitProductInfo], dict[str, CodeitProductInfo]]:
        self._require_file(self.class_mapping_csv)
        df = pd.read_csv(self.class_mapping_csv, low_memory=False)
        info_by_name = self._load_pill_info_master()
        products_by_class_id: dict[int, CodeitProductInfo] = {}
        products_by_name: dict[str, CodeitProductInfo] = {}
        for _, row in df.iterrows():
            class_id = safe_int(row.get("final_class_idx_merged"))
            name = clean_text(row.get("dl_name"))
            if class_id is None or not name or class_id in products_by_class_id:
                continue
            detail = info_by_name.get(normalize_name(name), {})
            product = CodeitProductInfo(
                class_id=class_id,
                product_name=name,
                company=clean_text(detail.get("dl_company") or row.get("dl_company")),
                item_seq=clean_text(detail.get("item_seq")),
                ingredient=extract_ingredients(detail.get("material") or row.get("dl_material")),
                etc_otc_code=clean_text(detail.get("etc_otc") or row.get("di_etc_otc_code")),
                print_front=clean_text(row.get("print_front")),
                print_back=clean_text(row.get("print_back")),
                drug_shape=clean_text(row.get("drug_shape")),
                color_class1=clean_text(row.get("color_class1")),
                color_class2=clean_text(row.get("color_class2")),
            )
            products_by_class_id[class_id] = product
            products_by_name.setdefault(normalize_name(name), product)
        return products_by_class_id, products_by_name

    def _load_pill_info_master(self) -> dict[str, dict[str, Any]]:
        if not self.pill_info_json.exists():
            return {}
        with self.pill_info_json.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        values = raw.values() if isinstance(raw, dict) else raw
        by_name: dict[str, dict[str, Any]] = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            name = clean_text(item.get("dl_name"))
            if name:
                by_name.setdefault(normalize_name(name), item)
        return by_name

    def _predict_rtmdet(self, image_rgb: np.ndarray):
        from mmdet.apis import inference_detector

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return inference_detector(self._load_detector(), image_bgr)

    def _load_detector(self):
        if self.detector is None:
            from mmdet.apis import init_detector

            self._require_file(self.rtmdet_config)
            self._require_file(self.rtmdet_checkpoint)
            self.detector = init_detector(
                str(self.rtmdet_config),
                str(self.rtmdet_checkpoint),
                device=self.device,
            )
        return self.detector

    def _load_cnn_model(self):
        if self.cnn_model is None:
            self._require_file(self.cnn_weights)
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 119)
            checkpoint = torch.load(str(self.cnn_weights), map_location=self.torch_device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(self.torch_device)
            model.eval()
            self.cnn_model = model
        return self.cnn_model

    def _group_rtmdet_predictions(self, raw_result, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        pred = raw_result.pred_instances
        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        h_img, w_img = image_rgb.shape[:2]
        raw_items = []
        for bbox, score, label in zip(bboxes, scores, labels):
            score = float(score)
            if score < self.raw_score_thr:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, min(int(x1), w_img - 1))
            y1 = max(0, min(int(y1), h_img - 1))
            x2 = max(0, min(int(x2), w_img))
            y2 = max(0, min(int(y2), h_img))
            if x1 >= x2 or y1 >= y2:
                continue
            raw_items.append({"bbox": (x1, y1, x2, y2), "class_id": int(label), "score": score})
        raw_items.sort(key=lambda item: item["score"], reverse=True)

        groups = []
        for item in raw_items:
            assigned = False
            for group in groups:
                if bbox_iou_xyxy(item["bbox"], group["rep_bbox"]) > self.group_iou_thr:
                    group["items"].append(item)
                    if item["score"] > group["rep_score"]:
                        group["rep_bbox"] = item["bbox"]
                        group["rep_score"] = item["score"]
                    assigned = True
                    break
            if not assigned:
                groups.append({"rep_bbox": item["bbox"], "rep_score": item["score"], "items": [item]})

        detections = []
        for group in groups:
            class_best: dict[int, float] = {}
            for item in group["items"]:
                cid = item["class_id"]
                score = item["score"]
                if cid not in class_best or score > class_best[cid]:
                    class_best[cid] = score
            rtmdet_top = [
                {
                    "rank": rank,
                    "class_id": int(cid),
                    "class_name": self.rtmdet_class_mapping.get(int(cid), f"Unknown ({cid})"),
                    "confidence": float(conf),
                }
                for rank, (cid, conf) in enumerate(
                    sorted(class_best.items(), key=lambda item: item[1], reverse=True)[:3],
                    start=1,
                )
            ]
            if not rtmdet_top or rtmdet_top[0]["confidence"] < self.display_top1_thr:
                continue
            x1, y1, x2, y2 = group["rep_bbox"]
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(rtmdet_top[0]["confidence"]),
                    "rtmdet_top": rtmdet_top,
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections[: self.max_detections]

    def _recognize_crop_candidates(
        self,
        crop_rgb: np.ndarray,
        top_k: int,
        rtmdet_top: list[dict[str, Any]],
    ) -> list[ProductCandidate]:
        candidates: list[ProductCandidate] = []
        seen_names: set[str] = set()
        for raw in self._predict_cnn_topk(crop_rgb, max(top_k, 3)):
            candidate = self._candidate_from_codeit_class(
                rank=len(candidates) + 1,
                class_id=int(raw["class_id"]),
                score=float(raw["confidence"]) * 100.0,
                source="codeit_cnn_efficientnet_b0",
                matched="codeit CNN crop classifier",
            )
            if normalize_name(candidate.product_name or candidate.pill_id) in seen_names:
                continue
            seen_names.add(normalize_name(candidate.product_name or candidate.pill_id))
            candidates.append(candidate)
            if len(candidates) >= top_k:
                break

        if len(candidates) < top_k:
            for raw in rtmdet_top:
                candidate = self._candidate_from_codeit_name(
                    rank=len(candidates) + 1,
                    class_id=safe_int(raw.get("class_id")) or -1,
                    product_name=clean_text(raw.get("class_name")) or "Unknown",
                    score=float(raw.get("confidence") or 0.0) * 100.0,
                    source="codeit_rtmdet",
                    matched="codeit RTMDet class candidate",
                )
                key = normalize_name(candidate.product_name or candidate.pill_id)
                if key in seen_names:
                    continue
                seen_names.add(key)
                candidates.append(candidate)
                if len(candidates) >= top_k:
                    break
        return candidates

    def _predict_cnn_topk(self, crop_rgb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if crop_rgb is None or crop_rgb.size == 0:
            return []
        pil_crop = Image.fromarray(crop_rgb).convert("RGB")
        input_tensor = self.cnn_transform(pil_crop).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            outputs = self._load_cnn_model()(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_cls = torch.topk(probs, min(max(top_k + 3, top_k), probs.shape[1]), dim=1)
        rows = []
        for cid, conf in zip(top_cls[0].detach().cpu().tolist(), top_prob[0].detach().cpu().tolist()):
            cid = int(cid)
            if cid == 118:
                continue
            rows.append({"class_id": cid, "confidence": float(conf)})
            if len(rows) >= top_k:
                break
        return rows

    def _candidate_from_codeit_class(
        self,
        rank: int,
        class_id: int,
        score: float,
        source: str,
        matched: str,
    ) -> ProductCandidate:
        product = self.products_by_class_id.get(class_id)
        if product is None:
            return self._candidate_from_codeit_name(rank, class_id, f"Unknown ({class_id})", score, source, matched)
        return product_candidate_from_info(product, rank, score, source, matched)

    def _candidate_from_codeit_name(
        self,
        rank: int,
        class_id: int,
        product_name: str,
        score: float,
        source: str,
        matched: str,
    ) -> ProductCandidate:
        product = self.products_by_name.get(normalize_name(product_name))
        if product is not None:
            return product_candidate_from_info(product, rank, score, source, matched)
        return ProductCandidate(
            rank=rank,
            pill_id=f"CODEIT-{class_id}",
            score=round(max(0.0, min(score, 100.0)), 2),
            source=source,
            product_name=product_name,
            ingredient=None,
            caution_points=[],
            matched=matched,
            reference_image_url=None,
        )

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required codeit asset is missing: {path}")


def product_candidate_from_info(
    product: CodeitProductInfo,
    rank: int,
    score: float,
    source: str,
    matched: str,
) -> ProductCandidate:
    return ProductCandidate(
        rank=rank,
        pill_id=f"CODEIT-{product.class_id}",
        score=round(max(0.0, min(float(score), 100.0)), 2),
        source=source,
        product_name=product.product_name,
        ingredient=product.ingredient,
        caution_points=[],
        company=product.company,
        item_seq=product.item_seq,
        etc_otc_code=product.etc_otc_code,
        print_front=product.print_front,
        print_back=product.print_back,
        drug_shape=product.drug_shape,
        color_class1=product.color_class1,
        color_class2=product.color_class2,
        matched=matched,
        reference_image_url=None,
    )


def determine_status(candidates: list[ProductCandidate]) -> tuple[str, str | None]:
    if not candidates:
        return "no_candidate", "No codeit product candidate remained after classification."
    if candidates[0].score < 25:
        return "low_confidence", "The top codeit CNN candidate score is low."
    if len(candidates) > 1 and candidates[0].score - candidates[1].score < 5:
        return "ambiguous", "The top codeit candidates are close. Ask the user to choose."
    return "recognized", None


def extract_ingredients(material: Any) -> str | None:
    text = clean_text(material)
    if not text:
        return None
    names = []
    for match in re.finditer(r"성분명\s*:\s*([^|;\n]+)", text):
        value = match.group(1).strip()
        if value and value not in names:
            names.append(value)
    if not names:
        return text[:240]
    return ", ".join(names)


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def bbox_iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
