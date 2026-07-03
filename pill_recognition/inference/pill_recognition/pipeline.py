from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from .image_quality import assess_image_quality
from .product_db import (
    ProductSearchQuery,
    load_product_index,
    product_reference_image_url,
    search_products,
)
from .retrieval import AIHubResNetRetriever
from .retrieval import query_preprocess_modes
from .schemas import (
    PillDetection,
    ProductCandidate,
    RecognitionResult,
    VisionObservation,
    VisionProductCandidate,
)
from .settings import Settings
from .query_preprocess import preprocess_query_crop
from .vision_providers import create_vision_provider
from .visual_features import CropVisualFeatures, estimate_crop_visual_features
from pill_recognition_legacy.aihub_classifier import AIHubPillClassifier
from pill_recognition_legacy.schemas import Candidate


class PillRecognitionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        detector=None,
        vision_provider=None,
        retriever=None,
        aihub_classifier=None,
        product_index: dict | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.detector = detector
        self.retriever = retriever or self._load_retriever()
        self.aihub_classifier = aihub_classifier or self._load_aihub_classifier()
        self.vision_provider = vision_provider
        if self.settings.recognizer == "gemini" and self.vision_provider is None:
            self.vision_provider = create_vision_provider(self.settings)
        self.product_index = (
            product_index
            if product_index is not None
            else load_product_index(self.settings.aihub_mapping)
        )

    def _load_detector(self):
        from pill_recognition_legacy.assets import download_model_assets
        from pill_recognition_legacy.detector import RTMDetPillDetector
        from pill_recognition_legacy.settings import Settings as LegacySettings

        legacy_settings = LegacySettings.from_env()
        assets = download_model_assets(legacy_settings)
        return RTMDetPillDetector(
            assets.checkpoint,
            assets.class_names,
            legacy_settings,
        )

    def _load_retriever(self):
        if self.settings.recognizer != "retrieval":
            return None
        return AIHubResNetRetriever.from_settings(self.settings)

    def _load_aihub_classifier(self):
        if self.settings.recognizer != "aihub_classifier":
            return None
        if self.settings.aihub_weights is None or self.settings.aihub_mapping is None:
            raise FileNotFoundError("AI Hub classifier assets are not configured")
        return AIHubPillClassifier(
            self.settings.aihub_weights,
            self.settings.aihub_mapping,
            self.settings.device,
        )

    def recognize(
        self,
        image_rgb: np.ndarray,
        allowed_pill_ids: set[str] | None = None,
    ) -> RecognitionResult:
        total_start = perf_counter()
        image_rgb = ensure_rgb_uint8(image_rgb)
        height, width = image_rgb.shape[:2]
        quality_start = perf_counter()
        warnings = assess_image_quality(image_rgb, context="image")
        candidate_scope = self.candidate_scope_summary(allowed_pill_ids)
        quality_ms = elapsed_ms(quality_start)
        detected_crops = []
        detections = []
        detector_start = perf_counter()
        detector = self._get_detector()

        for pill_id, (bbox, detector_candidates) in enumerate(
            detector.predict(image_rgb),
            start=1,
        ):
            bbox = clamp_bbox(bbox, width, height)
            crop_bbox = expand_bbox(
                bbox,
                width,
                height,
                self.settings.crop_padding_ratio,
            )
            x1, y1, x2, y2 = crop_bbox
            crop = image_rgb[y1:y2, x1:x2]
            detected_crops.append(
                (
                    pill_id,
                    bbox,
                    crop_bbox,
                    crop,
                    float(detector_candidates[0].confidence)
                    if detector_candidates
                    else 0.0,
                )
            )
        detector_ms = elapsed_ms(detector_start)

        crops = [crop for _, _, _, crop, _ in detected_crops]
        recognition_start = perf_counter()
        recognition_batches, vision_observations = self._recognize_crops(
            crops,
            allowed_pill_ids=allowed_pill_ids,
        )
        recognition_ms = elapsed_ms(recognition_start)

        postprocess_start = perf_counter()
        for (
            pill_id,
            bbox,
            crop_bbox,
            _,
            detector_confidence,
        ), candidates, vision in zip(
            detected_crops,
            recognition_batches,
            vision_observations,
        ):
            status, status_reason = determine_status(
                candidates,
                min_score=self.settings.candidate_min_score,
                ambiguity_margin=self.settings.candidate_ambiguity_margin,
            )
            detections.append(
                PillDetection(
                    pill_id=pill_id,
                    bbox=bbox,
                    crop_bbox=crop_bbox,
                    detector_confidence=round(detector_confidence, 4),
                    vision=vision,
                    candidates=candidates,
                    status=status,
                    status_reason=status_reason,
                )
            )

        if not self.product_index:
            warnings.append("AI Hub product metadata is unavailable.")
        if scope_has_no_recognition_match(candidate_scope):
            warnings.append(
                "No allowed pill IDs are present in the active recognition index. "
                "Check the user's medication K-ID mapping."
            )
        if not detections:
            warnings.append("No pill was detected. Retake the photo with separated pills.")
        postprocess_ms = elapsed_ms(postprocess_start)

        return RecognitionResult(
            image_width=width,
            image_height=height,
            pill_count=len(detections),
            model_version=f"rtmdet-single-class+{self._recognizer_version()}",
            detections=detections,
            warnings=warnings,
            timings_ms={
                "quality": quality_ms,
                "detector": detector_ms,
                "recognition": recognition_ms,
                "postprocess": postprocess_ms,
                "total": elapsed_ms(total_start),
            },
            candidate_scope=candidate_scope,
        )

    def recognize_crop(
        self,
        crop_rgb: np.ndarray,
        allowed_pill_ids: set[str] | None = None,
    ) -> RecognitionResult:
        result = self.recognize_crops_batch(
            [crop_rgb],
            allowed_pill_ids=allowed_pill_ids,
        )
        result.model_version = f"single-crop+{self._recognizer_version()}"
        return result

    def recognize_crops_batch(
        self,
        crops_rgb: list[np.ndarray],
        allowed_pill_ids: set[str] | None = None,
    ) -> RecognitionResult:
        total_start = perf_counter()
        preprocess_start = perf_counter()
        crops = [ensure_rgb_uint8(crop) for crop in crops_rgb]
        candidate_scope = self.candidate_scope_summary(allowed_pill_ids)
        warnings = []
        for index, crop in enumerate(crops, start=1):
            warnings.extend(assess_image_quality(crop, context=f"crop {index}"))
        preprocess_ms = elapsed_ms(preprocess_start)
        recognition_start = perf_counter()
        recognition_batches, vision_observations = self._recognize_crops(
            crops,
            allowed_pill_ids=allowed_pill_ids,
        )
        recognition_ms = elapsed_ms(recognition_start)
        postprocess_start = perf_counter()
        detections = []
        max_width = 0
        max_height = 0

        for pill_id, crop, candidates, vision in zip(
            range(1, len(crops) + 1),
            crops,
            recognition_batches,
            vision_observations,
        ):
            height, width = crop.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            status, status_reason = determine_status(
                candidates,
                min_score=self.settings.candidate_min_score,
                ambiguity_margin=self.settings.candidate_ambiguity_margin,
            )
            detections.append(
                PillDetection(
                    pill_id=pill_id,
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
            model_version=f"crop-batch+{self._recognizer_version()}",
            detections=detections,
            warnings=scope_warnings(
                warnings if detections else ["No crop was provided."],
                candidate_scope,
            ),
            timings_ms={
                "preprocess": preprocess_ms,
                "recognition": recognition_ms,
                "postprocess": elapsed_ms(postprocess_start),
                "total": elapsed_ms(total_start),
            },
            candidate_scope=candidate_scope,
        )

    def _recognize_crops(
        self,
        crops: list[np.ndarray],
        allowed_pill_ids: set[str] | None = None,
    ) -> tuple[list[list[ProductCandidate]], list[VisionObservation]]:
        if self.settings.recognizer == "retrieval":
            return (
                recognize_crops_with_retriever(
                    self.retriever,
                    crops,
                    self.settings.top_k,
                    allowed_pill_ids=allowed_pill_ids,
                ),
                [local_visual_observation(crop) for crop in crops],
            )
        if self.settings.recognizer == "aihub_classifier":
            return (
                recognize_crops_with_aihub_classifier(
                    self.aihub_classifier,
                    crops,
                    self.settings.top_k,
                    query_preprocess=self.settings.aihub_classifier_query_preprocess,
                    allowed_pill_ids=allowed_pill_ids,
                ),
                [local_visual_observation(crop) for crop in crops],
            )
        observations = inspect_crops_safely(self.vision_provider, crops)
        return (
            [
                llm_product_candidates(observation)[: self.settings.top_k]
                for observation in observations
            ],
            observations,
        )

    def _recognizer_version(self) -> str:
        if self.settings.recognizer == "retrieval" and self.retriever is not None:
            return self.retriever.model_version
        if (
            self.settings.recognizer == "aihub_classifier"
            and self.aihub_classifier is not None
        ):
            return (
                f"{self.aihub_classifier.model_version}"
                f":{self.settings.aihub_classifier_query_preprocess}"
            )
        if self.vision_provider is not None:
            return f"{self.vision_provider.name}+gemini"
        return self.settings.recognizer

    def _get_detector(self):
        if self.detector is None:
            self.detector = self._load_detector()
        return self.detector

    def warmup(self, load_detector: bool = True) -> None:
        if load_detector:
            self._get_detector()
        if self.settings.recognizer == "retrieval" and self.retriever is not None:
            dummy_crop = np.full((96, 96, 3), 240, dtype=np.uint8)
            self.retriever.predict_batch([dummy_crop], top_k=1)
        if (
            self.settings.recognizer == "aihub_classifier"
            and self.aihub_classifier is not None
        ):
            dummy_crop = np.full((96, 96, 3), 240, dtype=np.uint8)
            recognize_crops_with_aihub_classifier(
                self.aihub_classifier,
                [dummy_crop],
                top_k=1,
                query_preprocess=self.settings.aihub_classifier_query_preprocess,
            )

    def candidate_scope_summary(self, allowed_pill_ids: set[str] | None) -> dict:
        allowed = {str(pill_id).strip() for pill_id in allowed_pill_ids or set()}
        allowed = {pill_id for pill_id in allowed if pill_id}
        if not allowed:
            return {}

        positions_by_id = getattr(self.retriever, "index_positions_by_pill_id", {}) or {}
        class_ids_by_name = getattr(self.aihub_classifier, "class_ids_by_name", {}) or {}
        retrieval_matches = {pill_id for pill_id in allowed if pill_id in positions_by_id}
        classifier_matches = {pill_id for pill_id in allowed if pill_id in class_ids_by_name}
        position_count = sum(len(positions_by_id[pill_id]) for pill_id in retrieval_matches)
        metadata_matches = {pill_id for pill_id in allowed if pill_id in self.product_index}
        summary = {
            "enabled": True,
            "allowed_count": len(allowed),
            "metadata_match_count": len(metadata_matches),
            "unknown_metadata_pill_ids": sorted(allowed - metadata_matches),
        }
        if self.retriever is not None:
            summary.update(
                {
                    "retrieval_id_match_count": len(retrieval_matches),
                    "retrieval_index_position_count": position_count,
                    "unknown_retrieval_pill_ids": sorted(allowed - retrieval_matches),
                }
            )
        if self.aihub_classifier is not None:
            summary.update(
                {
                    "classifier_id_match_count": len(classifier_matches),
                    "unknown_classifier_pill_ids": sorted(allowed - classifier_matches),
                }
            )
        return summary


def recognize_crops_with_retriever(
    retriever,
    crops: list[np.ndarray],
    top_k: int,
    allowed_pill_ids: set[str] | None = None,
) -> list[list[ProductCandidate]]:
    if not crops:
        return []
    if retriever is None:
        return [[] for _ in crops]
    valid_pairs = [(index, crop) for index, crop in enumerate(crops) if crop.size]
    results = [[] for _ in crops]
    if not valid_pairs:
        return results
    predictions = retriever.predict_batch(
        [crop for _, crop in valid_pairs],
        top_k,
        allowed_pill_ids=allowed_pill_ids,
    )
    for (index, _), candidates in zip(valid_pairs, predictions):
        results[index] = candidates
    return results


def recognize_crops_with_aihub_classifier(
    classifier,
    crops: list[np.ndarray],
    top_k: int,
    query_preprocess: str = "multi_foreground",
    allowed_pill_ids: set[str] | None = None,
) -> list[list[ProductCandidate]]:
    if not crops:
        return []
    if classifier is None:
        return [[] for _ in crops]

    valid_pairs = [(index, crop) for index, crop in enumerate(crops) if crop.size]
    results = [[] for _ in crops]
    if not valid_pairs:
        return results

    crop_features = {index: estimate_crop_visual_features(crop) for index, crop in valid_pairs}
    modes = query_preprocess_modes(query_preprocess)
    variant_images = []
    variant_owners = []
    variant_modes = []
    for index, crop in valid_pairs:
        for mode in modes:
            variant_images.append(preprocess_query_crop(crop, mode))
            variant_owners.append(index)
            variant_modes.append(mode)

    predictions = classifier.predict_batch(
        variant_images,
        top_k=max(top_k * 8, 20),
        allowed_pill_ids=allowed_pill_ids,
    )
    by_crop: dict[int, dict[str, tuple[Candidate, str, float, list[str]]]] = {
        index: {} for index, _ in valid_pairs
    }
    for owner, mode, candidates in zip(variant_owners, variant_modes, predictions):
        features = crop_features.get(owner, CropVisualFeatures())
        for candidate in candidates:
            reranked_score, metadata_matches = apply_classifier_metadata_rerank(
                candidate.confidence,
                candidate,
                features,
            )
            current = by_crop[owner].get(candidate.class_name)
            if current is None or reranked_score > current[2]:
                by_crop[owner][candidate.class_name] = (
                    candidate,
                    mode,
                    reranked_score,
                    metadata_matches,
                )

    for index, _ in valid_pairs:
        ranked = sorted(
            by_crop[index].values(),
            key=lambda item: item[2],
            reverse=True,
        )[:top_k]
        results[index] = [
            product_candidate_from_aihub_candidate(rank, candidate, mode, score, matches)
            for rank, (candidate, mode, score, matches) in enumerate(ranked, start=1)
        ]
    return results


def product_candidate_from_aihub_candidate(
    rank: int,
    candidate: Candidate,
    mode: str,
    score: float | None = None,
    metadata_matches: list[str] | None = None,
) -> ProductCandidate:
    matched = f"AIHub ResNet152 classifier ({mode})"
    if metadata_matches:
        matched = f"{matched} + metadata rerank ({', '.join(metadata_matches)})"
    return ProductCandidate(
        rank=rank,
        pill_id=candidate.class_name,
        score=round(max(0.0, min(float(score if score is not None else candidate.confidence), 1.0)) * 100.0, 2),
        source="aihub_resnet152_classifier",
        product_name=candidate.product_name,
        ingredient=candidate.ingredient,
        caution_points=[],
        company=candidate.company,
        item_seq=candidate.item_seq,
        etc_otc_code=candidate.etc_otc_code,
        print_front=candidate.print_front,
        print_back=candidate.print_back,
        drug_shape=candidate.drug_shape,
        color_class1=candidate.color_class1,
        color_class2=candidate.color_class2,
        matched=matched,
        reference_image_url=product_reference_image_url(candidate.class_name),
    )


def apply_classifier_metadata_rerank(
    score: float,
    candidate: Candidate,
    features: CropVisualFeatures,
) -> tuple[float, list[str]]:
    adjusted = float(score)
    matches = []

    color_adjustment, color_detail = color_metadata_adjustment(candidate, features)
    adjusted += color_adjustment
    if color_detail:
        matches.append(color_detail)

    shape_adjustment, shape_detail = shape_metadata_adjustment(candidate, features)
    adjusted += shape_adjustment
    if shape_detail:
        matches.append(shape_detail)

    return adjusted, matches


def color_metadata_adjustment(
    candidate: Candidate,
    features: CropVisualFeatures,
) -> tuple[float, str | None]:
    feature_color = normalize_metadata_text(features.color)
    product_colors = {
        normalize_metadata_text(candidate.color_class1),
        normalize_metadata_text(candidate.color_class2),
    }
    product_colors.discard("")
    if not feature_color or not product_colors:
        return 0.0, None
    if feature_color in product_colors:
        return 0.03, f"color={features.color}"
    return 0.0, None


def shape_metadata_adjustment(
    candidate: Candidate,
    features: CropVisualFeatures,
) -> tuple[float, str | None]:
    feature_shape = normalize_metadata_text(features.shape)
    product_shape = normalize_metadata_text(candidate.drug_shape)
    if not feature_shape or not product_shape:
        return 0.0, None
    if feature_shape == product_shape:
        return 0.015, f"shape={features.shape}"
    if {feature_shape, product_shape} <= {"타원형", "장방형"}:
        return 0.005, f"shape~{features.shape}"
    return 0.0, None


def normalize_metadata_text(value: str | None) -> str:
    return str(value or "").strip()


def scope_warnings(warnings: list[str], candidate_scope: dict) -> list[str]:
    output = list(warnings)
    if scope_has_no_recognition_match(candidate_scope):
        output.append(
            "No allowed pill IDs are present in the active recognition index. "
            "Check the user's medication K-ID mapping."
        )
    return output


def scope_has_no_recognition_match(candidate_scope: dict) -> bool:
    if not candidate_scope.get("enabled"):
        return False
    retrieval_matches = candidate_scope.get("retrieval_id_match_count")
    classifier_matches = candidate_scope.get("classifier_id_match_count")
    known_counts = [
        count
        for count in (retrieval_matches, classifier_matches)
        if count is not None
    ]
    return bool(known_counts) and all(int(count) == 0 for count in known_counts)


def local_visual_observation(crop: np.ndarray) -> VisionObservation:
    features = estimate_crop_visual_features(crop)
    confidence = 0.35 if features.color or features.shape else 0.0
    return VisionObservation(
        shape=features.shape,
        color=features.color,
        confidence=confidence,
        notes="local crop visual features only; product identity comes from retrieval candidates.",
    )


def product_query_from_observation(
    observation: VisionObservation,
    limit: int,
) -> ProductSearchQuery:
    text_parts = []
    for candidate in observation.product_candidates:
        text_parts.append(candidate.product_name)
        if candidate.ingredient:
            text_parts.append(candidate.ingredient)
    if not text_parts:
        text_parts.extend(observation.possible_product_names)
    return ProductSearchQuery(
        imprint="",
        shape="",
        color="",
        text=" ".join(text_parts),
        limit=limit,
    )


def inspect_crops_safely(
    vision_provider,
    crops: list[np.ndarray],
) -> list[VisionObservation]:
    if not crops:
        return []
    valid_pairs = [(index, crop) for index, crop in enumerate(crops) if crop.size]
    valid_crops = [crop for _, crop in valid_pairs]
    if len(valid_crops) != len(crops):
        observations = [
            VisionObservation(notes="empty crop")
            for _ in crops
        ]
        for (index, _), observation in zip(
            valid_pairs,
            inspect_crops_safely(vision_provider, valid_crops),
        ):
            observations[index] = observation
        return observations
    try:
        if hasattr(vision_provider, "inspect_crops"):
            observations = vision_provider.inspect_crops(crops)
        else:
            observations = [vision_provider.inspect_crop(crop) for crop in crops]
        if len(observations) != len(crops):
            raise ValueError(
                f"expected {len(crops)} observations, got {len(observations)}"
            )
        return observations
    except Exception as error:
        return [
            VisionObservation(
                confidence=0.0,
                notes=f"{vision_provider.name} provider failed: {type(error).__name__}: {error}",
                raw={"provider": vision_provider.name, "error": str(error)},
            )
            for _ in crops
        ]


def rank_product_candidates(rows: list[dict], limit: int) -> list[ProductCandidate]:
    candidates = []
    for rank, row in enumerate(rows[:limit], start=1):
        candidates.append(
            ProductCandidate(
                rank=rank,
                pill_id=row["pill_id"],
                score=int(row.get("score", 0)),
                source="aihub_db",
                product_name=row.get("product_name"),
                ingredient=row.get("ingredient"),
                caution_points=[],
                company=row.get("company"),
                item_seq=row.get("item_seq"),
                etc_otc_code=row.get("etc_otc_code"),
                print_front=row.get("print_front"),
                print_back=row.get("print_back"),
                drug_shape=row.get("drug_shape"),
                color_class1=row.get("color_class1"),
                color_class2=row.get("color_class2"),
                matched=row.get("matched"),
                reference_image_url=product_reference_image_url(row.get("pill_id")),
            )
        )
    return candidates


def merge_llm_and_db_candidates(
    observation: VisionObservation,
    db_candidates: list[ProductCandidate],
    limit: int,
) -> list[ProductCandidate]:
    merged = llm_product_candidates(observation)
    seen_names = {normalize_name(candidate.product_name) for candidate in merged}
    for candidate in db_candidates:
        if normalize_name(candidate.product_name) in seen_names:
            continue
        merged.append(candidate)
    return [
        ProductCandidate(
            rank=rank,
            pill_id=candidate.pill_id,
            score=candidate.score,
            source=candidate.source,
            product_name=candidate.product_name,
            ingredient=candidate.ingredient,
            caution_points=candidate.caution_points,
            company=candidate.company,
            item_seq=candidate.item_seq,
            etc_otc_code=candidate.etc_otc_code,
            print_front=candidate.print_front,
            print_back=candidate.print_back,
            drug_shape=candidate.drug_shape,
            color_class1=candidate.color_class1,
            color_class2=candidate.color_class2,
            matched=candidate.matched,
            reference_image_url=candidate.reference_image_url,
        )
        for rank, candidate in enumerate(merged[:limit], start=1)
    ]


def llm_product_candidates(observation: VisionObservation) -> list[ProductCandidate]:
    if observation.product_candidates:
        return [
            llm_candidate_from_vision_candidate(index, candidate, observation.confidence)
            for index, candidate in enumerate(observation.product_candidates, start=1)
        ]

    confidence = observation.confidence if observation.confidence is not None else 0.5
    base_score = confidence_to_score(confidence)
    candidates = []
    seen = set()
    for index, product_name in enumerate(observation.possible_product_names, start=1):
        normalized = normalize_name(product_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            ProductCandidate(
                rank=index,
                pill_id="GEMINI",
                score=max(1, base_score - ((index - 1) * 5)),
                source="gemini",
                product_name=product_name,
                ingredient=None,
                caution_points=[],
                matched="Gemini visual recognition",
            )
        )
    return candidates


def llm_candidate_from_vision_candidate(
    index: int,
    candidate: VisionProductCandidate,
    fallback_confidence: float | None,
) -> ProductCandidate:
    confidence = (
        candidate.confidence
        if candidate.confidence is not None
        else fallback_confidence
        if fallback_confidence is not None
        else 0.5
    )
    return ProductCandidate(
        rank=index,
        pill_id="GEMINI",
        score=max(1, confidence_to_score(confidence) - ((index - 1) * 5)),
        source="gemini",
        product_name=candidate.product_name,
        ingredient=candidate.ingredient,
        caution_points=candidate.caution_points,
        matched="Gemini product/ingredient recognition",
    )


def confidence_to_score(confidence: float) -> int:
    return max(1, min(100, round(confidence * 100)))


def normalize_name(value: str | None) -> str:
    return "".join(str(value or "").split()).upper()


def determine_status(
    candidates: list[ProductCandidate],
    min_score: float = 70.0,
    ambiguity_margin: float = 3.0,
) -> tuple[str, str]:
    if not candidates:
        return "no_candidate", "No product candidate was found for this detected pill."

    top_score = float(candidates[0].score)
    if top_score < min_score:
        return (
            "low_confidence",
            f"Top candidate score {top_score:.2f} is below the review threshold {min_score:.2f}.",
        )

    if len(candidates) >= 2:
        second_score = float(candidates[1].score)
        margin = top_score - second_score
        if margin < ambiguity_margin:
            return (
                "ambiguous",
                f"Top-2 candidate score margin {margin:.2f} is below {ambiguity_margin:.2f}.",
            )

    return (
        "needs_confirmation",
        "Candidate is available, but final medication identity must be confirmed by the user.",
    )


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    padding_x = round((x2 - x1) * padding_ratio)
    padding_y = round((y2 - y1) * padding_ratio)
    return clamp_bbox(
        (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y),
        width,
        height,
    )


def clamp_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(int(x1), width - 1)),
        max(0, min(int(y1), height - 1)),
        max(1, min(int(x2), width)),
        max(1, min(int(y2), height)),
    )


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return np.clip(array, 0, 255).astype(np.uint8)


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)
