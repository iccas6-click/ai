import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pill_recognition.api import (
    create_app,
    detect_image_media_type,
)
from pill_recognition.schemas import (
    PillDetection,
    ProductCandidate,
    RecognitionResult,
    VisionObservation,
)
from pill_recognition.settings import Settings
from pill_recognition.scope import parse_allowed_pill_ids
from pill_recognition_legacy.aihub_classifier import AIHubProductInfo


class FakePipeline:
    warmup_calls = 0
    last_allowed_pill_ids = None

    def warmup(self, load_detector=True):
        self.warmup_calls += 1

    def recognize(self, image_rgb, allowed_pill_ids=None):
        assert image_rgb.shape == (12, 16, 3)
        self.last_allowed_pill_ids = allowed_pill_ids
        return fake_result("fake", "needs_confirmation")

    def recognize_crop(self, image_rgb, allowed_pill_ids=None):
        assert image_rgb.shape == (12, 16, 3)
        self.last_allowed_pill_ids = allowed_pill_ids
        return fake_result("fake-crop", "needs_confirmation")

    def recognize_crops_batch(self, images_rgb, allowed_pill_ids=None):
        assert [image.shape for image in images_rgb] == [(12, 16, 3), (10, 8, 3)]
        self.last_allowed_pill_ids = allowed_pill_ids
        return RecognitionResult(
            image_width=16,
            image_height=12,
            pill_count=2,
            model_version="fake-crop-batch",
            detections=[
                fake_detection(1, "K-000001"),
                fake_detection(2, "K-000002"),
            ],
        )


class FailingWarmupPipeline(FakePipeline):
    def warmup(self, load_detector=True):
        raise RuntimeError("warmup boom")


def fake_result(model_version: str, status: str) -> RecognitionResult:
    return RecognitionResult(
        image_width=16,
        image_height=12,
        pill_count=1,
        model_version=model_version,
        detections=[
            PillDetection(
                pill_id=1,
                bbox=(1, 2, 10, 11),
                crop_bbox=(0, 1, 11, 12),
                detector_confidence=0.92,
                vision=VisionObservation(),
                candidates=[
                    ProductCandidate(
                        rank=1,
                        pill_id="K-000001",
                        score=88.0,
                        product_name="테스트정",
                        ingredient="성분A",
                    )
                ],
                status=status,
                status_reason="review required",
            )
        ],
    )


def fake_detection(pill_id: int, candidate_id: str) -> PillDetection:
    return PillDetection(
        pill_id=pill_id,
        bbox=(1, 2, 10, 11),
        crop_bbox=(0, 1, 11, 12),
        detector_confidence=0.92,
        vision=VisionObservation(),
        candidates=[
            ProductCandidate(
                rank=1,
                pill_id=candidate_id,
                score=88.0,
                product_name="테스트정",
                ingredient="성분A",
            )
        ],
        status="needs_confirmation",
        status_reason="review required",
    )


def test_health_returns_runtime_policy(monkeypatch):
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["recognizer"] == "retrieval"
    assert response.json()["recognition_policy"] == (
        "rtmdet_single_class_detector + aihub_resnet_retrieval_top_k"
    )
    assert response.json()["external_vision_default"] is False
    assert response.json()["experimental_gemini_enabled"] is False
    assert response.json()["top_k"] == 3
    assert response.json()["max_batch_crops"] == 12
    assert response.json()["max_upload_bytes"] == 10 * 1024 * 1024
    assert response.json()["max_image_pixels"] == 12_000_000
    assert response.json()["retrieval_query_preprocess"] == "none"
    assert response.json()["warmup"]["enabled"] is True


def test_startup_warmup_marks_health_ok():
    pipeline = FakePipeline()
    app = create_app(lambda: pipeline)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    warmup = response.json()["warmup"]
    assert warmup["status"] == "ok"
    assert warmup["duration_ms"] >= 0
    assert warmup["error"] is None
    assert pipeline.warmup_calls == 1


def test_startup_warmup_failure_is_reported_without_crashing():
    app = create_app(lambda: FailingWarmupPipeline())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    warmup = response.json()["warmup"]
    assert warmup["status"] == "failed"
    assert "RuntimeError" in warmup["error"]


def test_recognize_accepts_uploaded_image():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("pill.jpg", image_bytes(16, 12), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pill_count"] == 1
    assert payload["detections"][0]["status"] == "needs_confirmation"
    assert payload["detections"][0]["status_reason"] == "review required"
    assert payload["detections"][0]["candidates"][0]["pill_id"] == "K-000001"
    assert_api_timings(payload["timings_ms"])


def test_recognize_passes_allowed_pill_scope_to_pipeline():
    pipeline = FakePipeline()
    app = create_app(lambda: pipeline)
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("pill.jpg", image_bytes(16, 12), "image/jpeg")},
        data={"allowed_pill_ids": "[\"K-000001\", \"K-000002\", \"K-000003\"]"},
    )

    assert response.status_code == 200
    assert pipeline.last_allowed_pill_ids == {"K-000001", "K-000002", "K-000003"}


def test_recognize_resolves_item_seq_and_product_name_scope_to_pipeline():
    pipeline = FakePipeline()
    app = create_app(
        lambda: pipeline,
        product_index_factory=lambda: {
            "K-WARFARIN": AIHubProductInfo(
                pill_id="K-WARFARIN",
                product_name="대화와르파린나트륨정",
                item_seq="198601052",
            ),
            "K-TYLENOL": AIHubProductInfo(
                pill_id="K-TYLENOL",
                product_name="타이레놀정500mg",
                item_seq="199303108",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("pill.jpg", image_bytes(16, 12), "image/jpeg")},
        data={
            "allowed_item_seqs": "[\"198601052\"]",
            "allowed_product_names": "타이레놀정500mg",
        },
    )

    assert response.status_code == 200
    assert pipeline.last_allowed_pill_ids == {"K-WARFARIN", "K-TYLENOL"}
    payload = response.json()
    assert payload["candidate_scope"]["input_scope_resolution"] == {
        "input_item_seq_count": 1,
        "input_product_name_count": 1,
        "resolved_count": 2,
        "unresolved": [],
    }


def test_recognize_reports_missing_metadata_when_named_scope_is_requested():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("pill.jpg", image_bytes(16, 12), "image/jpeg")},
        data={"allowed_item_seqs": "198601052"},
    )

    assert response.status_code == 503


def test_parse_allowed_pill_ids_accepts_json_commas_and_whitespace():
    assert parse_allowed_pill_ids(
        ["K-000001,K-000002", "K-000003 K-000004", "[\"K-000005\"]"]
    ) == {"K-000001", "K-000002", "K-000003", "K-000004", "K-000005"}


def test_recognize_crop_accepts_uploaded_single_pill_crop():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/crops/recognize",
        files={"file": ("pill-crop.jpg", image_bytes(16, 12), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "fake-crop"
    assert payload["pill_count"] == 1
    assert payload["detections"][0]["bbox"] == [1, 2, 10, 11]
    assert payload["detections"][0]["candidates"][0]["product_name"] == "테스트정"
    assert_api_timings(payload["timings_ms"])


def test_recognize_crop_batch_accepts_multiple_uploaded_crops():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/crops/recognize-batch",
        files=[
            ("files", ("front.jpg", image_bytes(16, 12), "image/jpeg")),
            ("files", ("back.jpg", image_bytes(8, 10), "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "fake-crop-batch"
    assert payload["pill_count"] == 2
    assert [row["candidates"][0]["pill_id"] for row in payload["detections"]] == [
        "K-000001",
        "K-000002",
    ]
    assert_api_timings(payload["timings_ms"])


def test_recognize_crop_batch_passes_allowed_pill_scope_to_pipeline():
    pipeline = FakePipeline()
    app = create_app(lambda: pipeline)
    client = TestClient(app)

    response = client.post(
        "/crops/recognize-batch",
        files=[
            ("files", ("front.jpg", image_bytes(16, 12), "image/jpeg")),
            ("files", ("back.jpg", image_bytes(8, 10), "image/jpeg")),
        ],
        data={"allowed_pill_ids": "K-USER K-OTHER"},
    )

    assert response.status_code == 200
    assert pipeline.last_allowed_pill_ids == {"K-USER", "K-OTHER"}


def test_recognize_crop_batch_rejects_too_many_files(monkeypatch):
    monkeypatch.setattr(
        "pill_recognition.api.get_settings",
        lambda: Settings(max_batch_crops=1),
    )
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/crops/recognize-batch",
        files=[
            ("files", ("front.jpg", image_bytes(16, 12), "image/jpeg")),
            ("files", ("back.jpg", image_bytes(8, 10), "image/jpeg")),
        ],
    )

    assert response.status_code == 400
    assert "Too many crop images" in response.json()["detail"]


def test_recognize_rejects_non_image_file():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_recognize_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(
        "pill_recognition.api.get_settings",
        lambda: Settings(max_upload_bytes=5),
    )
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("large.jpg", image_bytes(16, 12), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "too large" in response.json()["detail"]


def test_recognize_rejects_image_with_too_many_pixels(monkeypatch):
    monkeypatch.setattr(
        "pill_recognition.api.get_settings",
        lambda: Settings(max_image_pixels=100),
    )
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("too-many-pixels.jpg", image_bytes(16, 12), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "too many pixels" in response.json()["detail"]


def test_product_search_returns_aihub_metadata_matches():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-000001": AIHubProductInfo(
                pill_id="K-000001",
                product_name="대화와르파린나트륨정",
                ingredient="와르파린나트륨",
                print_front="W분할선2",
                drug_shape="원형",
                color_class1="하양",
            )
        },
    )
    client = TestClient(app)

    response = client.get(
        "/products/search",
        params={
            "imprint": "W2",
            "shape": "원형",
            "color": "하양",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["query"]["limit"] == 5
    assert payload["results"][0]["pill_id"] == "K-000001"
    assert payload["results"][0]["ingredient"] == "와르파린나트륨"
    assert payload["results"][0]["matched"] == "각인 exact, 모양, 색"
    assert payload["results"][0]["reference_image_url"] == (
        "/products/K-000001/reference-image"
    )


def test_product_reference_image_endpoint_returns_aihub_crop(monkeypatch, tmp_path):
    crop_root = tmp_path / "pill_data_croped"
    product_dir = crop_root / "K-000001"
    product_dir.mkdir(parents=True)
    Image.new("RGB", (12, 12), "white").save(product_dir / "sample.png")
    mapping_path = crop_root / "pill_label_path_sharp_score.json"
    mapping_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "pill_recognition.api.get_settings",
        lambda: Settings(aihub_mapping=mapping_path, warmup_on_startup=False),
    )
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.get("/products/K-000001/reference-image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content


def test_product_detail_returns_aihub_metadata():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-000001": AIHubProductInfo(
                pill_id="K-000001",
                product_name="대화와르파린나트륨정",
                ingredient="와르파린나트륨",
                print_front="W분할선2",
                drug_shape="원형",
                color_class1="하양",
                company="대화제약",
            )
        },
    )
    client = TestClient(app)

    response = client.get("/products/K-000001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["pill_id"] == "K-000001"
    assert payload["product_name"] == "대화와르파린나트륨정"
    assert payload["ingredient"] == "와르파린나트륨"
    assert payload["reference_image_url"] == "/products/K-000001/reference-image"


def test_product_detail_reports_unknown_product():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-000001": AIHubProductInfo(pill_id="K-000001", product_name="후보")
        },
    )
    client = TestClient(app)

    response = client.get("/products/K-404")

    assert response.status_code == 404


def test_product_detail_reports_missing_metadata():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.get("/products/K-000001")

    assert response.status_code == 503


def test_product_reference_image_endpoint_rejects_unknown_id(monkeypatch, tmp_path):
    mapping_path = tmp_path / "pill_label_path_sharp_score.json"
    mapping_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "pill_recognition.api.get_settings",
        lambda: Settings(aihub_mapping=mapping_path, warmup_on_startup=False),
    )
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.get("/products/not-a-pill/reference-image")

    assert response.status_code == 404


def test_detect_image_media_type_uses_file_content_not_suffix(tmp_path):
    image_path = tmp_path / "looks-like.png"
    Image.new("RGB", (12, 12), "white").save(image_path, format="JPEG")

    assert detect_image_media_type(image_path) == "image/jpeg"


def test_product_search_requires_at_least_one_query_field():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.get("/products/search")

    assert response.status_code == 400


def test_product_search_reports_missing_product_metadata():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.get("/products/search", params={"text": "아시클로버"})

    assert response.status_code == 503


def test_product_scope_resolve_accepts_pill_ids_item_seqs_and_product_names():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-WARFARIN": AIHubProductInfo(
                pill_id="K-WARFARIN",
                product_name="대화와르파린나트륨정",
                ingredient="와르파린나트륨",
                item_seq="196400046",
            ),
            "K-ACYCLOVIR": AIHubProductInfo(
                pill_id="K-ACYCLOVIR",
                product_name="진양아시클로버정",
                ingredient="아시클로버",
                item_seq="200000001",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/scope/resolve",
        json={
            "pill_ids": ["K-WARFARIN"],
            "item_seqs": ["200000001"],
            "product_names": ["대화와르파린나트륨정"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed_pill_ids"] == ["K-WARFARIN", "K-ACYCLOVIR"]
    assert payload["count"] == 2
    assert payload["unresolved"] == []
    assert [row["input_type"] for row in payload["resolved"]] == [
        "pill_id",
        "item_seq",
        "product_name",
    ]
    assert payload["resolved"][0]["matches"][0]["ingredient"] == "와르파린나트륨"
    assert payload["resolved"][1]["matches"][0]["match_type"] == "item_seq_exact"
    assert payload["resolved"][2]["matches"][0]["reference_image_url"] == (
        "/products/K-WARFARIN/reference-image"
    )


def test_product_scope_resolve_limits_partial_product_name_matches_and_reports_unknowns():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-ONE": AIHubProductInfo(
                pill_id="K-ONE",
                product_name="테스트정A",
                item_seq="1",
            ),
            "K-TWO": AIHubProductInfo(
                pill_id="K-TWO",
                product_name="테스트정B",
                item_seq="2",
            ),
            "K-THREE": AIHubProductInfo(
                pill_id="K-THREE",
                product_name="테스트정C",
                item_seq="3",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/scope/resolve",
        json={
            "pill_ids": ["K-MISSING"],
            "product_names": ["테스트"],
            "limit_per_query": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed_pill_ids"] == ["K-ONE", "K-TWO"]
    assert payload["count"] == 2
    assert payload["limit_per_query"] == 2
    assert payload["resolved"][0]["match_type"] == "product_name_partial"
    assert payload["resolved"][0]["count"] == 2
    assert payload["unresolved"] == [{"input_type": "pill_id", "input": "K-MISSING"}]


def test_product_scope_resolve_requires_at_least_one_identifier():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.post("/products/scope/resolve", json={})

    assert response.status_code == 400


def test_product_scope_resolve_reports_missing_product_metadata():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.post(
        "/products/scope/resolve",
        json={"item_seqs": ["196400046"]},
    )

    assert response.status_code == 503


def test_product_refine_combines_image_candidates_with_metadata_search():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-WARFARIN": AIHubProductInfo(
                pill_id="K-WARFARIN",
                product_name="대화와르파린나트륨정",
                ingredient="와르파린나트륨",
                print_front="W분할선2",
                drug_shape="원형",
                color_class1="하양",
            ),
            "K-OTHER": AIHubProductInfo(
                pill_id="K-OTHER",
                product_name="다른정",
                ingredient="다른성분",
                print_front="AB",
                drug_shape="장방형",
                color_class1="노랑",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {"pill_id": "K-OTHER", "score": 95.0, "source": "retrieval"},
                {"pill_id": "K-WARFARIN", "score": 55.0, "source": "retrieval"},
            ],
            "imprint": "W2",
            "shape": "원형",
            "color": "하양",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["status"] == "needs_confirmation"
    assert payload["status_reason"]
    assert payload["results"][0]["pill_id"] == "K-WARFARIN"
    assert payload["results"][0]["image_score"] == 55.0
    assert payload["results"][0]["metadata_score"] == 170.0
    assert payload["results"][0]["score"] == 225.0
    assert payload["results"][0]["matched"] == "image candidate + 각인 exact, 모양, 색"
    assert payload["results"][0]["reference_image_url"] == (
        "/products/K-WARFARIN/reference-image"
    )


def test_product_refine_can_rank_existing_candidates_without_query():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-LOW": AIHubProductInfo(pill_id="K-LOW", product_name="낮은후보"),
            "K-HIGH": AIHubProductInfo(pill_id="K-HIGH", product_name="높은후보"),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {"pill_id": "K-LOW", "score": 30},
                {"pill_id": "K-HIGH", "score": 88},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["pill_id"] == "K-HIGH"


def test_product_refine_can_limit_results_to_user_medication_scope():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-USER": AIHubProductInfo(
                pill_id="K-USER",
                product_name="사용자복약정",
                ingredient="성분A",
                print_front="U1",
            ),
            "K-NOT-USER": AIHubProductInfo(
                pill_id="K-NOT-USER",
                product_name="복약목록밖정",
                ingredient="성분B",
                print_front="N1",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "allowed_pill_ids": ["K-USER", "K-MISSING"],
            "candidates": [
                {"pill_id": "K-NOT-USER", "score": 99},
                {"pill_id": "K-USER", "score": 62},
            ],
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_scope"] == {
        "enabled": True,
        "allowed_count": 2,
        "matched_count": 1,
        "unknown_pill_ids": ["K-MISSING"],
    }
    assert payload["count"] == 1
    assert payload["results"][0]["pill_id"] == "K-USER"
    assert payload["results"][0]["image_score"] == 62.0


def test_product_refine_searches_only_user_medication_scope_when_query_is_present():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-USER": AIHubProductInfo(
                pill_id="K-USER",
                product_name="사용자복약정",
                ingredient="성분A",
                print_front="U1",
            ),
            "K-NOT-USER": AIHubProductInfo(
                pill_id="K-NOT-USER",
                product_name="복약목록밖정",
                ingredient="성분B",
                print_front="U1",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "allowed_pill_ids": ["K-USER"],
            "imprint": "U1",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["pill_id"] == "K-USER"
    assert payload["results"][0]["source"] == "aihub_metadata_search"


def test_product_refine_boosts_candidate_seen_in_multiple_views():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-SINGLE": AIHubProductInfo(pill_id="K-SINGLE", product_name="한면후보"),
            "K-MULTI": AIHubProductInfo(pill_id="K-MULTI", product_name="양면후보"),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {
                    "pill_id": "K-SINGLE",
                    "score": 91,
                    "source": "aihub_resnet_retrieval",
                    "view": "front",
                },
                {
                    "pill_id": "K-MULTI",
                    "score": 88,
                    "source": "aihub_resnet_retrieval",
                    "view": "front",
                },
                {
                    "pill_id": "K-MULTI",
                    "score": 74,
                    "source": "aihub_resnet_retrieval",
                    "view": "back",
                },
            ],
            "limit": 2,
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["pill_id"] == "K-MULTI"
    assert result["image_score"] == 93.0
    assert result["image_score_max"] == 88.0
    assert result["image_evidence_count"] == 2
    assert result["views"] == ["back", "front"]
    assert result["candidate_sources"] == ["aihub_resnet_retrieval"]
    assert result["matched"] == "image candidate x2"


def test_product_refine_marks_ambiguous_when_scores_are_close():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-ONE": AIHubProductInfo(pill_id="K-ONE", product_name="후보1"),
            "K-TWO": AIHubProductInfo(pill_id="K-TWO", product_name="후보2"),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {"pill_id": "K-ONE", "score": 88},
                {"pill_id": "K-TWO", "score": 86},
            ],
            "limit": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ambiguous"
    assert "Top-2" in response.json()["status_reason"]


def test_product_refine_marks_no_candidate_when_all_candidate_ids_are_unknown():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-KNOWN": AIHubProductInfo(pill_id="K-KNOWN", product_name="후보")
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={"candidates": [{"pill_id": "K-UNKNOWN", "score": 99}]},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 0
    assert response.json()["status"] == "no_candidate"


def test_product_refine_requires_candidates_or_query():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.post("/products/refine", json={})

    assert response.status_code == 400


def image_bytes(width: int, height: int) -> bytes:
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def assert_api_timings(timings: dict) -> None:
    assert set(timings) == {
        "upload_decode",
        "pipeline_get",
        "pipeline_call",
        "api_total",
    }
    assert timings["api_total"] >= timings["pipeline_call"]
