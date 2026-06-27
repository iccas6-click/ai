from io import BytesIO

from PIL import Image

from pill_recognition.benchmark_api_latency import (
    allowed_scope_fields,
    build_generated_crop_bytes,
    make_multipart_body,
    parse_crop_counts,
    summarize_runs,
)


def test_parse_crop_counts_accepts_comma_separated_values():
    assert parse_crop_counts("1, 3,6,12") == [1, 3, 6, 12]


def test_summarize_runs_reports_latency_percentiles():
    summary = summarize_runs(
        [
            {"ok": True, "elapsed_ms": 10, "api_total_ms": 8, "recognition_ms": 5},
            {"ok": True, "elapsed_ms": 20, "api_total_ms": 16, "recognition_ms": 10},
            {"ok": True, "elapsed_ms": 30, "api_total_ms": 24, "recognition_ms": 15},
        ]
    )

    assert summary["count"] == 3
    assert summary["elapsed_ms"]["min"] == 10
    assert summary["elapsed_ms"]["p50"] == 20
    assert summary["elapsed_ms"]["p95"] == 29.0
    assert summary["api_total_ms"]["p50"] == 16
    assert summary["recognition_ms"]["max"] == 15


def test_build_generated_crop_bytes_returns_readable_jpeg():
    payload = build_generated_crop_bytes(size=96)

    image = Image.open(BytesIO(payload))

    assert image.size == (96, 96)
    assert image.mode == "RGB"
    assert image.format == "JPEG"


def test_make_multipart_body_contains_all_files():
    body, content_type = make_multipart_body(
        "files",
        [
            ("a.jpg", b"aaa", "image/jpeg"),
            ("b.jpg", b"bbb", "image/jpeg"),
        ],
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    assert body.count(b'name="files"') == 2
    assert b'filename="a.jpg"' in body
    assert b'filename="b.jpg"' in body


def test_make_multipart_body_can_include_allowed_pill_scope_field():
    body, _ = make_multipart_body(
        "file",
        [("crop.jpg", b"aaa", "image/jpeg")],
        fields=[("allowed_pill_ids", '["K-000001","K-000002"]')],
    )

    assert body.count(b'name="allowed_pill_ids"') == 1
    assert b'["K-000001","K-000002"]' in body
    assert b'filename="crop.jpg"' in body


def test_allowed_scope_fields_omits_empty_scope():
    assert allowed_scope_fields("") == []
    assert allowed_scope_fields("  ") == []
    assert allowed_scope_fields("K-000001") == [("allowed_pill_ids", "K-000001")]
