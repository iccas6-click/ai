import json

from PIL import Image

from pill_recognition.render_real_annotation_review import (
    candidate_hints,
    clamp_bbox,
    render_review_report,
)


def test_clamp_bbox_keeps_box_inside_image():
    assert clamp_bbox((100, 80), [-10, 2.4, 140, 200]) == (0, 2, 100, 80)


def test_candidate_hints_limits_review_rows():
    pill = {
        "candidate_hints": [
            {"rank": 1, "class_name": "K-000001", "product_name": "A"},
            {"rank": 2, "class_name": "K-000002", "product_name": "B"},
            {"rank": 3, "class_name": "K-000003", "product_name": "C"},
            {"rank": 4, "class_name": "K-000004", "product_name": "D"},
        ]
    }

    rows = candidate_hints(pill)

    assert [row["class_name"] for row in rows] == [
        "K-000001",
        "K-000002",
        "K-000003",
    ]


def test_render_review_report_writes_html_and_assets(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    output = tmp_path / "review"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (80, 60), "white").save(images / "photo.jpg")
    (annotations / "photo.json").write_text(
        json.dumps(
            {
                "image": "photo.jpg",
                "pills": [
                    {
                        "class_name": "K-000001",
                        "product_name": "테스트정",
                        "bbox_xyxy": [10, 12, 50, 44],
                        "needs_review": True,
                        "candidate_hints": [
                            {
                                "rank": 1,
                                "class_name": "K-000001",
                                "product_name": "테스트정",
                                "ingredient": "성분A",
                                "score": 88,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = render_review_report(
        images_dir=images,
        annotations_dir=annotations,
        output_dir=output,
        output_html=output / "index.html",
    )

    assert result["annotations"] == 1
    assert (output / "index.html").exists()
    assert (output / "assets" / "photo_review.jpg").exists()
    assert (output / "assets" / "photo_pill_01.jpg").exists()
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "K-000001" in html
    assert "assets/photo_pill_01.jpg" in html
