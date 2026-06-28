from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .evaluate_real_dataset import resolve_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a browser review report for real-smartphone annotations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/evaluation/real-smartphone"),
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real-review"))
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir or args.dataset_root / "images"
    annotations_dir = args.annotations_dir or args.dataset_root / "annotations"
    output_html = args.output_html or args.output_dir / "index.html"
    render_review_report(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=args.output_dir,
        output_html=output_html,
        pattern=args.pattern,
        limit=args.limit,
    )
    print(output_html)


def render_review_report(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    output_html: Path,
    pattern: str = "*.json",
    limit: int | None = None,
) -> dict[str, Any]:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(
        path for path in annotations_dir.glob(pattern) if path.is_file()
    )
    if limit is not None:
        annotation_paths = annotation_paths[:limit]

    rows = []
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(images_dir, annotation_path, payload)
        image = Image.open(image_path).convert("RGB")
        review_image_path, crop_rows = render_annotation_assets(
            image=image,
            annotation=payload,
            annotation_stem=annotation_path.stem,
            assets_dir=assets_dir,
        )
        for crop_row in crop_rows:
            crop_row["crop"] = crop_row["crop"].relative_to(output_html.parent)
        rows.append(
            {
                "image": image_path.name,
                "annotation": annotation_path.name,
                "review_image": review_image_path.relative_to(output_html.parent),
                "pills": crop_rows,
            }
        )

    output_html.write_text(build_html(rows), encoding="utf-8")
    return {"annotations": len(rows), "output_html": str(output_html)}


def render_annotation_assets(
    image: Image.Image,
    annotation: dict[str, Any],
    annotation_stem: str,
    assets_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    crop_rows = []
    for index, pill in enumerate(annotation.get("pills", []), start=1):
        bbox = clamp_bbox(image.size, pill.get("bbox_xyxy", [0, 0, 0, 0]))
        color = (28, 180, 80) if not pill.get("needs_review") else (235, 150, 30)
        draw.rectangle(bbox, outline=color, width=4)
        label = compact_label(index, pill)
        draw_label(draw, bbox, label, color, font)
        crop_path = assets_dir / f"{annotation_stem}_pill_{index:02d}.jpg"
        image.crop(bbox).save(crop_path, quality=92)
        crop_rows.append(
            {
                "index": index,
                "crop": crop_path,
                "class_name": pill.get("class_name") or "",
                "product_name": pill.get("product_name") or "",
                "needs_review": bool(pill.get("needs_review")),
                "candidates": candidate_hints(pill),
                "bbox": list(bbox),
            }
        )
    review_path = assets_dir / f"{annotation_stem}_review.jpg"
    canvas.save(review_path, quality=92)
    return review_path, crop_rows


def clamp_bbox(
    image_size: tuple[int, int],
    bbox: list[Any] | tuple[Any, ...],
) -> tuple[int, int, int, int]:
    width, height = image_size
    if len(bbox) != 4:
        raise ValueError(f"bbox_xyxy must contain 4 values: {bbox}")
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def compact_label(index: int, pill: dict[str, Any]) -> str:
    class_name = str(pill.get("class_name") or "?")
    product_name = str(pill.get("product_name") or "")
    if product_name:
        return f"{index}. {class_name} {product_name[:16]}"
    return f"{index}. {class_name}"


def draw_label(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, _, _ = bbox
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_y = max(0, y1 - text_height - 8)
    draw.rectangle(
        (x1, label_y, x1 + text_width + 8, label_y + text_height + 6),
        fill=color,
    )
    draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)


def candidate_hints(pill: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    hints = pill.get("candidate_hints") or []
    rows = []
    for candidate in hints[:limit]:
        rows.append(
            {
                "rank": candidate.get("rank"),
                "class_name": candidate.get("class_name"),
                "product_name": candidate.get("product_name"),
                "ingredient": candidate.get("ingredient"),
                "company": candidate.get("company"),
                "score": candidate.get("score"),
            }
        )
    return rows


def build_html(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(render_image_section(row) for row in rows)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pill Real Annotation Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f5f7fa; }}
    header {{ position: sticky; top: 0; padding: 16px 24px; background: #ffffff; border-bottom: 1px solid #d9e2ec; z-index: 2; }}
    h1 {{ margin: 0; font-size: 20px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    section {{ margin-bottom: 28px; padding: 18px; background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 1.2fr) minmax(320px, 1fr); gap: 18px; align-items: start; }}
    .review {{ width: 100%; height: auto; border: 1px solid #bcccdc; }}
    .pill {{ display: grid; grid-template-columns: 112px 1fr; gap: 12px; padding: 12px 0; border-top: 1px solid #e4e7eb; }}
    .pill:first-child {{ border-top: 0; }}
    .crop {{ width: 112px; height: 112px; object-fit: contain; background: #f0f4f8; border: 1px solid #bcccdc; }}
    .meta {{ font-size: 13px; line-height: 1.5; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; background: #fff3cd; color: #7c4700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12px; }}
    th, td {{ padding: 4px 6px; border: 1px solid #d9e2ec; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>Pill Real Annotation Review</h1></header>
  <main>
    {body}
  </main>
</body>
</html>
"""


def render_image_section(row: dict[str, Any]) -> str:
    pills = "\n".join(render_pill_row(pill) for pill in row["pills"])
    return f"""<section>
  <h2>{escape(row["image"])} <small>({escape(row["annotation"])})</small></h2>
  <div class="grid">
    <img class="review" src="{escape(str(row["review_image"]))}" alt="review image" />
    <div>{pills}</div>
  </div>
</section>"""


def render_pill_row(pill: dict[str, Any]) -> str:
    review_badge = '<span class="badge">review</span>' if pill["needs_review"] else ""
    candidates = render_candidates_table(pill["candidates"])
    return f"""<div class="pill">
  <img class="crop" src="{escape(str(pill["crop"]))}" alt="pill crop {pill["index"]}" />
  <div class="meta">
    <div><strong>#{pill["index"]}</strong> {review_badge}</div>
    <div>K-ID: <strong>{escape(pill["class_name"])}</strong></div>
    <div>제품명: {escape(pill["product_name"])}</div>
    <div>bbox: {escape(str(pill["bbox"]))}</div>
    {candidates}
  </div>
</div>"""


def render_candidates_table(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(candidate.get('rank'))}</td>"
        f"<td>{escape(candidate.get('class_name'))}</td>"
        f"<td>{escape(candidate.get('product_name'))}</td>"
        f"<td>{escape(candidate.get('ingredient'))}</td>"
        f"<td>{escape(candidate.get('score'))}</td>"
        "</tr>"
        for candidate in candidates
    )
    return f"""<table>
  <thead><tr><th>Rank</th><th>K-ID</th><th>제품명</th><th>성분</th><th>Score</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


if __name__ == "__main__":
    main()
