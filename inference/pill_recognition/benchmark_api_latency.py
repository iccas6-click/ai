from __future__ import annotations

import argparse
import json
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from PIL import Image, ImageDraw


def parse_crop_counts(value: str) -> list[int]:
    counts = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        count = int(chunk)
        if count <= 0:
            raise ValueError("crop counts must be positive")
        counts.append(count)
    if not counts:
        raise ValueError("at least one crop count is required")
    return counts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["elapsed_ms", "api_total_ms", "pipeline_call_ms", "recognition_ms"]
    summary: dict[str, Any] = {"count": len(runs)}
    for field in fields:
        values = [
            float(run[field])
            for run in runs
            if isinstance(run.get(field), int | float)
        ]
        if not values:
            continue
        summary[field] = {
            "min": round(min(values), 2),
            "p50": round(percentile(values, 0.50), 2),
            "p95": round(percentile(values, 0.95), 2),
            "max": round(max(values), 2),
        }
    return summary


def build_generated_crop_bytes(size: int = 160) -> bytes:
    image = Image.new("RGB", (size, size), (238, 236, 230))
    draw = ImageDraw.Draw(image)
    margin_x = int(size * 0.20)
    margin_y = int(size * 0.32)
    draw.ellipse(
        (margin_x, margin_y, size - margin_x, size - margin_y),
        fill=(245, 247, 249),
        outline=(185, 188, 192),
        width=max(2, size // 80),
    )
    draw.line(
        (int(size * 0.50), margin_y + 8, int(size * 0.50), size - margin_y - 8),
        fill=(205, 208, 212),
        width=max(2, size // 90),
    )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def load_crop_bytes(path: Path | None) -> bytes:
    if path is None:
        return build_generated_crop_bytes()
    return path.read_bytes()


def make_multipart_body(
    field_name: str,
    files: list[tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----pill-benchmark-{uuid.uuid4().hex}"
    body = bytearray()
    for filename, payload, content_type in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_multipart(
    base_url: str,
    endpoint: str,
    field_name: str,
    files: list[tuple[str, bytes, str]],
    timeout: float,
) -> dict[str, Any]:
    body, content_type = make_multipart_body(field_name, files)
    url = f"{base_url.rstrip('/')}{endpoint}"
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            status_code = response.status
    except HTTPError as exc:
        payload = exc.read()
        status_code = exc.code
    except URLError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": False,
            "status_code": None,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": str(exc.reason),
        }
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        response_json = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        response_json = {"raw": payload.decode("utf-8", errors="replace")}
    timings = response_json.get("timings_ms", {}) if isinstance(response_json, dict) else {}
    return {
        "ok": 200 <= status_code < 300,
        "status_code": status_code,
        "elapsed_ms": round(elapsed_ms, 2),
        "api_total_ms": timings.get("api_total"),
        "pipeline_call_ms": timings.get("pipeline_call"),
        "recognition_ms": timings.get("recognition"),
        "pill_count": response_json.get("pill_count")
        if isinstance(response_json, dict)
        else None,
        "error": None if 200 <= status_code < 300 else response_json,
    }


def run_crop_once(base_url: str, crop_bytes: bytes, timeout: float) -> dict[str, Any]:
    return post_multipart(
        base_url=base_url,
        endpoint="/crops/recognize",
        field_name="file",
        files=[("crop.jpg", crop_bytes, "image/jpeg")],
        timeout=timeout,
    )


def run_batch_once(
    base_url: str,
    crop_bytes: bytes,
    crop_count: int,
    timeout: float,
) -> dict[str, Any]:
    files = [
        (f"crop-{index + 1}.jpg", crop_bytes, "image/jpeg")
        for index in range(crop_count)
    ]
    return post_multipart(
        base_url=base_url,
        endpoint="/crops/recognize-batch",
        field_name="files",
        files=files,
        timeout=timeout,
    )


def benchmark_endpoint(
    name: str,
    crop_count: int,
    iterations: int,
    warmup: int,
    call_once,
) -> dict[str, Any]:
    for _ in range(warmup):
        call_once()
    runs = [call_once() for _ in range(iterations)]
    return {
        "endpoint": name,
        "crop_count": crop_count,
        "summary": summarize_runs([run for run in runs if run.get("ok")]),
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CLICK pill recognition API crop latency."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--crop-counts", default="1,3,6,12")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--mode",
        choices=["crop", "batch", "both"],
        default="both",
        help="crop benchmarks /crops/recognize once; batch benchmarks /crops/recognize-batch.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_counts = parse_crop_counts(args.crop_counts)
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")

    crop_bytes = load_crop_bytes(args.image)
    results = []
    if args.mode in {"crop", "both"}:
        results.append(
            benchmark_endpoint(
                name="/crops/recognize",
                crop_count=1,
                iterations=args.iterations,
                warmup=args.warmup,
                call_once=lambda: run_crop_once(args.base_url, crop_bytes, args.timeout),
            )
        )
    if args.mode in {"batch", "both"}:
        for crop_count in crop_counts:
            results.append(
                benchmark_endpoint(
                    name="/crops/recognize-batch",
                    crop_count=crop_count,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    call_once=lambda count=crop_count: run_batch_once(
                        args.base_url,
                        crop_bytes,
                        count,
                        args.timeout,
                    ),
                )
            )

    payload = {
        "base_url": args.base_url,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "crop_counts": crop_counts,
        "image": str(args.image) if args.image else "generated",
        "results": results,
    }
    for result in results:
        elapsed = result["summary"].get("elapsed_ms", {})
        api_total = result["summary"].get("api_total_ms", {})
        print(
            f"{result['endpoint']} crops={result['crop_count']} "
            f"elapsed_p50={elapsed.get('p50', 'n/a')}ms "
            f"elapsed_p95={elapsed.get('p95', 'n/a')}ms "
            f"api_p50={api_total.get('p50', 'n/a')}ms"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
