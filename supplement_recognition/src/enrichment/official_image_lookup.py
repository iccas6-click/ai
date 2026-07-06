from __future__ import annotations

import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


@dataclass(frozen=True)
class OfficialImageResult:
    image_url: str
    source_url: str | None = None


def lookup_official_product_image(
    product_name: str,
    manufacturer: str | None = None,
) -> OfficialImageResult | None:
    """Find a verified official product image URL using Gemini Search grounding.

    The current MFDS health supplement dataset does not expose package images.
    This optional enrichment step asks Gemini to search the web, then accepts
    only direct public image URLs that the model marks as official.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    payload = {
        "model": os.environ.get("SUPPLEMENT_IMAGE_LOOKUP_MODEL", "gemini-3.5-flash"),
        "input": _build_prompt(product_name, manufacturer),
        "tools": [{"type": "google_search"}],
    }
    request = urllib.request.Request(
        _INTERACTIONS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None

    candidate = _parse_candidate(_extract_output_text(raw))
    if candidate is None:
        return None

    image_url = _normalize_url(candidate.get("image_url"))
    source_url = _normalize_url(candidate.get("source_url"))
    if not image_url or not bool(candidate.get("is_official")):
        return None
    if not _is_public_http_url(image_url):
        return None
    if source_url and not _is_public_http_url(source_url):
        return None
    if not _looks_like_image(image_url):
        return None

    return OfficialImageResult(image_url=image_url, source_url=source_url)


def _build_prompt(product_name: str, manufacturer: str | None) -> str:
    maker = (manufacturer or "").strip() or "제조사 미상"
    return f"""\
한국 건강기능식품 제품의 공식 제품 이미지 URL을 찾아줘.

제품명: {product_name}
제조사: {maker}

반드시 아래 JSON 한 줄만 반환해.
{{"image_url": string|null, "source_url": string|null, "is_official": boolean, "reason": string}}

규칙:
- image_url은 jpg, jpeg, png, webp 같은 실제 이미지 파일을 직접 가리키는 공개 URL이어야 한다.
- 제조사 공식몰, 브랜드 공식 페이지, 공공기관처럼 공식 출처라고 판단되는 경우만 is_official=true.
- 쇼핑몰, 블로그, 커뮤니티, 뉴스, 임의 이미지 검색 결과면 image_url=null, is_official=false.
- 확실하지 않으면 image_url=null, is_official=false.
"""


def _extract_output_text(response: dict) -> str:
    direct = response.get("output_text") or response.get("outputText")
    if isinstance(direct, str):
        return direct

    chunks: list[str] = []
    for step in response.get("steps", []) or []:
        if step.get("type") != "model_output":
            continue
        for item in step.get("content", []) or []:
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _parse_candidate(text: str) -> dict | None:
    if not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _normalize_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return raw


def _is_public_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def _looks_like_image(url: str) -> bool:
    request = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-2047"})
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            content_type = response.headers.get("Content-Type", "").lower()
    except (OSError, urllib.error.URLError):
        return False
    return content_type.startswith("image/")


def _timeout() -> float:
    try:
        return float(os.environ.get("SUPPLEMENT_IMAGE_LOOKUP_TIMEOUT", "12"))
    except ValueError:
        return 12.0
