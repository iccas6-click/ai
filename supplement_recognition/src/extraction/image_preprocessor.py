"""
Gemini 전송 전 이미지 전처리.

실제 사용자가 찍은 사진은 배경 노이즈, 흐림, 노출 문제가 있어서
Gemini가 제품 라벨 텍스트에 집중하기 어려울 수 있음.

처리 순서:
1. 크기 정규화 (너무 크면 리사이즈, 너무 작으면 업스케일)
2. 노이즈 제거 (약한 가우시안 블러)
3. 대비/선명도 강화
4. 라벨 영역 자동 크롭 (엣지 기반)
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

# Gemini에 보낼 최적 해상도 범위
_MIN_SIZE = 512
_MAX_SIZE = 1024


def _normalize_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_side = max(w, h)
    min_side = min(w, h)

    if max_side > _MAX_SIZE:
        scale = _MAX_SIZE / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif min_side < _MIN_SIZE:
        scale = _MIN_SIZE / min_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def _denoise(img: Image.Image) -> Image.Image:
    """약한 가우시안 블러로 고주파 노이즈 제거."""
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def _enhance(img: Image.Image) -> Image.Image:
    """대비와 선명도 강화."""
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img


def _auto_crop(img: Image.Image) -> Image.Image:
    """
    엣지 감지로 콘텐츠 영역 추정 후 여백 제거.
    배경이 단색이거나 균일한 경우 효과적.
    실패 시 원본 반환.
    """
    try:
        gray = img.convert("L")
        # 엣지 강조 후 임계값으로 콘텐츠 영역 bbox 추정
        edges = gray.filter(ImageFilter.FIND_EDGES)
        bbox = edges.point(lambda x: 255 if x > 15 else 0).getbbox()
        if bbox is None:
            return img

        # 여백을 10% 추가하되 이미지 경계를 넘지 않도록
        w, h = img.size
        pad_x = int((bbox[2] - bbox[0]) * 0.10)
        pad_y = int((bbox[3] - bbox[1]) * 0.10)
        left  = max(0, bbox[0] - pad_x)
        upper = max(0, bbox[1] - pad_y)
        right = min(w, bbox[2] + pad_x)
        lower = min(h, bbox[3] + pad_y)

        # 크롭 결과가 원본의 20% 이하면 잘못된 크롭이므로 원본 반환
        crop_area = (right - left) * (lower - upper)
        if crop_area < w * h * 0.20:
            return img

        return img.crop((left, upper, right, lower))
    except Exception:
        return img


def preprocess(image_path: Path | str) -> Path:
    """
    이미지 전처리 후 임시 파일로 저장하여 경로 반환.
    원본 파일은 수정하지 않음.
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")

    img = _auto_crop(img)
    img = _normalize_size(img)
    img = _denoise(img)
    img = _enhance(img)

    out_path = image_path.parent / f"_preprocessed_{image_path.name}"
    img.save(out_path, quality=92)
    return out_path
