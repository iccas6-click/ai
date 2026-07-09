"""
처방전/약봉투 인식 정확도 평가 스크립트.

사용법:
    python scripts/evaluate_prescription_accuracy.py

측정 지표:
    - Gemini 약품명 추출 성공률: 이미지 당 1개 이상 추출된 비율
    - DB 매칭률: 추출된 약품명 중 백엔드 DB에서 매칭된 비율
    - 문서 유형 분포

환경 변수 (필요 시 설정):
    PILL_MYSQL_HOST / PILL_MYSQL_PORT / PILL_MYSQL_DATABASE / PILL_MYSQL_USER / PILL_MYSQL_PASSWORD
    CBNUAI_API_KEY 또는 GEMINI_API_KEY
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# --- 환경 변수 설정 ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CBNUAI_API_KEY", "dsu3nbxMK0K4kcwPnuLRRjUCyJnLAvxX")
os.environ.setdefault("PILL_MYSQL_HOST", "127.0.0.1")
os.environ.setdefault("PILL_MYSQL_PORT", "3307")
os.environ.setdefault("PILL_MYSQL_DATABASE", "click_backend_db")
os.environ.setdefault("PILL_MYSQL_USER", "click_user")
os.environ.setdefault("PILL_MYSQL_PASSWORD", "clickbackend0625")

from app.services.prescription_recognition import recognize_prescription_document

IMAGE_DIR = ROOT / "pill_recognition" / "datasets" / "evaluation" / "real-smartphone" / "images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def run():
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"이미지 없음: {IMAGE_DIR}")
        return

    total_images = len(images)
    extraction_success = 0   # 약품명 1개 이상 추출된 이미지
    total_drugs = 0          # 추출된 약품명 총 수
    db_matched = 0           # DB 매칭 성공 (not_found 아닌 것)
    doc_types: dict[str, int] = {}
    failures: list[str] = []

    print(f"\n총 {total_images}개 이미지")
    print("=" * 70)

    for i, img_path in enumerate(images, 1):
        t0 = time.perf_counter()
        print(f"\n[{i:2}/{total_images}] {img_path.name}")

        try:
            result = recognize_prescription_document(img_path)
        except Exception as e:
            print(f"       오류: {e}")
            failures.append(img_path.name)
            continue

        elapsed = time.perf_counter() - t0
        doc_type = result.get("document_type", "unknown")
        medications = result.get("medications") or []
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        print(f"       문서 유형: {doc_type}  |  처리 시간: {elapsed:.1f}초")

        if not medications:
            print("       약품명 추출: 없음")
            failures.append(img_path.name)
            continue

        extraction_success += 1
        for med in medications:
            total_drugs += 1
            match_type = med.get("match_type", "not_found")
            matched = match_type not in ("not_found", "llm_only")
            if matched:
                db_matched += 1
            ingredients = med.get("ingredients") or []
            marker = "✓" if matched else "✗"
            print(f"       {marker} {med['product_name']}")
            print(f"         match={match_type}  성분={', '.join(ingredients) if ingredients else '없음'}")

    # --- 요약 ---
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"테스트 이미지 수          : {total_images}장")
    print(f"약품명 추출 성공          : {extraction_success}/{total_images} = {extraction_success/total_images*100:.1f}%")
    print(f"추출된 약품명 총 수       : {total_drugs}건")
    if total_drugs:
        print(f"DB 매칭 성공              : {db_matched}/{total_drugs} = {db_matched/total_drugs*100:.1f}%")
        print(f"DB 미매칭 (not_found 등)  : {total_drugs - db_matched}/{total_drugs} = {(total_drugs-db_matched)/total_drugs*100:.1f}%")
    print(f"\n문서 유형 분포:")
    for dtype, count in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count}건")
    if failures:
        print(f"\n실패/미추출 이미지 ({len(failures)}건):")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    run()
