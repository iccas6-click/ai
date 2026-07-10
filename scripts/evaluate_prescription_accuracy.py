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
    extraction_success = 0   # 경구 의약품 1개 이상 추출된 이미지
    total_drugs = 0          # 추출된 경구 약품명 수
    product_db_matched = 0   # 제품명이 pill_products에 실제 존재 (legacy_product_table / official_product_catalog)
    ingredient_matched = 0   # 제품명 DB 없음, Gemini 성분명 → canonical_drug_entities 매핑 (ingredient_text)
    llm_candidate = 0        # Gemini 2차 추출 → canonical (llm_product_ingredient_candidate)
    not_matched = 0          # 완전 미매칭 (not_found / llm_only)
    external_use_images = 0  # 외용제(안약 등) 이미지 수
    doc_types: dict[str, int] = {}
    failures: list[str] = []

    _PRODUCT_DB_TYPES = {"legacy_product_table", "official_product_catalog"}

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

        # 외용제 이미지: 약품명 추출은 됐지만 DB 매칭 대상 아님
        if doc_type == "eye_drop":
            external_use_images += 1
            for med in medications:
                print(f"       [외용제] {med.get('product_name', '')}  (DB 매칭 제외)")
            continue

        if not medications:
            print("       약품명 추출: 없음")
            failures.append(img_path.name)
            continue

        extraction_success += 1
        for med in medications:
            total_drugs += 1
            match_type = med.get("match_type", "not_found")
            if match_type in _PRODUCT_DB_TYPES:
                product_db_matched += 1
                marker = "DB"
            elif match_type == "ingredient_text":
                ingredient_matched += 1
                marker = "성분"
            elif match_type == "llm_product_ingredient_candidate":
                llm_candidate += 1
                marker = "LLM"
            else:
                not_matched += 1
                marker = "X"
            ingredients = med.get("ingredients") or []
            print(f"       [{marker}] {med['product_name']}")
            print(f"         match={match_type}  성분={', '.join(ingredients) if ingredients else '없음'}")

    oral_images = total_images - external_use_images

    # --- 요약 ---
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"테스트 이미지 수          : {total_images}장")
    print(f"  경구 의약품 이미지       : {oral_images}장")
    print(f"  외용제(안약 등) 이미지   : {external_use_images}장 (DB 매칭 제외)")
    print(f"약품명 추출 성공          : {extraction_success}/{oral_images} = {extraction_success/oral_images*100:.1f}%" if oral_images else "약품명 추출 성공: N/A")
    print(f"추출된 경구 약품명 총 수  : {total_drugs}건")
    if total_drugs:
        print()
        print(f"[매칭 유형별 분류]")
        print(f"  제품명 DB 직접 매칭     : {product_db_matched}/{total_drugs} = {product_db_matched/total_drugs*100:.1f}%  (pill_products 실제 등재)")
        print(f"  성분명 canonical 매칭   : {ingredient_matched}/{total_drugs} = {ingredient_matched/total_drugs*100:.1f}%  (Gemini 성분 → canonical_drug_entities)")
        print(f"  LLM 2차 추출 매칭       : {llm_candidate}/{total_drugs} = {llm_candidate/total_drugs*100:.1f}%  (Gemini 2차 성분 → canonical)")
        print(f"  완전 미매칭             : {not_matched}/{total_drugs} = {not_matched/total_drugs*100:.1f}%")
    print(f"\n문서 유형 분포:")
    for dtype, count in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count}건")
    if failures:
        print(f"\n실패/미추출 이미지 ({len(failures)}건):")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    run()
