"""
정확도 측정 스크립트.

준비:
    supplement_recognition/data/samples/ 폴더에 이미지를 넣어두세요.
    파일명이 정답 제품명으로 사용됩니다.

    예)
        data/samples/TWK10 100억 분말.jpg
        data/samples/종근당 오메가3.png

실행 (ai/ 루트에서):
    python supplement_recognition/scripts/evaluate_accuracy.py
    python supplement_recognition/scripts/evaluate_accuracy.py --samples data/samples/
"""

import argparse
import os
import sys
import time
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz

from supplement_recognition.src.extraction.llm_extractor import extract_product_candidates
from supplement_recognition.src.pipeline import recognize

SAMPLES_DIR = Path("supplement_recognition/data/samples")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(samples_dir: Path) -> list[Path]:
    if not samples_dir.exists():
        return []
    return sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run(images: list[Path]) -> None:
    total = len(images)
    gemini_ok = 0
    db_match_ok = 0
    completed = 0
    similarity_sum = 0.0

    rows = []

    print(f"\n총 {total}개 이미지\n{'='*60}")

    for i, img_path in enumerate(images, 1):
        expected = img_path.stem  # 파일명(확장자 제외) = 정답 제품명
        print(f"[{i:2d}/{total}] {img_path.name}")
        print(f"       정답: {expected}")

        # Gemini 단독 추출 (후보 최대 3개)
        try:
            candidates = extract_product_candidates(img_path)
        except Exception as e:
            candidates = []
            print(f"       Gemini 오류: {e}")

        gemini_name = candidates[0] if candidates else ""
        sim = fuzz.partial_ratio(expected, gemini_name)
        similarity_sum += sim

        if gemini_name.strip():
            gemini_ok += 1

        print(f"       Gemini 후보: {candidates}")
        print(f"       1순위 유사도: {sim:.0f}%")

        # 전체 파이프라인
        t0 = time.time()
        result = recognize(img_path)
        elapsed = time.time() - t0

        if result.product and result.product.product_code is not None:
            matched_name = result.product.product_name
            name_sim = fuzz.partial_ratio(expected, matched_name)
            db_correct = name_sim >= 70
            if db_correct:
                db_match_ok += 1
            marker = "✓" if db_correct else "✗ (틀린 제품)"
            print(f"       DB 매칭: {marker}")
            print(f"         매칭된: {matched_name}  (정답 유사도: {name_sim:.0f}%)")
        else:
            db_correct = False
            matched_name = ""
            print(f"       DB 매칭: ✗  (미매칭)")

        if result.status == "completed" and db_correct:
            completed += 1

        print(f"       처리 시간: {elapsed:.1f}초\n")

        rows.append({
            "파일": img_path.name,
            "정답": expected,
            "Gemini 추출": gemini_name,
            "Gemini 유사도": f"{sim:.0f}%",
            "매칭된 제품": matched_name,
            "DB 정확": "✓" if db_correct else "✗",
            "상태": result.status,
        })

        if i < total:
            time.sleep(1.0)

    # 요약
    print("=" * 60)
    print(f"Gemini 추출 성공률:  {gemini_ok}/{total} = {gemini_ok/total*100:.1f}%")
    print(f"Gemini 제품명 유사도: {similarity_sum/total:.1f}%  (추출명 vs 정답)")
    print(f"DB 정확 매칭률:      {db_match_ok}/{total} = {db_match_ok/total*100:.1f}%  (올바른 제품 매칭)")
    print(f"파이프라인 완료율:   {completed}/{total} = {completed/total*100:.1f}%")

    wrong = [r for r in rows if r["DB 정확"] == "✗"]
    if wrong:
        print(f"\n⚠ DB 매칭 실패 또는 틀린 제품 ({len(wrong)}건):")
        for r in wrong:
            print(f"   [{r['파일']}]")
            print(f"     정답:     {r['정답']}")
            print(f"     Gemini:   {r['Gemini 추출']}  ({r['Gemini 유사도']})")
            if r["매칭된 제품"]:
                print(f"     매칭됨:   {r['매칭된 제품']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=str(SAMPLES_DIR), help="이미지 폴더 경로")
    args = parser.parse_args()

    samples_dir = Path(args.samples)
    images = collect_images(samples_dir)

    if not images:
        print(f"이미지가 없습니다: {samples_dir.resolve()}")
        print(f"\n폴더를 만들고 이미지를 넣어주세요:")
        print(f"  {samples_dir.resolve()}/")
        print(f"\n파일명이 정답 제품명으로 사용됩니다.")
        print(f"  예) 'TWK10 100억 분말.jpg'  →  정답: 'TWK10 100억 분말'")
        sys.exit(0)

    run(images)
