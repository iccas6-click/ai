"""
파이프라인 테스트 스크립트.
사용법: python scripts/test_pipeline.py data/samples/이미지파일명.jpg
"""
import json
import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")

from src.pipeline import recognize
from src.extraction.llm_extractor import extract_product_name
from src.matching.mfds_client import search_product
from dotenv import load_dotenv
load_dotenv()


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else input("이미지 경로: ")
    print(f"\n이미지: {image_path}")
    print("-" * 50)

    gemini_result = extract_product_name(image_path)
    print(f"[Gemini 인식] {gemini_result}")

    db_result = search_product(gemini_result)
    print(f"[DB 매칭]    {db_result.product_name if db_result else '없음'}")
    print("-" * 50)

    result = recognize(image_path)

    print(f"상태:     {result.status}")
    print(f"확인필요: {result.needs_confirmation}")

    if result.product:
        p = result.product
        print(f"\n제품명:   {p.product_name}")
        print(f"신고번호: {p.product_code}")
        print(f"제조사:   {p.manufacturer}")
        print(f"신뢰도:   {p.confidence}")
        print(f"\n주요기능:\n{p.main_function}")
        print(f"\n기준규격(성분):\n{p.base_standard}")

    if result.warnings:
        print(f"\n경고: {result.warnings}")

    if result.error_code:
        print(f"\n오류: {result.error_code} - {result.error_detail}")


if __name__ == "__main__":
    main()
