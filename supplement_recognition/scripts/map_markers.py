"""
supplement_product_markers.marker_text 를
백엔드 supplement_aliases 와 대조해 supplement_id 를 채우는 스크립트.

매핑 전략 (우선순위 순):
  1. 정확 일치       marker_text == alias
  2. 정규화 일치     normalize(marker_text) == normalize(alias)
  3. 부분 포함       alias in marker_text  (alias 길이 >= 2)

사용법: python supplement_recognition/scripts/map_markers.py
"""
from __future__ import annotations

import os
import re
import sys

import mysql.connector
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# 백엔드 supplement_aliases (supplement_id, alias)
ALIASES: list[tuple[str, str]] = [
    # SUPP_001 인삼
    ("SUPP_001", "홍삼"), ("SUPP_001", "홍삼추출물"), ("SUPP_001", "인삼추출물"),
    ("SUPP_001", "홍삼분말"), ("SUPP_001", "Korean red ginseng"), ("SUPP_001", "Red ginseng"),
    ("SUPP_001", "Panax ginseng"), ("SUPP_001", "진세노사이드"), ("SUPP_001", "Ginsenoside"),
    ("SUPP_001", "진세노사이드 Rg1"), ("SUPP_001", "진세노사이드 Rb1"),
    # SUPP_002 프로바이오틱스
    ("SUPP_002", "유산균"), ("SUPP_002", "Lactobacillus"), ("SUPP_002", "Bifidobacterium"),
    ("SUPP_002", "락토바실러스"), ("SUPP_002", "비피도박테리움"), ("SUPP_002", "TWK10"),
    ("SUPP_002", "Lactiplantibacillus plantarum TWK10"), ("SUPP_002", "Lactobacillus plantarum TWK10"),
    ("SUPP_002", "Lactobacillus sakei Probio65"), ("SUPP_002", "Bifidobacterium breve B-3"),
    ("SUPP_002", "HY7714"), ("SUPP_002", "드시모네"), ("SUPP_002", "리스펙타"),
    ("SUPP_002", "L. curvatus HY7601"), ("SUPP_002", "L. plantarum KY1032"),
    # SUPP_003 알로에
    ("SUPP_003", "알로에베라"), ("SUPP_003", "알로에겔"), ("SUPP_003", "알로에추출물"),
    ("SUPP_003", "Aloe vera"),
    # SUPP_004 오메가-3
    ("SUPP_004", "오메가3"), ("SUPP_004", "오메가-3"), ("SUPP_004", "EPA"), ("SUPP_004", "DHA"),
    ("SUPP_004", "피쉬오일"), ("SUPP_004", "어유"), ("SUPP_004", "피시오일"), ("SUPP_004", "EPA 및 DHA"),
    # SUPP_005 밀크씨슬
    ("SUPP_005", "실리마린"), ("SUPP_005", "밀크시슬"), ("SUPP_005", "카르두스마리아누스"),
    ("SUPP_005", "엉겅퀴"), ("SUPP_005", "Silybum marianum"),
    # SUPP_006 감마리놀렌산
    ("SUPP_006", "달맞이꽃종자유"), ("SUPP_006", "GLA"), ("SUPP_006", "보리지오일"),
    ("SUPP_006", "감마리놀렌산"),
    # SUPP_007 당귀
    ("SUPP_007", "당귀추출물"), ("SUPP_007", "당귀분말"), ("SUPP_007", "Angelica gigas"),
    # SUPP_008 마테
    ("SUPP_008", "마테차"), ("SUPP_008", "예르바마테"), ("SUPP_008", "Ilex paraguariensis"),
    # SUPP_009 돌외잎
    ("SUPP_009", "돌외잎추출물"), ("SUPP_009", "교소란"), ("SUPP_009", "지피노사이드"),
    ("SUPP_009", "Gynostemma pentaphyllum"),
    # SUPP_010 대두
    ("SUPP_010", "이소플라본"), ("SUPP_010", "대두이소플라본"), ("SUPP_010", "콩이소플라본"),
    ("SUPP_010", "Glycine max"),
    # SUPP_011 L-카르니틴
    ("SUPP_011", "카르니틴"), ("SUPP_011", "Carnitine"), ("SUPP_011", "엘카르니틴"),
    ("SUPP_011", "L-Carnitine"),
    # SUPP_012 녹차
    ("SUPP_012", "녹차추출물"), ("SUPP_012", "카테킨"), ("SUPP_012", "EGCG"),
    ("SUPP_012", "녹차폴리페놀"), ("SUPP_012", "Camellia sinensis"),
    # SUPP_013 키토산/키토올리고당
    ("SUPP_013", "키토올리고당"), ("SUPP_013", "키토산"), ("SUPP_013", "Chitosan"),
    ("SUPP_013", "chitooligosaccharide"),
    # SUPP_014 스피루리나
    ("SUPP_014", "스피루리나분말"), ("SUPP_014", "Arthrospira"), ("SUPP_014", "Spirulina"),
    # SUPP_015 글루코사민
    ("SUPP_015", "글루코사민염산염"), ("SUPP_015", "글루코사민황산염"),
    ("SUPP_015", "N-아세틸글루코사민"), ("SUPP_015", "Glucosamine"),
    # SUPP_016 석류
    ("SUPP_016", "석류농축액"), ("SUPP_016", "석류추출물"), ("SUPP_016", "석류분말"),
    ("SUPP_016", "엘라그산"), ("SUPP_016", "Ellagic acid"), ("SUPP_016", "Punica granatum"),
    # SUPP_017 가시오갈피
    ("SUPP_017", "가시오가피"), ("SUPP_017", "가시오갈피추출물"), ("SUPP_017", "시베리아인삼"),
    ("SUPP_017", "Eleutherococcus senticosus"),
    # SUPP_018 아프리카망고
    ("SUPP_018", "아프리카 망고"), ("SUPP_018", "아프리카망고추출물"),
    ("SUPP_018", "Irvingia gabonensis"),
    # SUPP_019 클로렐라
    ("SUPP_019", "클로렐라분말"), ("SUPP_019", "Chlorella"),
    # SUPP_020 공액리놀레산
    ("SUPP_020", "CLA"), ("SUPP_020", "공액리놀레산유지"), ("SUPP_020", "Conjugated linoleic acid"),
    # SUPP_021 코엔자임Q10
    ("SUPP_021", "코큐텐"), ("SUPP_021", "CoQ10"), ("SUPP_021", "유비퀴논"),
    ("SUPP_021", "유비퀴놀"), ("SUPP_021", "코엔자임Q10"),
    # SUPP_022 은행잎
    ("SUPP_022", "은행잎추출물"), ("SUPP_022", "징코"), ("SUPP_022", "진코"),
    ("SUPP_022", "Ginkgo biloba"), ("SUPP_022", "Ginkgo"),
    # SUPP_023 쏘팔메토
    ("SUPP_023", "쏘팔메토추출물"), ("SUPP_023", "소팔메토"), ("SUPP_023", "Serenoa repens"),
    ("SUPP_023", "Saw palmetto"),
    # SUPP_024 포스파티딜세린
    ("SUPP_024", "PS"), ("SUPP_024", "Phosphatidylserine"),
    # SUPP_025 크랜베리
    ("SUPP_025", "크랜베리추출물"), ("SUPP_025", "크렌베리"),
    ("SUPP_025", "Vaccinium macrocarpon"), ("SUPP_025", "Cranberry"),
    # SUPP_026 감초
    ("SUPP_026", "감초추출물"), ("SUPP_026", "스페인감초추출물"), ("SUPP_026", "리코리스"),
    ("SUPP_026", "Glycyrrhiza"), ("SUPP_026", "Licorice"),
    # SUPP_027 울금(커큐민)
    ("SUPP_027", "커큐민"), ("SUPP_027", "강황"), ("SUPP_027", "터메릭"),
    ("SUPP_027", "울금추출물"), ("SUPP_027", "Curcumin"), ("SUPP_027", "Turmeric"),
    ("SUPP_027", "울금[커큐민]"), ("SUPP_027", "울금 [커큐민]"),
    # SUPP_028 마늘
    ("SUPP_028", "마늘추출물"), ("SUPP_028", "마늘분말"), ("SUPP_028", "알리신"),
    ("SUPP_028", "흑마늘"), ("SUPP_028", "흑마늘추출물"), ("SUPP_028", "Allicin"),
    ("SUPP_028", "Garlic"),
    # SUPP_029 오미자
    ("SUPP_029", "오미자추출물"), ("SUPP_029", "오미자분말"), ("SUPP_029", "오미자베리"),
    ("SUPP_029", "Schisandra"),
    # SUPP_030 호로파
    ("SUPP_030", "호로파"), ("SUPP_030", "호로파종자추출물"), ("SUPP_030", "Fenugreek"),
    ("SUPP_030", "Trigonella foenum-graecum"),
    # SUPP_031 루바브
    ("SUPP_031", "루바브"), ("SUPP_031", "대황"), ("SUPP_031", "Rhubarb"), ("SUPP_031", "Rheum"),
    # SUPP_032 글루코사민-콘드로이틴
    ("SUPP_032", "콘드로이틴"), ("SUPP_032", "콘드로이친"), ("SUPP_032", "글루코사민콘드로이틴"),
    ("SUPP_032", "Chondroitin"),
    # SUPP_033 인동덩굴
    ("SUPP_033", "인동덩굴추출물"), ("SUPP_033", "인동덩굴꽃봉오리추출물"),
    ("SUPP_033", "그린세라-F"), ("SUPP_033", "인동"), ("SUPP_033", "금은화"),
    ("SUPP_033", "Lonicera japonica"),
]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-·•,，()（）\[\]]+", "", text).lower()


def _build_lookup() -> tuple[dict[str, str], dict[str, str], list[tuple[str, str]]]:
    """정확 일치 / 정규화 일치 / 부분 포함 순으로 룩업 자료구조 생성."""
    exact: dict[str, str] = {}        # alias → supplement_id
    normalized: dict[str, str] = {}   # normalize(alias) → supplement_id
    partial: list[tuple[str, str]] = []  # (alias, supplement_id) 길이 내림차순

    for supp_id, alias in ALIASES:
        exact[alias] = supp_id
        normalized[_normalize(alias)] = supp_id

    # 부분 포함: alias 길이 긴 것 우선 (짧은 alias가 먼저 매칭되는 오류 방지)
    partial = sorted(ALIASES, key=lambda x: len(x[1]), reverse=True)
    return exact, normalized, partial


def _match(marker_text: str, exact, normalized_map, partial) -> str | None:
    # 1. 정확 일치
    if marker_text in exact:
        return exact[marker_text]
    # 2. 정규화 일치
    norm = _normalize(marker_text)
    if norm in normalized_map:
        return normalized_map[norm]
    # 3. 부분 포함 (alias가 marker_text 안에 있거나 marker_text가 alias 안에)
    for supp_id, alias in partial:
        if len(alias) < 2:
            continue
        if alias in marker_text or marker_text in alias:
            return supp_id
    return None


def _get_conn():
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        charset="utf8mb4",
    )


def map_markers() -> None:
    exact, normalized_map, partial = _build_lookup()

    conn = _get_conn()
    cursor = conn.cursor(dictionary=True)

    print("supplement_product_markers 로드 중...")
    cursor.execute("SELECT marker_id, marker_text FROM supplement_product_markers")
    rows = cursor.fetchall()
    print(f"  총 {len(rows):,}건")

    mapped = 0
    unmapped = 0

    for row in rows:
        supp_id = _match(row["marker_text"], exact, normalized_map, partial)
        if supp_id:
            cursor.execute(
                "UPDATE supplement_product_markers "
                "SET supplement_id = %s, mapping_status = 'mapped' "
                "WHERE marker_id = %s",
                (supp_id, row["marker_id"]),
            )
            mapped += 1
        else:
            unmapped += 1

        if (mapped + unmapped) % 10000 == 0:
            conn.commit()
            print(f"  진행: {mapped + unmapped:,}건 처리 ({mapped:,} mapped)")

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n완료: mapped {mapped:,}건 / unmapped {unmapped:,}건")
    print(f"매핑률: {mapped / (mapped + unmapped) * 100:.1f}%")


if __name__ == "__main__":
    map_markers()
