"""
상호작용 정확도 평가 스크립트.

측정 항목:
  1. DB 내부 검증 — 475건 모두 (supplement_name_ko + canonical_drug_name_ko)로 역조회 가능한지
  2. 처방전 약물 커버리지 — 실제 처방전에서 추출된 성분명이 canonical_drug_entities에 매핑되는 비율
  3. 상호작용 히트율 — 처방전 약물 × 33종 건기식 성분 조합 중 상호작용이 존재하는 비율
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mysql.connector

DB = dict(
    host="127.0.0.1",
    port=3307,
    database="click_backend_db",
    user="click_user",
    password="clickbackend0625",
    charset="utf8mb4",
)

# 실제 처방전 20장에서 추출된 약물 성분명 (evaluate_prescription_accuracy.py 결과 기반)
PRESCRIPTION_INGREDIENTS = [
    "클래리트로마이신", "로녹시캄", "메틸프레드니솔론", "레보드로프로피진", "레바미피드",
    "모메타손 푸로에이트", "덱시부프로펜", "아세트아미노펜", "애엽에탄올연조엑스", "에르도스테인",
    "소브레롤", "시프로플록사신", "록소프로펜", "라니티딘", "부틸스코폴라민브롬화물",
    "티아넵틴나트륨", "염산메칠페니데이트", "레보설피리드", "니자티딘", "토피소팜",
    "아세틸시스테인", "세파클러", "클로나제팜", "프로프라놀롤", "설트랄린염산염",
    "탄산리튬", "아리피프라졸", "플루옥세틴", "쿠에티아핀푸마르산염", "플루니트라제팜",
    "알프라졸람", "트라조돈염산염", "트리아졸람", "티로프라미드염산염", "바실루스서브틸리스",
    "엔테로코쿠스페시움", "트리메부틴말레산염", "디옥타헤드랄스멕타이트", "로페라미드염산염",
    "에스오메프라졸마그네슘삼수화물", "아미트리프틸린염산염", "프레가발린", "신바로건조엑스",
    "아미트립틸린염산염", "프레가발린",
    "덱시부프로펜", "레바미피드", "디히드로코데인타르타르산염", "클로르페니라민말레산염",
    "DL-메틸에페드린염산염", "구아이페네신", "슈도에페드린염산염",
    "셀레콕시브", "베포타스틴", "일라프라졸", "티로프라미드", "글리메피리드", "메트포르민",
    "트라마돌", "세파클러", "레보드로프로피진", "엘도스테인", "클로르페니라민말레산염",
    "페닐레프린염산염", "모니플루메이트",
    "아목시실린 수화물", "클라불란산칼륨", "에피나스틴염산염", "몬테루카스트나트륨",
    "dl-메틸에페드린염산염", "암모늄염화물",
    "아목시실린나트륨", "묽은클라불란산칼륨", "베포타스틴베실산염",
    "올로파타딘염산염", "레보플록사신수화물", "히알루론산나트륨",
    "세레콕시브", "티아프리드염산염", "글리메피리드", "메트포르민염산염",
    "아세트아미노펜", "트라마돌염산염",
    "아세클로페낙", "에페리손염산염", "애엽95%에탄올연조엑스",
]


def _normalize(name: str) -> str:
    return re.sub(r"[\s\-_()/·ㆍ.,]+", "", name.strip().lower())


def run():
    conn = mysql.connector.connect(**DB)
    c = conn.cursor(dictionary=True)

    # ── 1. DB 내부 검증 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. DB 내부 검증 — 475건 역조회 가능 여부")
    print("=" * 70)

    c.execute("""
        SELECT si.interaction_id, se.supplement_name_ko, cde.canonical_drug_name_ko
        FROM standardized_interactions si
        JOIN supplement_entities se ON se.supplement_id = si.supplement_id
        JOIN canonical_drug_entities cde ON cde.canonical_drug_id = si.canonical_drug_id
    """)
    all_interactions = c.fetchall()
    total = len(all_interactions)

    retrievable = 0
    not_found = []
    for row in all_interactions:
        supp_name = row["supplement_name_ko"]
        drug_name = row["canonical_drug_name_ko"]
        c.execute("""
            SELECT si.interaction_id
            FROM standardized_interactions si
            JOIN supplement_entities se ON se.supplement_id = si.supplement_id
            JOIN canonical_drug_entities cde ON cde.canonical_drug_id = si.canonical_drug_id
            WHERE se.supplement_name_ko = %s AND cde.canonical_drug_name_ko = %s
            LIMIT 1
        """, (supp_name, drug_name))
        if c.fetchone():
            retrievable += 1
        else:
            not_found.append(row["interaction_id"])

    print(f"전체 상호작용: {total}건")
    print(f"역조회 성공:   {retrievable}/{total} = {retrievable/total*100:.1f}%")
    if not_found:
        print(f"실패 interaction_id: {not_found[:10]}")

    # ── 2. 처방전 약물 커버리지 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. 처방전 약물 → canonical_drug_entities 매핑률")
    print("=" * 70)

    unique_ingredients = list(dict.fromkeys(PRESCRIPTION_INGREDIENTS))  # 중복 제거
    mapped = []
    unmapped = []

    for ing in unique_ingredients:
        norm = _normalize(ing)
        c.execute("""
            SELECT canonical_drug_id, canonical_drug_name_ko
            FROM canonical_drug_entities
            WHERE canonical_drug_name_ko = %s
               OR REPLACE(REPLACE(REPLACE(LOWER(canonical_drug_name_ko), ' ', ''), '-', ''), '.', '') = %s
               OR REPLACE(REPLACE(REPLACE(LOWER(canonical_drug_name_en), ' ', ''), '-', ''), '.', '') = %s
            LIMIT 1
        """, (ing, norm, norm))
        row = c.fetchone()
        if row:
            mapped.append((ing, row["canonical_drug_name_ko"]))
        else:
            # drug_aliases로 재시도
            c.execute("""
                SELECT cde.canonical_drug_id, cde.canonical_drug_name_ko
                FROM drug_aliases da
                JOIN canonical_drug_entities cde ON cde.canonical_drug_id = da.canonical_drug_id
                WHERE da.alias_name_normalized LIKE %s
                LIMIT 1
            """, (f"%{norm}%",))
            row = c.fetchone()
            if row:
                mapped.append((ing, row["canonical_drug_name_ko"]))
            else:
                unmapped.append(ing)

    print(f"처방전 추출 성분 (중복 제거): {len(unique_ingredients)}건")
    print(f"canonical_drug_entities 매핑 성공: {len(mapped)}/{len(unique_ingredients)} = {len(mapped)/len(unique_ingredients)*100:.1f}%")
    if unmapped:
        print(f"\n미매핑 성분 ({len(unmapped)}건):")
        for u in unmapped:
            print(f"  - {u}")

    # ── 3. 상호작용 히트율 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. 처방전 약물 × 33종 건기식 상호작용 히트율")
    print("=" * 70)

    c.execute("SELECT supplement_id, supplement_name_ko FROM supplement_entities ORDER BY supplement_id")
    supplements = c.fetchall()

    mapped_drug_ids = []
    for ing, _ in mapped:
        norm = _normalize(ing)
        c.execute("""
            SELECT canonical_drug_id FROM canonical_drug_entities
            WHERE canonical_drug_name_ko = %s
               OR REPLACE(REPLACE(REPLACE(LOWER(canonical_drug_name_ko), ' ', ''), '-', ''), '.', '') = %s
            LIMIT 1
        """, (ing, norm))
        row = c.fetchone()
        if row:
            mapped_drug_ids.append(row["canonical_drug_id"])

    unique_drug_ids = list(dict.fromkeys(mapped_drug_ids))
    total_combinations = len(unique_drug_ids) * len(supplements)

    hit_pairs = []
    for drug_id in unique_drug_ids:
        for supp in supplements:
            c.execute("""
                SELECT interaction_id FROM standardized_interactions
                WHERE canonical_drug_id = %s AND supplement_id = %s
                LIMIT 1
            """, (drug_id, supp["supplement_id"]))
            if c.fetchone():
                c.execute("SELECT canonical_drug_name_ko FROM canonical_drug_entities WHERE canonical_drug_id = %s", (drug_id,))
                drug_row = c.fetchone()
                hit_pairs.append((drug_row["canonical_drug_name_ko"], supp["supplement_name_ko"]))

    print(f"처방전 약물 (canonical 매핑된 것): {len(unique_drug_ids)}종")
    print(f"건기식 성분: {len(supplements)}종")
    print(f"전체 조합: {total_combinations}건")
    print(f"상호작용 존재 조합: {len(hit_pairs)}건 = {len(hit_pairs)/total_combinations*100:.1f}%")

    if hit_pairs:
        print(f"\n상호작용 발견 약물-건기식 조합 (상위 20건):")
        for drug, supp in hit_pairs[:20]:
            print(f"  {drug}  ×  {supp}")
        if len(hit_pairs) > 20:
            print(f"  ... 외 {len(hit_pairs)-20}건")

    conn.close()

    # ── 요약 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"① DB 내부 역조회 성공률       : {retrievable}/{total} = {retrievable/total*100:.1f}%")
    print(f"② 처방전 약물 DB 매핑률        : {len(mapped)}/{len(unique_ingredients)} = {len(mapped)/len(unique_ingredients)*100:.1f}%")
    print(f"③ 처방전×건기식 상호작용 히트율: {len(hit_pairs)}/{total_combinations} = {len(hit_pairs)/total_combinations*100:.1f}%")


if __name__ == "__main__":
    run()
