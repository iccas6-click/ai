"""
건기식 성분 해석율(resolve rate) 측정 스크립트.

실제 앱에서 들어올 법한 성분명 표현을 넣어서
백엔드 supplement_resolver가 supplement_id로 매핑하는 비율을 측정.

실행 (ai/ 루트에서):
    python supplement_recognition/scripts/evaluate_resolve_rate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.chdir(Path(__file__).resolve().parent.parent.parent)

from dotenv import load_dotenv
load_dotenv()

import mysql.connector

# ────────────────────────────────────────────────────────────────
# 테스트셋: (입력 성분명, 정답 supplement_id 또는 None)
# 실제 앱에서 이미지 파이프라인이나 사용자가 입력할 법한 표현들
TEST_CASES: list[tuple[str, str | None]] = [
    # 오메가-3 계열
    ("오메가3", "SUPP_004"),
    ("오메가-3", "SUPP_004"),
    ("EPA 및 DHA", "SUPP_004"),
    ("EPA·DHA", "SUPP_004"),
    ("알티지 오메가3", "SUPP_004"),
    ("rTG 오메가3", "SUPP_004"),
    # 프로바이오틱스
    ("프로바이오틱스", "SUPP_002"),
    ("유산균", "SUPP_002"),
    ("생유산균", "SUPP_002"),
    ("락토바실러스", "SUPP_002"),
    # 인삼/홍삼
    ("인삼", "SUPP_001"),
    ("홍삼", "SUPP_001"),
    ("고려인삼", "SUPP_001"),
    ("홍삼농축액", "SUPP_001"),
    # 밀크씨슬
    ("밀크씨슬", "SUPP_005"),
    ("실리마린", "SUPP_005"),
    ("카르두스 마리아누스", "SUPP_005"),
    # 코엔자임 Q10
    ("코엔자임 Q10", "SUPP_021"),
    ("코큐텐", "SUPP_021"),
    ("CoQ10", "SUPP_021"),
    # 글루코사민
    ("글루코사민", "SUPP_015"),
    ("글루코사민 황산염", "SUPP_015"),
    # 은행잎
    ("은행잎", "SUPP_022"),
    ("징코", "SUPP_022"),
    ("Ginkgo", "SUPP_022"),
    # 녹차
    ("녹차", "SUPP_012"),
    ("녹차추출물", "SUPP_012"),
    ("EGCG", "SUPP_012"),
    # 울금/커큐민
    ("울금", "SUPP_027"),
    ("커큐민", "SUPP_027"),
    ("강황", "SUPP_027"),
    # 마늘
    ("마늘", "SUPP_028"),
    ("알리신", "SUPP_028"),
    # 크랜베리
    ("크랜베리", "SUPP_025"),
    ("크랜베리 추출물", "SUPP_025"),
    # 알로에
    ("알로에", "SUPP_003"),
    ("알로에 베라", "SUPP_003"),
    # 스피루리나
    ("스피루리나", "SUPP_014"),
    # 클로렐라
    ("클로렐라", "SUPP_019"),
    # 쏘팔메토
    ("쏘팔메토", "SUPP_023"),
    ("쏘 팔메토", "SUPP_023"),
    # 포스파티딜세린
    ("포스파티딜세린", "SUPP_024"),
    ("PS", None),  # 약어 → 해석 어려움
    # 키토산
    ("키토산", "SUPP_013"),
    ("키토올리고당", "SUPP_013"),
    # 감마리놀렌산
    ("감마리놀렌산", "SUPP_006"),
    ("GLA", "SUPP_006"),
    # 공액리놀레산
    ("공액리놀레산", "SUPP_020"),
    ("CLA", "SUPP_020"),
    # L-카르니틴
    ("L-카르니틴", "SUPP_011"),
    ("엘카르니틴", "SUPP_011"),
    # 가시오갈피
    ("가시오갈피", "SUPP_017"),
    # 석류
    ("석류", "SUPP_016"),
    # 돌외잎
    ("돌외잎", "SUPP_009"),
    ("지아노사이드", None),  # 성분명 아님
    # 당귀
    ("당귀", "SUPP_007"),
    # 대두
    ("대두", "SUPP_010"),
    ("대두이소플라본", "SUPP_010"),
    ("이소플라본", "SUPP_010"),
    # 아프리카망고
    ("아프리카망고", "SUPP_018"),
    # 호로파
    ("호로파", "SUPP_030"),
    ("호로파 종자", "SUPP_030"),
    # 감초
    ("감초", "SUPP_026"),
    # 마테
    ("마테", "SUPP_008"),
    # 완전 비매핑 케이스 (None이 정답)
    ("비타민C", None),
    ("비타민D", None),
    ("칼슘", None),
    ("마그네슘", None),
    ("아연", None),
    ("루테인", None),
    ("지아잔틴", None),
    ("콜라겐", None),
    ("히알루론산", None),
    ("NAC", None),
]
# ────────────────────────────────────────────────────────────────


_BACKEND_DB = {
    "host": "127.0.0.1",
    "port": 3307,
    "database": "click_backend_db",
    "user": "click_user",
    "password": "clickbackend0625",
    "charset": "utf8mb4",
}


def _get_conn():
    return mysql.connector.connect(**_BACKEND_DB)


def resolve(name: str, cursor) -> str | None:
    """supplement_entities에서 이름으로 supplement_id 조회 (다단계 fallback)."""
    clean = name.strip()
    normalized = clean.lower().replace(" ", "").replace("-", "").replace("_", "")

    # 1) 정확 일치 (ko)
    cursor.execute(
        "SELECT supplement_id FROM supplement_entities WHERE supplement_name_ko = %s LIMIT 1",
        (clean,),
    )
    row = cursor.fetchone()
    if row:
        return row["supplement_id"]

    # 2) 정확 일치 (en)
    cursor.execute(
        "SELECT supplement_id FROM supplement_entities WHERE LOWER(supplement_name_en) = LOWER(%s) LIMIT 1",
        (clean,),
    )
    row = cursor.fetchone()
    if row:
        return row["supplement_id"]

    # 3) 정규화 일치
    cursor.execute(
        """
        SELECT supplement_id FROM supplement_entities
        WHERE REPLACE(REPLACE(REPLACE(LOWER(supplement_name_ko), ' ', ''), '-', ''), '_', '') = %s
           OR REPLACE(REPLACE(REPLACE(LOWER(COALESCE(supplement_name_en,'')), ' ', ''), '-', ''), '_', '') = %s
        LIMIT 1
        """,
        (normalized, normalized),
    )
    row = cursor.fetchone()
    if row:
        return row["supplement_id"]

    # 4) 부분 일치 (ko 포함 or 포함됨)
    cursor.execute(
        """
        SELECT supplement_id FROM supplement_entities
        WHERE supplement_name_ko LIKE %s OR %s LIKE CONCAT('%%', supplement_name_ko, '%%')
        ORDER BY LENGTH(supplement_name_ko) ASC
        LIMIT 1
        """,
        (f"%{clean}%", clean),
    )
    row = cursor.fetchone()
    if row:
        return row["supplement_id"]

    return None


def run():
    conn = _get_conn()
    cursor = conn.cursor(dictionary=True)

    total = len(TEST_CASES)
    # 정답이 있는 케이스 (None 아닌 것)
    positive_cases = [(n, sid) for n, sid in TEST_CASES if sid is not None]
    # 정답이 없는 케이스 (비매핑이어야 정상)
    negative_cases = [(n, sid) for n, sid in TEST_CASES if sid is None]

    tp = 0  # 정답 있고 올바르게 매핑
    fn = 0  # 정답 있는데 못 찾음
    fp = 0  # 정답 없는데 뭔가 매핑됨 (오매핑)
    tn = 0  # 정답 없고 매핑 안 됨 (정상)

    wrong_rows = []
    false_positive_rows = []

    print(f"\n{'='*60}")
    print(f"건기식 성분 해석율(Resolve Rate) 평가")
    print(f"{'='*60}")

    print(f"\n[매핑 대상 케이스 — {len(positive_cases)}개]")
    for name, expected_id in positive_cases:
        got_id = resolve(name, cursor)
        ok = got_id == expected_id
        if ok:
            tp += 1
        else:
            fn += 1
            wrong_rows.append((name, expected_id, got_id))
        status = "✓" if ok else f"✗ (got: {got_id})"
        print(f"  {name:<30} → {status}")

    print(f"\n[비매핑 케이스 — {len(negative_cases)}개]")
    for name, _ in negative_cases:
        got_id = resolve(name, cursor)
        if got_id is None:
            tn += 1
            status = "✓ (None)"
        else:
            fp += 1
            false_positive_rows.append((name, got_id))
            status = f"✗ 오매핑→ {got_id}"
        print(f"  {name:<30} → {status}")

    cursor.close()
    conn.close()

    # 결과 출력
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"[결과 요약]")
    print(f"  TP (정답 있고 올바르게 매핑):  {tp}/{len(positive_cases)} = {tp/len(positive_cases)*100:.1f}%")
    print(f"  FN (정답 있는데 못 찾음):       {fn}/{len(positive_cases)} = {fn/len(positive_cases)*100:.1f}%")
    print(f"  TN (비매핑 케이스, 정상 처리):  {tn}/{len(negative_cases)} = {tn/len(negative_cases)*100:.1f}%")
    print(f"  FP (없어야 하는데 오매핑):      {fp}/{len(negative_cases)} = {fp/len(negative_cases)*100:.1f}%")
    print(f"")
    print(f"  Precision:  {precision*100:.1f}%")
    print(f"  Recall:     {recall*100:.1f}%")
    print(f"  F1 Score:   {f1*100:.1f}%")

    if wrong_rows:
        print(f"\n⚠ FN (매핑 실패) {len(wrong_rows)}건:")
        for name, expected, got in wrong_rows:
            print(f"  '{name}' → 기대: {expected}, 실제: {got}")

    if false_positive_rows:
        print(f"\n⚠ FP (오매핑) {len(false_positive_rows)}건:")
        for name, got in false_positive_rows:
            print(f"  '{name}' → 잘못 매핑됨: {got}")


if __name__ == "__main__":
    run()
