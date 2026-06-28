"""
base_standard / main_fnctn 텍스트에서 성분명 리스트를 파싱.

전략:
- main_fnctn: [성분명] 브래킷 패턴 + 개별인정 괄호 제거 → 신뢰도 높음
- base_standard: 성상/세균수/중금속 등 비성분 키워드 필터링 후 성분명 추출
"""
from __future__ import annotations

import re

# base_standard에서 걸러낼 비성분 키워드
_NON_INGREDIENT_KEYWORDS = (
    "성상", "세균수", "대장균", "납", "카드뮴", "비소", "수은", "붕해", "용출",
    "총균수", "진균수", "메틸수은", "총비소", "잔류", "이물", "산가", "과산화물가",
    "수분", "회분", "조단백", "조지방", "총 플라보노이드", "CFU", "Plate Count",
    "Yeast", "Mould", "E. coli", "S.", "Hexane", "헥산",
)

# 번호 패턴: ①②③ 또는 1. 2. 3. 또는 (1)(2)(3) 또는 ⑴⑵⑶
_NUMBERING = re.compile(
    r"^[\s]*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]"
    r"|^[\s]*[⑴⑵⑶⑷⑸⑹⑺⑻⑼]"
    r"|^[\s]*\d+[\.\)]\s*"
    r"|^[\s]*\(\d+\)\s*"
    r"|^[\s]*[\-\•·]\s*",
    re.MULTILINE,
)


def _is_non_ingredient(line: str) -> bool:
    return any(kw in line for kw in _NON_INGREDIENT_KEYWORDS)


def parse_from_main_fnctn(text: str) -> list[str]:
    """
    main_fnctn에서 [성분명] 패턴으로 성분명 추출.
    개별인정 번호, 괄호 부연설명 제거.
    """
    if not text:
        return []

    results = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        raw = match.group(1).strip()
        # 개별인정 번호 제거: (개별인정 제xxx호), (개별인정형 제xxx호)
        raw = re.sub(r"\(개별인정[형]?\s*제?\s*[\d\-]+호?\)", "", raw).strip()
        # 프로바이오틱스 등 복합물 괄호 제거
        raw = re.sub(r"\([^)]*\)", "", raw).strip()
        if raw and not _is_non_ingredient(raw):
            results.append(raw)

    return results


def parse_from_base_standard(text: str) -> list[str]:
    """
    base_standard에서 성분명 추출.
    비성분 행(성상, 중금속, 세균수 등) 필터링 후 성분명만 반환.
    """
    if not text:
        return []

    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_non_ingredient(line):
            continue

        # 번호/기호 제거 후 콜론 앞 부분만 추출 (콜론 뒤는 기준값)
        line = _NUMBERING.sub("", line).strip()
        if ":" in line:
            candidate = line.split(":")[0].strip()
        elif "：" in line:  # 전각 콜론
            candidate = line.split("：")[0].strip()
        else:
            continue  # 콜론 없는 줄은 성분 행이 아닐 가능성 높음

        # 괄호 안 단위/수치 제거 후 너무 짧거나 긴 건 스킵
        candidate = re.sub(r"\([^)]*\)", "", candidate).strip()
        if len(candidate) < 2 or len(candidate) > 40:
            continue
        if _is_non_ingredient(candidate):
            continue

        results.append(candidate)

    return results


def extract_ingredients(main_fnctn: str | None, base_standard: str | None) -> list[str]:
    """
    main_fnctn → base_standard 순으로 성분명 추출.
    main_fnctn에서 충분히 찾으면 base_standard는 보완용으로만 사용.
    중복 제거 후 반환.
    """
    from_main = parse_from_main_fnctn(main_fnctn or "")
    from_base = parse_from_base_standard(base_standard or "")

    seen: set[str] = set()
    merged: list[str] = []
    for name in from_main + from_base:
        if name not in seen:
            seen.add(name)
            merged.append(name)

    return merged
