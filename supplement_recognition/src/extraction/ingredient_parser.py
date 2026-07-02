"""
base_standard / main_fnctn 텍스트에서 성분명 리스트를 파싱.

전략:
- main_fnctn: [성분명] 브래킷 패턴 + 개별인정 괄호 제거 → 신뢰도 높음
- base_standard: 비성분 키워드 필터링 후 성분명 추출
  - 콜론 기준 분리 (기본)
  - 쉼표 복합 성분 분리 ("A, B 및 C의 합" 패턴)
  - 괄호 약어 추출 (콜론 없는 영문 성분행: "Docosahexaenoic acid (DHA), ...")
"""
from __future__ import annotations

import re

# base_standard에서 걸러낼 비성분 키워드
_NON_INGREDIENT_KEYWORDS = (
    "성상", "세균수", "대장균", "납", "카드뮴", "비소", "수은", "붕해", "용출",
    "총균수", "진균수", "메틸수은", "총비소", "잔류", "이물", "산가", "과산화물가",
    "수분", "회분", "조단백", "조지방", "총 플라보노이드", "CFU", "Plate Count",
    "Yeast", "Mould", "E. coli", "S.", "Salmonella", "spp.", "TYMC",
    "Bile tolerant gram negative bacteria", "Hexane", "헥산",
    "국문", "영문", "May help", "도움을 줄 수",
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

# "진세노사이드 Rg1, Rb1 및 Rg3의 합" 같은 복합 성분 패턴
# → 베이스 이름(진세노사이드)만 추출
_COMBINED_SUFFIX = re.compile(
    r"(\S+)\s+[A-Za-z0-9가-힣]+(?:[,，]\s*[A-Za-z0-9가-힣]+)+\s*(?:및|과|와|and)?\s*[A-Za-z0-9가-힣]+\s*의\s*합"
)

# "Docosahexaenoic acid (DHA)" 처럼 영문명 뒤 괄호 약어 패턴
_ABBREV_IN_PAREN = re.compile(r"\(([A-Z]{2,6})\)")


def _is_non_ingredient(line: str) -> bool:
    return any(kw in line for kw in _NON_INGREDIENT_KEYWORDS)


def _resolve_combined(candidate: str) -> list[str]:
    """
    "진세노사이드 Rg1, Rb1 및 Rg3의 합" → ["진세노사이드"]
    쉼표로 나열된 복합 성분에서 베이스 이름만 추출.
    """
    m = _COMBINED_SUFFIX.search(candidate)
    if m:
        return [m.group(1).strip()]
    if candidate.startswith("프로바이오틱스"):
        return ["프로바이오틱스"]
    return [candidate]


def _extract_abbrevs(line: str) -> list[str]:
    """
    "Docosahexaenoic acid (DHA), eicosapentaenoic acid (EPA)" 같은
    콜론 없는 영문 성분행에서 괄호 약어(DHA, EPA 등) 추출.
    """
    return _ABBREV_IN_PAREN.findall(line)


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
        # 나머지 괄호 부연설명 제거
        raw = re.sub(r"\([^)]*\)", "", raw).strip()
        if raw and not _is_non_ingredient(raw):
            results.append(raw)

    return results


def parse_from_base_standard(text: str) -> list[str]:
    """
    base_standard에서 성분명 추출.
    1. 비성분 행 필터링
    2. 콜론 기준 성분명 추출 + 복합 성분 분리
    3. 콜론 없는 영문 성분행: 괄호 약어 추출
    """
    if not text:
        return []

    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 번호/기호 제거
        line = _NUMBERING.sub("", line).strip()

        if ":" in line or "：" in line:
            sep = ":" if ":" in line else "："
            candidate = line.split(sep)[0].strip()

            # 케이스 2: 콜론 앞이 너무 길면 영문 약어 추출 시도
            # "Docosahexaenoic acid (DHA), eicosapentaenoic acid (EPA), ... : 표시량의 80%"
            if len(candidate) > 60:
                abbrevs = _extract_abbrevs(candidate)
                if abbrevs:
                    results.extend(abbrevs)
                continue

            # 괄호 안 단위/수치 제거
            candidate = re.sub(r"\([^)]*\)", "", candidate).strip()
            if len(candidate) < 2:
                continue
            if _is_non_ingredient(candidate):
                continue

            # 케이스 1: "진세노사이드 Rg1, Rb1 및 Rg3의 합" → ["진세노사이드"]
            results.extend(_resolve_combined(candidate))

        else:
            if _is_non_ingredient(line):
                continue
            # 케이스 2 (콜론 없는 영문 성분행)
            abbrevs = _extract_abbrevs(line)
            if abbrevs:
                results.extend(abbrevs)

    return results


def extract_ingredients(main_fnctn: str | None, base_standard: str | None) -> list[str]:
    """
    main_fnctn → base_standard 순으로 성분명 추출 후 중복 제거.
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
