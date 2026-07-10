from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import mysql.connector
from openai import OpenAI

_GATEWAY_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
_MODEL_PRIMARY = "gemini-3.5-flash"
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MODEL_FALLBACK = "gemini-2.0-flash"

_MAX_RETRIES = 2
_RETRY_DELAY = 1.0
_CONFIDENCE_THRESHOLD = 0.72

_PROMPT = """\
이 이미지는 처방전, 조제약 봉투, 복약 안내문, 약국 영수증, 안약(점안액) 등일 수 있습니다.
이미지에 적힌 의약품 정보를 읽어서 약품명을 구조화해 주세요.

반드시 아래 JSON 형식만 반환하세요. 설명, 마크다운, 코드블록은 금지합니다.
{
  "document_type": "prescription|medicine_bag|medication_guide|receipt|eye_drop|unknown",
  "medications": [
    {
      "product_name": "이미지에 적힌 약품명",
      "dosage": "제품 자체의 함량/규격만. 예: 50mg, 250밀리그램, 10mg. 없으면 빈 문자열",
      "administration": "사용법만. 예: 1일 3회 식후 30분. 없으면 빈 문자열",
      "ingredient_names": ["확실한 주성분명 후보. 제품명을 그대로 반복하지 마세요"],
      "confidence": 0.0
    }
  ],
  "warnings": []
}

규칙:
- 안약(점안액), 점이액, 외용제, 흡입제 등 경구 복용이 아닌 의약품이면 document_type을 "eye_drop"으로 설정합니다.
- 약품명은 처방/조제된 의약품만 포함합니다.
- 병원명, 약국명, 환자명, 의사명, 날짜, 금액, 보험/청구 문구는 제외합니다.
- "1일 3회", "7일분"처럼 사용법만 있고 약 이름이 아닌 줄은 약품명에서 제외합니다.
- dosage에는 "1일", "하루", "식전", "식후", "아침", "점심", "저녁", "취침", "복용" 같은 복용법 표현을 절대 넣지 않습니다.
- 복용법 표현은 administration에만 넣습니다.
- 같은 약이 중복으로 보이면 하나로 합칩니다.
- 제품명 일부가 애매하면 보이는 그대로 적고 confidence를 낮춥니다.
- 성분명이 이미지에 없더라도 제품명으로 주성분을 확실히 알 수 있으면 ingredient_names에 주성분 후보를 넣습니다.
- 주성분이 확실하지 않으면 ingredient_names는 빈 배열로 둡니다.
- 한글 텍스트를 우선 사용합니다.
"""

_EXTERNAL_USE_DOC_TYPES = frozenset({"eye_drop"})

_INGREDIENT_PROMPT = """\
다음은 한국 처방전/약봉투 OCR에서 추출된 의약품 제품명입니다.
각 제품명의 주성분 후보를 한국어 일반명/성분명으로 찾아 JSON만 반환하세요.

형식:
{
  "items": [
    {"product_name": "입력 제품명", "ingredient_names": ["주성분명"], "confidence": 0.0}
  ]
}

규칙:
- 제품명, 회사명, 제형명(정/캡슐/서방정 등)을 성분명으로 반복하지 마세요.
- 제품명으로 주성분을 확실히 알 수 있는 경우만 성분명을 넣으세요.
- 확실하지 않으면 ingredient_names를 빈 배열로 두고 confidence를 낮게 주세요.
- 한글 성분명을 우선 사용하세요.
"""

_ADMINISTRATION_PATTERN = re.compile(
    r"(1일|하루|매일|매주|식전|식후|식간|아침|점심|저녁|취침|공복|복용|투여|"
    r"씩|마다|회|일분|일간|일수|분복|필요시)",
)
_STRENGTH_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|g|mcg|μg|ug|㎎|㎍|iu|IU|ml|mL|밀리그램|마이크로그램|그램|밀리리터)",
    flags=re.IGNORECASE,
)


def _normalize_match_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("밀리그램", "mg")
    normalized = normalized.replace("마이크로그램", "mcg")
    normalized = normalized.replace("그램", "g")
    normalized = normalized.replace("밀리리터", "ml")
    return re.sub(r"[\s\-_()/·ㆍ.,]+", "", normalized)


def _clean_unique(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean:
            continue
        key = _normalize_match_key(clean)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(clean)
    return cleaned


def _clean_official_ingredient_name(value: str | None) -> str:
    return re.sub(r"\[[A-Za-z]\d{3,}\]", "", str(value or "")).strip()


def _lookup_mode() -> str:
    return os.environ.get("PRESCRIPTION_DRUG_LOOKUP_MODE", "cache").strip().lower()


def _product_lookup_keys(value: str) -> list[str]:
    compact = _normalize_match_key(value)
    without_units = re.sub(
        r"\d+(\.\d+)?(mg|g|mcg|μg|ug|iu|ml|㎎|㎍)?",
        "",
        compact,
        flags=re.IGNORECASE,
    )
    without_form = re.sub(r"(서방정|장용정|필름코팅정|연질캡슐|캡슐|정|시럽|액)$", "", without_units)
    return _clean_unique([compact, without_units, without_form])


def _strip_strength_from_name(product_name: str, strength: str) -> str:
    if not product_name or not strength:
        return product_name.strip()
    pattern = re.escape(strength).replace(r"\ ", r"\s*")
    return re.sub(pattern, "", product_name, flags=re.IGNORECASE).strip()


def _split_dosage_and_administration(product_name: str, raw_dosage: Any, raw_administration: Any) -> tuple[str, str]:
    dosage_text = str(raw_dosage or "").strip()
    administration_text = str(raw_administration or "").strip()
    administration_parts: list[str] = []

    if administration_text:
        administration_parts.append(administration_text)
    if dosage_text and _ADMINISTRATION_PATTERN.search(dosage_text):
        administration_parts.append(dosage_text)

    strength_candidates = _clean_unique(_STRENGTH_PATTERN.findall(dosage_text) + _STRENGTH_PATTERN.findall(product_name))
    dosage = strength_candidates[0] if strength_candidates else ""

    if not dosage and dosage_text and not _ADMINISTRATION_PATTERN.search(dosage_text) and len(dosage_text) <= 8:
        dosage = dosage_text
    if not dosage:
        suffix_number = re.search(r"(?:정|캡슐|캡|서방정|오로스정)(\d+(?:\.\d+)?)$", product_name)
        if suffix_number:
            dosage = suffix_number.group(1)

    administration = " / ".join(_clean_unique(administration_parts))
    return dosage.strip(), administration.strip()


def _connect_db():
    return mysql.connector.connect(
        host=os.environ.get("PILL_MYSQL_HOST", os.environ.get("MYSQL_HOST", "localhost")),
        port=int(os.environ.get("PILL_MYSQL_PORT", os.environ.get("MYSQL_PORT", "3306"))),
        database=os.environ.get("PILL_MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", "click_db")),
        user=os.environ.get("PILL_MYSQL_USER", os.environ.get("MYSQL_USER", "click_user")),
        password=os.environ.get("PILL_MYSQL_PASSWORD", os.environ.get("MYSQL_PASSWORD", "")),
    )


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name = %s
        """,
        (table_name,),
    )
    row = cursor.fetchone()
    return bool(row and row.get("count") > 0)


def _build_messages(image_data: str, mime_type: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
            ],
        }
    ]


def _call_with_retry(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> str:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * (2**attempt))
    raise last_exc  # type: ignore[misc]


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object.")
    return parsed


def _read_document_with_llm(image_path: Path) -> dict[str, Any]:
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    ext = image_path.suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    messages = _build_messages(image_data, f"image/{ext}")

    primary_key = os.environ.get("CBNUAI_API_KEY", "")
    if primary_key:
        try:
            client = OpenAI(api_key=primary_key, base_url=_GATEWAY_BASE_URL)
            return _extract_json_object(_call_with_retry(client, _MODEL_PRIMARY, messages))
        except Exception:
            pass

    fallback_key = os.environ.get("GEMINI_API_KEY", "")
    if fallback_key:
        client = OpenAI(api_key=fallback_key, base_url=_GOOGLE_BASE_URL)
        return _extract_json_object(_call_with_retry(client, _MODEL_FALLBACK, messages))

    raise RuntimeError("CBNUAI_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")


def _read_ingredients_with_llm(product_names: list[str]) -> dict[str, list[str]]:
    names = _clean_unique(product_names)
    if not names:
        return {}
    messages = [
        {
            "role": "user",
            "content": f"{_INGREDIENT_PROMPT}\n\n제품명 목록:\n" + "\n".join(f"- {name}" for name in names),
        }
    ]

    parsed: dict[str, Any] | None = None
    primary_key = os.environ.get("CBNUAI_API_KEY", "")
    if primary_key:
        try:
            client = OpenAI(api_key=primary_key, base_url=_GATEWAY_BASE_URL)
            parsed = _extract_json_object(_call_with_retry(client, _MODEL_PRIMARY, messages))
        except Exception:
            parsed = None

    if parsed is None:
        fallback_key = os.environ.get("GEMINI_API_KEY", "")
        if not fallback_key:
            return {}
        try:
            client = OpenAI(api_key=fallback_key, base_url=_GOOGLE_BASE_URL)
            parsed = _extract_json_object(_call_with_retry(client, _MODEL_FALLBACK, messages))
        except Exception:
            return {}

    items = parsed.get("items") if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        return {}

    inferred: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        product_name = str(item.get("product_name") or "").strip()
        try:
            confidence = float(item.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0
        ingredients = _clean_unique([str(value) for value in item.get("ingredient_names") or []])
        if product_name and ingredients and confidence >= 0.7:
            inferred[_normalize_match_key(product_name)] = ingredients
    return inferred


def _lookup_official_product_ingredients(cursor, product_name: str) -> tuple[list[str], str, str | None, dict[str, str]]:
    if not _table_exists(cursor, "official_drug_products") or not _table_exists(
        cursor,
        "official_drug_product_ingredients",
    ):
        return [], "not_found", None, {}

    keys = _product_lookup_keys(product_name)
    if not keys:
        return [], "not_found", None, {}

    like_params = [f"%{key}%" for key in keys[:4]]
    while len(like_params) < 4:
        like_params.append("__NO_MATCH__")

    cursor.execute(
        """
        SELECT cde.canonical_name_ko,
               cde.canonical_name_en,
               odpi.ingredient_name,
               odp.product_image_url,
               odp.efficacy_text,
               odp.use_method_text,
               odp.warning_text,
               odp.interaction_text,
               odp.side_effect_text,
               odp.storage_text
        FROM official_drug_products odp
        LEFT JOIN official_drug_product_ingredients odpi ON odpi.item_seq = odp.item_seq
        LEFT JOIN canonical_drug_entities cde ON odpi.canonical_drug_id = cde.canonical_drug_id
        WHERE odp.product_name = %s
           OR odp.normalized_product_name = %s
           OR odp.normalized_product_name LIKE %s
           OR odp.normalized_product_name LIKE %s
           OR odp.normalized_product_name LIKE %s
           OR odp.normalized_product_name LIKE %s
        ORDER BY odp.updated_at DESC, odpi.id
        LIMIT 12
        """,
        (product_name, keys[0], *like_params),
    )
    rows = cursor.fetchall()
    ingredients = _clean_unique(
        [
            _clean_official_ingredient_name(
                row.get("canonical_name_ko")
                or row.get("canonical_name_en")
                or row.get("ingredient_name")
            )
            for row in rows
        ]
    )
    image_url = next((row.get("product_image_url") for row in rows if row.get("product_image_url")), None)
    info_row = rows[0] if rows else {}
    drug_info = {
        "efficacy": str(info_row.get("efficacy_text") or "").strip(),
        "use_method": str(info_row.get("use_method_text") or "").strip(),
        "warning": str(info_row.get("warning_text") or "").strip(),
        "interaction": str(info_row.get("interaction_text") or "").strip(),
        "side_effect": str(info_row.get("side_effect_text") or "").strip(),
        "storage": str(info_row.get("storage_text") or "").strip(),
    }
    drug_info = {key: value for key, value in drug_info.items() if value}
    return ingredients, "official_product_catalog" if rows else "not_found", image_url, drug_info


def _build_official_product_import_command(product_name: str) -> tuple[list[str], Path, dict[str, str]] | None:
    explicit_path = os.environ.get("OFFICIAL_DRUG_IMPORTER_PATH", "").strip()
    importer_path = Path(explicit_path) if explicit_path else (
        Path(__file__).resolve().parents[2].parent / "backend" / "scripts" / "import_official_drug_products.py"
    )
    if not importer_path.exists():
        return None

    backend_dir = importer_path.parents[1]
    python_path = backend_dir / ".venv" / "bin" / "python"
    if not python_path.exists():
        python_path = Path(sys.executable)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for source, target in (
        ("PILL_MYSQL_HOST", "MYSQL_HOST"),
        ("PILL_MYSQL_PORT", "MYSQL_PORT"),
        ("PILL_MYSQL_DATABASE", "MYSQL_DATABASE"),
        ("PILL_MYSQL_USER", "MYSQL_USER"),
        ("PILL_MYSQL_PASSWORD", "MYSQL_PASSWORD"),
    ):
        if env.get(source):
            env[target] = env[source]

    command = [
        str(python_path),
        str(importer_path),
        "--query",
        product_name,
        "--timeout",
        os.environ.get("OFFICIAL_DRUG_IMPORT_TIMEOUT", "3"),
        "--sleep",
        "0",
    ]
    return command, backend_dir, env


def _run_official_product_import(product_name: str, *, wait: bool) -> bool:
    """공식 제품 캐시 miss 시 importer를 호출한다. 기본은 응답 지연을 피하기 위해 백그라운드 실행."""
    command_config = _build_official_product_import_command(product_name)
    if command_config is None:
        return False
    command, backend_dir, env = command_config

    if not wait:
        try:
            subprocess.Popen(
                command,
                cwd=str(backend_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            return False
        return False

    try:
        result = subprocess.run(
            command,
            cwd=str(backend_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("OFFICIAL_DRUG_IMPORT_SUBPROCESS_TIMEOUT", "9")),
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _official_import_blocking_enabled() -> bool:
    return os.environ.get("OFFICIAL_DRUG_IMPORT_BLOCKING", "").strip().lower() in {"1", "true", "yes", "on"}


def _official_import_on_demand_enabled() -> bool:
    return os.environ.get("OFFICIAL_DRUG_IMPORT_ON_DEMAND", "").strip().lower() in {"1", "true", "yes", "on"}


def _lookup_legacy_product_ingredients(cursor, product_name: str) -> tuple[list[str], str, str | None, dict[str, str]]:
    keys = _product_lookup_keys(product_name)
    if not keys:
        return [], "not_found", None, {}

    like_params = [f"%{key}%" for key in keys[:4]]
    while len(like_params) < 4:
        like_params.append("__NO_MATCH__")

    try:
        cursor.execute(
            """
            SELECT cde.canonical_drug_name_ko, cde.canonical_drug_name_en, ppi.ingredient_name
            FROM pill_products pp
            JOIN pill_product_ingredients ppi ON ppi.pill_product_id = pp.pill_product_id
            JOIN canonical_drug_entities cde ON ppi.canonical_drug_id = cde.canonical_drug_id
            WHERE pp.product_name = %s
               OR pp.product_name_normalized = %s
               OR pp.product_name_normalized LIKE %s
               OR pp.product_name_normalized LIKE %s
               OR pp.product_name_normalized LIKE %s
               OR pp.product_name_normalized LIKE %s
            LIMIT 12
            """,
            (product_name, keys[0], *like_params),
        )
    except Exception:
        return [], "not_found", None, {}
    rows = cursor.fetchall()
    ingredients = _clean_unique(
        [
            row.get("canonical_drug_name_ko") or row.get("canonical_drug_name_en") or row.get("ingredient_name")
            for row in rows
        ]
    )
    return ingredients, "legacy_product_table" if ingredients else "not_found", None, {}


def _lookup_product_ingredients(cursor, product_name: str) -> tuple[list[str], str, str | None, dict[str, str]]:
    official_ingredients, official_match_type, official_image_url, official_info = _lookup_official_product_ingredients(
        cursor,
        product_name,
    )
    if (
        not official_ingredients
        and _official_import_on_demand_enabled()
        and _run_official_product_import(product_name, wait=_official_import_blocking_enabled())
    ):
        try:
            cursor.execute("COMMIT")
        except Exception:
            pass
        official_ingredients, official_match_type, official_image_url, official_info = _lookup_official_product_ingredients(
            cursor,
            product_name,
        )
    if official_ingredients:
        return official_ingredients, official_match_type, official_image_url, official_info

    legacy_ingredients, legacy_match_type, _, _ = _lookup_legacy_product_ingredients(cursor, product_name)
    if legacy_ingredients:
        return legacy_ingredients, legacy_match_type, official_image_url, official_info
    return [], official_match_type, official_image_url, official_info


def _lookup_canonical_ingredients(cursor, names: list[str]) -> list[str]:
    resolved: list[str] = []
    for name in _clean_unique(names):
        normalized = _normalize_match_key(name)
        try:
            cursor.execute(
                """
                SELECT cde.canonical_drug_name_ko, cde.canonical_drug_name_en
                FROM canonical_drug_entities cde
                LEFT JOIN drug_aliases da ON da.canonical_drug_id = cde.canonical_drug_id
                WHERE cde.canonical_drug_name_ko = %s
                   OR LOWER(cde.canonical_drug_name_en) = LOWER(%s)
                   OR REPLACE(REPLACE(REPLACE(LOWER(cde.canonical_drug_name_ko), ' ', ''), '-', ''), '.', '') = %s
                   OR REPLACE(REPLACE(REPLACE(LOWER(cde.canonical_drug_name_en), ' ', ''), '-', ''), '.', '') = %s
                   OR da.alias_name_normalized LIKE %s
                LIMIT 4
                """,
                (name, name, normalized, normalized, f"%{normalized}%"),
            )
        except Exception:
            resolved.append(name)
            continue
        rows = cursor.fetchall()
        if rows:
            resolved.extend(row.get("canonical_drug_name_ko") or row.get("canonical_drug_name_en") for row in rows)
        else:
            resolved.append(name)
    return _clean_unique(resolved)


def _enrich_medications(raw_medications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    conn = None
    cursor = None
    try:
        conn = _connect_db()
        cursor = conn.cursor(dictionary=True)
        enriched: list[dict[str, Any]] = []
        unresolved_for_llm: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_medications):
            item_started_at = time.perf_counter()
            product_name = str(raw.get("product_name") or raw.get("name") or "").strip()
            if not product_name:
                continue

            llm_ingredients = _clean_unique([str(value) for value in raw.get("ingredient_names") or []])
            db_ingredients, match_type, image_url, drug_info = _lookup_product_ingredients(cursor, product_name)
            ingredients = db_ingredients or _lookup_canonical_ingredients(cursor, llm_ingredients)
            dosage, administration = _split_dosage_and_administration(
                product_name,
                raw.get("dosage"),
                raw.get("administration") or raw.get("usage") or raw.get("directions"),
            )
            confidence = raw.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.6 if ingredients else 0.4

            item = {
                "id": f"rx-{index}",
                "product_name": product_name,
                "name": product_name,
                "dosage": dosage,
                "administration": administration,
                "ingredients": ingredients,
                "analysis_names": ingredients or [product_name],
                "image_url": image_url,
                "product_image_url": image_url,
                "drug_info": drug_info,
                "confidence": max(0.0, min(confidence_value, 1.0)),
                "match_type": match_type if db_ingredients else ("ingredient_text" if ingredients else "not_found"),
                "needs_confirmation": not ingredients or confidence_value < _CONFIDENCE_THRESHOLD,
            }
            if not ingredients:
                unresolved_for_llm.append(item)
            enriched.append(item)
            print(
                "[prescription] enrich_item "
                f"product={product_name} match={item['match_type']} "
                f"ingredients={len(ingredients)} elapsed={time.perf_counter() - item_started_at:.2f}s",
                flush=True,
            )

        if unresolved_for_llm:
            llm_started_at = time.perf_counter()
            inferred = _read_ingredients_with_llm([item["product_name"] for item in unresolved_for_llm])
            print(
                "[prescription] ingredient_llm "
                f"items={len(unresolved_for_llm)} elapsed={time.perf_counter() - llm_started_at:.2f}s",
                flush=True,
            )
            for item in unresolved_for_llm:
                inferred_ingredients = inferred.get(_normalize_match_key(item["product_name"]), [])
                if not inferred_ingredients:
                    continue
                ingredients = _lookup_canonical_ingredients(cursor, inferred_ingredients)
                item["ingredients"] = ingredients
                item["analysis_names"] = ingredients or inferred_ingredients
                item["match_type"] = "llm_product_ingredient_candidate"
                item["needs_confirmation"] = True
        print(
            f"[prescription] enrich_total items={len(enriched)} elapsed={time.perf_counter() - started_at:.2f}s",
            flush=True,
        )
        return enriched
    except Exception:
        return [
            {
                "id": f"rx-{index}",
                "product_name": str(raw.get("product_name") or raw.get("name") or "").strip(),
                "name": str(raw.get("product_name") or raw.get("name") or "인식된 처방약").strip(),
                "dosage": _split_dosage_and_administration(
                    str(raw.get("product_name") or raw.get("name") or ""),
                    raw.get("dosage"),
                    raw.get("administration") or raw.get("usage") or raw.get("directions"),
                )[0],
                "administration": _split_dosage_and_administration(
                    str(raw.get("product_name") or raw.get("name") or ""),
                    raw.get("dosage"),
                    raw.get("administration") or raw.get("usage") or raw.get("directions"),
                )[1],
                "ingredients": _clean_unique([str(value) for value in raw.get("ingredient_names") or []]),
                "analysis_names": _clean_unique([str(value) for value in raw.get("ingredient_names") or []])
                or [str(raw.get("product_name") or raw.get("name") or "").strip()],
                "image_url": None,
                "product_image_url": None,
                "drug_info": {},
                "confidence": float(raw.get("confidence") or 0.4),
                "match_type": "llm_only",
                "needs_confirmation": True,
            }
            for index, raw in enumerate(raw_medications)
            if str(raw.get("product_name") or raw.get("name") or "").strip()
        ]
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def _to_detection_compat(medication: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "pill_id": medication["id"],
                "product_name": medication["product_name"],
                "ingredient": ", ".join(medication.get("ingredients") or []),
                "score": round(float(medication.get("confidence") or 0) * 100, 2),
                "reference_image_url": medication.get("image_url") or medication.get("product_image_url"),
            }
        ]
    }


def recognize_prescription_document(image_path: Path | str, request_id: str | None = None) -> dict[str, Any]:
    started_at = time.perf_counter()
    rid = request_id or f"rec_pill_doc_{uuid.uuid4().hex[:8]}"
    path = Path(image_path)
    read_started_at = time.perf_counter()
    parsed = _read_document_with_llm(path)
    print(
        f"[prescription] read_document document_type={parsed.get('document_type') or 'unknown'} "
        f"raw_items={len(parsed.get('medications') or [])} elapsed={time.perf_counter() - read_started_at:.2f}s",
        flush=True,
    )
    raw_medications = parsed.get("medications") or []
    if not isinstance(raw_medications, list):
        raw_medications = []

    doc_type = parsed.get("document_type") or "unknown"
    if doc_type in _EXTERNAL_USE_DOC_TYPES:
        medications = [
            {
                "id": f"rx-{i}",
                "product_name": str(item.get("product_name") or "").strip(),
                "name": str(item.get("product_name") or "").strip(),
                "dosage": "",
                "administration": str(item.get("administration") or "").strip(),
                "ingredients": [],
                "analysis_names": [],
                "image_url": None,
                "product_image_url": None,
                "drug_info": {},
                "confidence": float(item.get("confidence") or 0.5),
                "match_type": "external_use",
                "needs_confirmation": True,
            }
            for i, item in enumerate(raw_medications)
            if isinstance(item, dict) and str(item.get("product_name") or "").strip()
        ]
    else:
        medications = _enrich_medications([item for item in raw_medications if isinstance(item, dict)])
    status = "completed" if medications and not any(item["needs_confirmation"] for item in medications) else "needs_confirmation"
    if not medications:
        status = "failed"

    warnings = [str(value) for value in parsed.get("warnings") or [] if str(value).strip()]
    if status == "failed":
        warnings.append("처방전 또는 약봉투에서 약품명을 찾지 못했습니다.")

    print(
        f"[prescription] total request_id={rid} status={status} "
        f"items={len(medications)} elapsed={time.perf_counter() - started_at:.2f}s",
        flush=True,
    )
    return {
        "request_id": rid,
        "status": status,
        "document_type": parsed.get("document_type") or "unknown",
        "medications": medications,
        "detections": [_to_detection_compat(item) for item in medications],
        "needs_confirmation": status != "completed",
        "warnings": warnings,
    }
