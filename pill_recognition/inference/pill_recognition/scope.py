from __future__ import annotations

import json
import re


def normalize_pill_id_set(pill_ids: list[str]) -> set[str]:
    return {str(pill_id).strip() for pill_id in pill_ids if str(pill_id).strip()}


def parse_allowed_pill_ids(values: list[str] | str | None) -> set[str]:
    if isinstance(values, str):
        values = [values]

    pill_ids = []
    for value in values or []:
        raw = str(value or "").strip()
        if not raw:
            continue
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                pill_ids.extend(str(item) for item in parsed)
                continue
        pill_ids.extend(part for part in re.split(r"[\s,]+", raw) if part)
    return normalize_pill_id_set(pill_ids)
