from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"필수 환경변수 '{key}'가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return val

MYSQL_HOST = _require("MYSQL_HOST")
MYSQL_PORT = int(_require("MYSQL_PORT"))
MYSQL_DATABASE = _require("MYSQL_DATABASE")
MYSQL_USER = _require("MYSQL_USER")
MYSQL_PASSWORD = _require("MYSQL_PASSWORD")
CBNUAI_API_KEY = os.environ.get("CBNUAI_API_KEY", "")
