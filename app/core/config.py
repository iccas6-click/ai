from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

MYSQL_HOST = os.environ["MYSQL_HOST"]
MYSQL_PORT = int(os.environ["MYSQL_PORT"])
MYSQL_DATABASE = os.environ["MYSQL_DATABASE"]
MYSQL_USER = os.environ["MYSQL_USER"]
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
CBNUAI_API_KEY = os.environ["CBNUAI_API_KEY"]
