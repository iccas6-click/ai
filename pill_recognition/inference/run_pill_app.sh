#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ -x ../.venv/bin/python ]; then
  PYTHON_BIN="../.venv/bin/python"
elif [ -x ../../.venv/bin/python ]; then
  PYTHON_BIN="../../.venv/bin/python"
else
  echo "Python virtualenv not found. Expected ../.venv or ../../.venv." >&2
  exit 1
fi

exec "$PYTHON_BIN" -m pill_recognition.app
