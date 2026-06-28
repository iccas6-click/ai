#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run bootstrap_server.sh before starting training." >&2
  exit 1
fi

mkdir -p training/runs
exec .venv/bin/python -m training.rtmdet_single_class.scripts.train "$@"
