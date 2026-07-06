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

USE_BEST_LOCAL_RECOGNIZER="${PILL_USE_BEST_LOCAL_RECOGNIZER:-1}"
BEST_CLASSIFIER="$PWD/artifacts/classifier/aihub-resnet152-synthetic-layer4-none-v2-continued.pt"
BEST_DETECTOR="$PWD/artifacts/rtmdet-single-class/model-aihub-synthetic-v2.pth"
BEST_CLASSES="$PWD/artifacts/rtmdet-single-class/pill.yaml"

if [ "$USE_BEST_LOCAL_RECOGNIZER" != "0" ] && [ -f "$BEST_CLASSIFIER" ]; then
  export PILL_RECOGNIZER="aihub_classifier"
  export PILL_AIHUB_WEIGHTS="$BEST_CLASSIFIER"
  export PILL_AIHUB_CLASSIFIER_QUERY_PREPROCESS="${PILL_AIHUB_CLASSIFIER_QUERY_PREPROCESS:-none}"
  export PILL_CROP_PADDING_RATIO="${PILL_CROP_PADDING_RATIO:-0}"
fi

if [ "$USE_BEST_LOCAL_RECOGNIZER" != "0" ] && [ -f "$BEST_DETECTOR" ] && [ -f "$BEST_CLASSES" ]; then
  export PILL_DETECTOR_CHECKPOINT="$BEST_DETECTOR"
  export PILL_DETECTOR_CLASSES="$BEST_CLASSES"
fi

mode="${1:-api}"
case "$mode" in
  api)
    exec "$PYTHON_BIN" -m pill_recognition.api \
      --host "${PILL_API_HOST:-0.0.0.0}" \
      --port "${PILL_API_PORT:-8013}"
    ;;
  gradio)
    export PILL_GRADIO_HOST="${PILL_GRADIO_HOST:-127.0.0.1}"
    export PILL_GRADIO_PORT="${PILL_GRADIO_PORT:-7860}"
    exec "$PYTHON_BIN" -m pill_recognition.app
    ;;
  *)
    echo "Usage: $0 [api|gradio]" >&2
    exit 2
    ;;
esac
