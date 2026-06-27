# Pill Recognition v2

RTMDet 탐지와 AI Hub 공식 ResNet152 feature retrieval을 사용하는 알약 인식 파이프라인입니다.

```text
image
-> RTMDet single-class detector
-> crop per pill
-> AI Hub ResNet152 feature embedding
-> cosine search against AI Hub 1000-class reference prototypes
-> optional metadata rerank using estimated color/shape
-> Top-N product candidates with ingredients
```

기존 `RTMDet + AIHub ResNet152 + EfficientNet` baseline은 `inference/pill_recognition_legacy/`에 보존합니다.

## 실행

기본값은 외부 API 없이 실행되는 retrieval recognizer입니다. 실행 전 AI Hub reference index를 한 번 생성해야 합니다.

```bash
cd /home/gyuha_lee/pill/code/ai/inference
source ../.venv/bin/activate
python -m pill_recognition.build_retrieval_index --samples-per-class 32 --index-mode prototype
```

```bash
cd /home/gyuha_lee/pill/code/ai/inference
source ../.venv/bin/activate
python -m pill_recognition.app
```

평가:

```bash
python -m pill_recognition.evaluate_retrieval \
  --samples-per-class 8 \
  --offset 64 \
  --output outputs/evaluation/retrieval-aihub-resnet.json
```

`--index-mode reference`는 sampled reference embedding을 모두 저장하는 비교 실험용 옵션입니다. 현재 AIHub held-out crop 기준으로는 prototype 평균 인덱스가 더 안정적입니다.

색상/형상 기반 메타데이터 재랭킹은 실제 스마트폰 사진 평가셋에서 A/B 비교하기 위한 선택 기능입니다. AIHub held-out crop 기준에서는 기본 retrieval이 더 안정적이라 기본값은 off입니다.

```bash
export PILL_RETRIEVAL_METADATA_RERANK=1
```

Gemini는 비교 실험용으로만 유지합니다.

```bash
export PILL_RECOGNIZER=gemini
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```
