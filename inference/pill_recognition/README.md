# Pill Recognition v2

RTMDet 탐지와 AI Hub 공식 ResNet152 feature retrieval을 사용하는 알약 인식 파이프라인입니다.

```text
image
-> RTMDet single-class detector
-> crop per pill
-> AI Hub ResNet152 feature embedding
-> cosine search against AI Hub 1000-class reference prototypes
-> Top-N product candidates with ingredients
```

기존 `RTMDet + AIHub ResNet152 + EfficientNet` baseline은 `inference/pill_recognition_legacy/`에 보존합니다.

## 실행

기본값은 외부 API 없이 실행되는 retrieval recognizer입니다. 실행 전 AI Hub reference index를 한 번 생성해야 합니다.

```bash
cd /home/gyuha_lee/pill/code/ai/inference
source ../.venv/bin/activate
python -m pill_recognition.build_retrieval_index --samples-per-class 32
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

Gemini는 비교 실험용으로만 유지합니다.

```bash
export PILL_RECOGNIZER=gemini
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```
