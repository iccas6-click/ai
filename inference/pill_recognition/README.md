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

## Foundation embedding 비교 실험

AI Hub ResNet retrieval보다 나은 reference search encoder가 있는지 비교하기 위한 DINOv2 실험 스크립트입니다. 기본 앱 경로에는 연결하지 않고, 같은 AIHub held-out crop 평가셋으로 먼저 비교합니다.

```bash
python -m pill_recognition.build_foundation_index \
  --torchhub-repo facebookresearch/dinov2 \
  --torchhub-model dinov2_vits14 \
  --samples-per-class 16 \
  --index-mode prototype \
  --output artifacts/retrieval/dinov2_vits14_prototype16.pt

python -m pill_recognition.evaluate_foundation_retrieval \
  --index artifacts/retrieval/dinov2_vits14_prototype16.pt \
  --samples-per-class 4 \
  --offset 128 \
  --output outputs/evaluation/dinov2-vits14-prototype16-smoke.json
```

현재 서버 실험 결과 기준으로는 DINOv2를 운영 기본값으로 쓰지 않습니다.

| Encoder/index | 평가 범위 | Top-1 | Top-3 | Top-5 | 판단 |
|---|---:|---:|---:|---:|---|
| AI Hub ResNet152 prototype64 | 1000 classes x 4 crops | 0.81625 | 0.95725 | 0.98300 | 기본값 유지 |
| DINOv2 ViT-S/14 518px prototype16 | 1000 classes x 4 crops | 0.59925 | 0.76075 | 0.82825 | 제외 |
| DINOv2 ViT-S/14 518px reference16 | 1000 classes x 4 crops | 0.60350 | 0.76825 | 0.82950 | 제외 |

100클래스 smoke에서는 DINOv2 518px가 높게 보였지만, 전체 1000클래스 평가에서 성능이 크게 떨어졌습니다. 범용 image embedding보다 AI Hub 제공 ResNet152가 이 데이터 분포에 더 잘 맞습니다.

Gemini는 비교 실험용으로만 유지합니다.

```bash
export PILL_RECOGNIZER=gemini
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```
