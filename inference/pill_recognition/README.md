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

합성 multi-pill scene의 detector와 제품 인식을 함께 평가하려면 end-to-end 평가 스크립트를 사용합니다.

```bash
python -m pill_recognition.evaluate_pipeline_dataset \
  --dataset-root ../datasets/processed/rtmdet-aihub-synthetic-realistic-max10 \
  --split val \
  --limit 200 \
  --top-k 5 \
  --output outputs/evaluation/pipeline-realistic-val-200.json
```

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

## End-to-end synthetic scene 평가 결과

`rtmdet-aihub-synthetic-realistic-max10` val 200장 기준으로 detector는 안정적이지만, 합성 scene crop의 제품 인식은 낮습니다.

| Index | Detector F1 | Recognition Top-3 on matched | End-to-end Top-3 on GT | 판단 |
|---|---:|---:|---:|---|
| AI Hub ResNet152 prototype64, padding 0.12 | 0.98578 | 0.03558 | 0.03494 | 합성 crop 분포 불일치 |
| AI Hub ResNet152 prototype64, padding 0 | 0.98578 | 0.04615 | 0.04533 | padding 문제가 아님 |
| Augmented reference prototype32, padding 0 | 0.98578 | 0.07981 | 0.07838 | 개선되지만 부족 |
| Augmented reference16, padding 0 | 0.98578 | 0.09231 | 0.09065 | 개선되지만 부족 |

결론: 합성 scene은 RTMDet 단일 클래스 detector 학습/평가에는 유용하지만, 제품명 retrieval 성능 판단용으로는 부적합합니다. 합성 과정에서 누끼, 임의 회전, 조명/배경 합성이 AI Hub 원본 crop과 다른 분포를 만들기 때문입니다. 제품 인식 평가는 실제 스마트폰 사진 기반의 작은 검증셋으로 별도 구축해야 합니다.

합성 scene 스타일 reference index를 만들 수는 있지만 현재 기본값으로 쓰지 않습니다.

```bash
python -m pill_recognition.build_augmented_retrieval_index \
  --samples-per-class 16 \
  --index-mode reference \
  --output artifacts/retrieval/aihub_resnet_augmented_reference16.pt
```

## Real smartphone 평가셋

실서비스 판단은 실제 스마트폰 사진으로 해야 합니다. 권장 최소셋은 30~50장, 총 150~300알입니다.

```text
datasets/evaluation/real-smartphone/
├── images/
│   ├── IMG_0001.jpg
│   └── IMG_0002.jpg
└── annotations/
    ├── IMG_0001.json
    └── IMG_0002.json
```

annotation 포맷은 `real_eval_schema.example.json`을 따릅니다. `class_name`은 AIHub K-ID이고, `bbox_xyxy`는 원본 이미지 픽셀 좌표 `[x1, y1, x2, y2]`입니다. bbox가 있어야 detector와 제품 인식을 end-to-end로 분리 평가할 수 있습니다.

```json
{
  "image": "IMG_0001.jpg",
  "pills": [
    {
      "pill_id": 1,
      "class_name": "K-000000",
      "product_name": "제품명",
      "bbox_xyxy": [100, 120, 220, 260]
    }
  ]
}
```

평가:

```bash
python -m pill_recognition.evaluate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --top-k 5 \
  --output outputs/evaluation/real-smartphone.json
```

이 결과의 핵심 지표는 `detector_f1`, `recognition_top3_on_matched`, `end_to_end_top3_on_gt`입니다.

Gemini는 비교 실험용으로만 유지합니다.

```bash
export PILL_RECOGNIZER=gemini
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```
