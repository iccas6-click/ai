# Pill Recognition v2

RTMDet 탐지와 AI Hub 공식 ResNet152 feature retrieval을 사용하는 알약 인식 파이프라인입니다.

```text
image
-> RTMDet single-class detector
-> crop per pill
-> AI Hub ResNet152 feature embedding
-> cosine search against AI Hub 1000-class reference prototypes
-> optional metadata rerank using estimated color/shape
-> Top-3 product candidates with ingredients and review status
```

기존 `RTMDet + AIHub ResNet152 + EfficientNet` baseline은 `inference/pill_recognition_legacy/`에 보존합니다.

앱 연동 시 호출 순서와 status별 사용자 흐름은 [`SERVICE_FLOW.md`](./SERVICE_FLOW.md)를 따릅니다.

## 실행

기본값은 외부 API 없이 실행되는 retrieval recognizer입니다. 실행 전 AI Hub reference index를 한 번 생성해야 합니다.

서비스 기본 응답은 알약별 제품 후보 Top-3입니다. 후보가 없으면 `no_candidate`, 점수가 낮으면 `low_confidence`, 1·2위 후보가 붙어 있으면 `ambiguous`, 그 외에는 `needs_confirmation`으로 반환합니다. 모든 상태는 최종 복용 전 사용자 확인이 필요하다는 전제를 유지합니다.

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

백엔드 연동용 HTTP API:

```bash
cd /home/gyuha_lee/pill/code/ai/inference
source ../.venv/bin/activate
python -m pill_recognition.api --host 0.0.0.0 --port 8001
```

API 서버는 기본적으로 startup 시 pipeline warmup을 수행합니다. retrieval 모델/index와 detector를 미리 로드해 첫 사용자 요청 지연을 줄입니다. 필요하면 끌 수 있습니다.

```bash
export PILL_WARMUP_ON_STARTUP=0
```

```bash
curl -X POST http://127.0.0.1:8001/recognize \
  -F "file=@sample.jpg"
```

응답은 `RecognitionResult.to_dict()`와 같은 JSON이며, 알약별 `vision.color`, `vision.shape`, `candidates`, `status`, `status_reason`을 포함합니다. `warnings`에는 낮은 해상도, 어두움, 과노출, 낮은 대비, 흔들림처럼 재촬영이 필요한 입력 품질 문제가 포함됩니다. `timings_ms`에는 파이프라인 내부 `quality`, `detector`, `recognition`, `postprocess`, `total`과 API 레벨 `upload_decode`, `pipeline_get`, `pipeline_call`, `api_total` latency가 포함됩니다. Warmup이 정상 완료되면 일반 요청의 `pipeline_get`은 매우 작아야 합니다.

선택한 알약 crop 단독 인식 API:

```bash
curl -X POST http://127.0.0.1:8001/crops/recognize \
  -F "file=@pill_crop.jpg"
```

이 endpoint는 RTMDet 탐지를 다시 돌리지 않고 업로드된 crop을 바로 AIHub retrieval에 넣습니다. 앱에서는 사용자가 특정 알약을 선택한 뒤 반대면을 추가 촬영하거나, 프론트에서 이미 잘라낸 crop을 재확인할 때 사용합니다.

여러 crop batch 인식 API:

```bash
curl -X POST http://127.0.0.1:8001/crops/recognize-batch \
  -F "files=@front_crop.jpg" \
  -F "files=@back_crop.jpg"
```

이 endpoint는 여러 crop을 한 번의 batch로 AIHub retrieval에 넣습니다. 앞/뒷면 crop이나 여러 선택 알약을 재확인할 때 HTTP 왕복과 모델 호출 부담을 줄일 수 있습니다. 기본 최대 crop 수는 `PILL_MAX_BATCH_CROPS=12`이고, 이미지 1장 기본 제한은 `PILL_MAX_UPLOAD_BYTES=10485760`, `PILL_MAX_IMAGE_PIXELS=12000000`입니다.

crop batch 응답의 `warnings`는 `crop 1`, `crop 2`처럼 crop 번호를 포함합니다. 프론트는 해당 crop만 다시 찍도록 안내할 수 있습니다.

crop batch의 `timings_ms`는 파이프라인 내부 `preprocess`, `recognition`, `postprocess`, `total`과 API 레벨 `upload_decode`, `pipeline_get`, `pipeline_call`, `api_total`을 반환합니다. 여러 crop은 한 번의 retrieval batch로 처리되므로 앱에서는 crop을 따로 여러 번 호출하지 말고 batch endpoint를 우선 사용합니다.

각인/색/모양/텍스트 보정 검색 API:

```bash
curl "http://127.0.0.1:8001/products/search?imprint=W2&shape=원형&color=하양&limit=5"
```

이 endpoint는 AIHub 제품 DB를 직접 검색합니다. 앱에서는 인식 후보가 애매할 때 사용자가 읽은 각인, 앞/뒤면 촬영 결과, 또는 OCR 결과를 넣어 후보를 다시 좁히는 데 사용합니다.

제품 후보의 `reference_image_url`은 AIHub reference crop 이미지입니다. 후보 확인 UI에서 제품명/성분과 함께 표시할 수 있습니다.

```bash
curl http://127.0.0.1:8001/products/K-000001/reference-image \
  --output reference.png
```

인식 후보 보정/재정렬 API:

```bash
curl -X POST http://127.0.0.1:8001/products/refine \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [
      {"pill_id": "K-001732", "score": 55.0, "source": "aihub_resnet_retrieval", "view": "front"},
      {"pill_id": "K-001732", "score": 76.0, "source": "aihub_resnet_retrieval", "view": "back"},
      {"pill_id": "K-012914", "score": 92.0, "source": "aihub_resnet_retrieval", "view": "front"}
    ],
    "imprint": "W2",
    "shape": "원형",
    "color": "하양",
    "limit": 3
  }'
```

이 endpoint는 이미지 recognition 후보의 `score`와 AIHub 제품 DB의 각인/색/모양/텍스트 점수를 합산해 다시 정렬합니다. 같은 `pill_id`가 앞/뒷면 crop에서 반복 등장하면 `image_evidence_count`, `views`를 기록하고 작은 multi-view 보너스를 부여합니다. 응답에는 `status`, `status_reason`이 포함되며, 이미지가 헷갈린 경우에도 각인 exact match가 있으면 후보 순위가 앞으로 올라옵니다.

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

## Query crop 전처리 실험

RTMDet crop에 배경이 섞이는 실제 촬영 상황을 대비해 retrieval 직전 crop을 전경 중심으로 다시 자르는 실험 옵션을 제공합니다.

```bash
export PILL_RETRIEVAL_QUERY_PREPROCESS=foreground
# or
export PILL_RETRIEVAL_QUERY_PREPROCESS=foreground_dark
```

현재 운영 기본값은 `none`입니다. AIHub held-out crop 1000클래스 x 2장 smoke 기준으로 전경 재합성은 오히려 성능을 낮췄습니다.

| Query preprocess | Top-1 | Top-3 | Top-5 | 판단 |
|---|---:|---:|---:|---|
| `none` | 0.8150 | 0.9500 | 0.9795 | 기본값 유지 |
| `foreground` | 0.8060 | 0.9420 | 0.9725 | 기본값 제외 |
| `foreground_dark` | 0.2685 | 0.4165 | 0.4995 | 기본값 제외 |

이 결과는 “전경 분리 자체가 항상 좋다”가 아니라, AIHub ResNet152가 학습/평가된 crop 분포에 민감하다는 의미입니다. 실제 스마트폰 검증셋이 쌓이면 이 옵션을 다시 A/B 평가합니다.

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

이미지만 먼저 넣어둔 상태라면 detector와 retrieval 결과로 annotation 초안을 만들 수 있습니다.

```bash
python -m pill_recognition.draft_real_annotations \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --top-k 5
```

초안의 `class_name`, `product_name`, `bbox_xyxy`는 반드시 사람이 확인해야 합니다. `candidate_hints`에는 현재 pipeline의 후보가 들어가며, 맞는 후보가 없으면 AIHub 제품 DB 검색 탭이나 원본 AIHub K-ID 목록으로 확인합니다.

평가:

```bash
python -m pill_recognition.evaluate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --top-k 5 \
  --output outputs/evaluation/real-smartphone.json
```

이 결과의 핵심 지표는 `detector_f1`, `recognition_top3_on_matched`, `end_to_end_top3_on_gt`, `mean_total_ms`, `p95_total_ms`입니다.

결과 JSON에는 `analysis` 섹션도 포함됩니다. 이 섹션은 다음 케이스를 따로 모아 실제 개선 우선순위를 정하는 데 씁니다.

- `count_mismatch`: 사진별 실제 알약 수와 탐지 수가 다른 경우
- `detector_misses`: 실제 알약이 탐지되지 않은 경우
- `false_positives`: 알약이 아닌 영역 또는 중복 bbox가 탐지된 경우
- `recognition_top3_misses`: bbox는 맞았지만 정답 K-ID가 Top-3에 없는 경우
- `status_review`: `no_candidate`, `low_confidence`, `ambiguous`로 사용자 재확인이 필요한 detection
- `warning_images`: 흐림, 과노출, 저해상도 등 촬영 품질 경고가 있는 사진

Gemini는 비교 실험용으로만 유지합니다.

```bash
export PILL_RECOGNIZER=gemini
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```
