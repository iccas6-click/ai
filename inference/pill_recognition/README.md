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
모델 선택 근거와 외부 자료 검토 요약은 [`RESEARCH_NOTES.md`](./RESEARCH_NOTES.md)에 정리합니다.

## Production 방향

실서비스 기본 경로는 외부 멀티모달 LLM 호출이 아니라 로컬 GPU 기반 retrieval입니다.

- RTMDet는 제품명을 맞히지 않고 알약 위치만 찾습니다.
- 제품 인식은 crop을 AIHub ResNet152 embedding으로 변환한 뒤 reference index에서 Top-K 검색합니다.
- 응답은 단일 정답이 아니라 항상 제품 후보 Top-3, 성분, reference image, 확인 status를 반환합니다.
- Gemini 같은 외부 비전 LLM은 latency, 비용, 출력 변동성 때문에 기본 경로에서 제외합니다.
- 낮은 신뢰도, 비슷한 후보, 흐림/반사/겹침은 사용자 확인, 반대면 crop, 각인 입력, 수동 검색으로 넘깁니다.

즉 이 서비스의 책임은 "복용 가능한 최종 판정"이 아니라 "사진 속 알약 후보를 빠르게 좁히고 사용자가 확인할 수 있게 만드는 것"입니다.

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

사용자의 복약목록 K-ID가 있으면 최초 인식 요청부터 검색 범위를 줄일 수 있습니다. `allowed_pill_ids`는 JSON 배열 문자열, 쉼표 구분, 공백 구분을 모두 허용합니다.

```bash
curl -X POST http://127.0.0.1:8001/recognize \
  -F "file=@sample.jpg" \
  -F 'allowed_pill_ids=["K-001732","K-012914","K-000845"]'
```

응답은 `RecognitionResult.to_dict()`와 같은 JSON이며, 알약별 `vision.color`, `vision.shape`, `candidates`, `status`, `status_reason`을 포함합니다. `warnings`에는 낮은 해상도, 어두움, 과노출, 낮은 대비, 흔들림처럼 재촬영이 필요한 입력 품질 문제가 포함됩니다. `timings_ms`에는 파이프라인 내부 `quality`, `detector`, `recognition`, `postprocess`, `total`과 API 레벨 `upload_decode`, `pipeline_get`, `pipeline_call`, `api_total` latency가 포함됩니다. Warmup이 정상 완료되면 일반 요청의 `pipeline_get`은 매우 작아야 합니다.

`allowed_pill_ids`를 보낸 경우 응답의 `candidate_scope`에 scope 적용 결과가 포함됩니다. `retrieval_id_match_count`가 0이면 입력한 K-ID가 retrieval index에 없다는 뜻이므로, 모델 판단보다 사용자 복약목록 K-ID 매핑을 먼저 확인해야 합니다.

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
  -F "files=@back_crop.jpg" \
  -F 'allowed_pill_ids=["K-001732","K-012914","K-000845"]'
```

이 endpoint는 여러 crop을 한 번의 batch로 AIHub retrieval에 넣습니다. 앞/뒷면 crop이나 여러 선택 알약을 재확인할 때 HTTP 왕복과 모델 호출 부담을 줄일 수 있습니다. 기본 최대 crop 수는 `PILL_MAX_BATCH_CROPS=12`이고, 이미지 1장 기본 제한은 `PILL_MAX_UPLOAD_BYTES=10485760`, `PILL_MAX_IMAGE_PIXELS=12000000`입니다.

crop batch 응답의 `warnings`는 `crop 1`, `crop 2`처럼 crop 번호를 포함합니다. 프론트는 해당 crop만 다시 찍도록 안내할 수 있습니다.

crop batch의 `timings_ms`는 파이프라인 내부 `preprocess`, `recognition`, `postprocess`, `total`과 API 레벨 `upload_decode`, `pipeline_get`, `pipeline_call`, `api_total`을 반환합니다. 여러 crop은 한 번의 retrieval batch로 처리되므로 앱에서는 crop을 따로 여러 번 호출하지 말고 batch endpoint를 우선 사용합니다.

운영 속도 기준선은 crop API latency benchmark로 확인합니다. 앱에서는 한 사진에서 탐지된 알약 crop을 모아 batch endpoint로 보내는 흐름이 기본입니다.

```bash
python -m pill_recognition.benchmark_api_latency \
  --base-url http://127.0.0.1:8001 \
  --crop-counts 1,3,6,12 \
  --iterations 10 \
  --warmup 1 \
  --output outputs/evaluation/api-latency.json
```

복약목록 기반 scoped retrieval latency를 측정하려면 같은 benchmark에 `--allowed-pill-ids`를 추가합니다.

```bash
python -m pill_recognition.benchmark_api_latency \
  --base-url http://127.0.0.1:8001 \
  --crop-counts 1,3,6,12 \
  --iterations 10 \
  --warmup 1 \
  --allowed-pill-ids '["K-000059","K-000069","K-000080"]' \
  --output outputs/evaluation/api-latency-scoped.json
```

출력의 `elapsed_ms.p50/p95`는 클라이언트에서 체감하는 HTTP 왕복 시간이고, `api_total_ms.p50/p95`는 서버 내부 처리 시간입니다. `recognition_ms`가 대부분을 차지하면 retrieval batch 최적화 대상이고, `pipeline_get`이 크면 warmup 또는 프로세스 재사용 문제입니다.

서버 RTX 3080 smoke benchmark 기준, crop batch recognition은 복약목록 scope를 넣어도 latency가 거의 변하지 않습니다.

| Mode | Crops | elapsed p50 | API p50 | recognition p50 |
|---|---:|---:|---:|---:|
| unscoped | 1 | 42.43ms | 41.39ms | 41.03ms |
| scoped 3 IDs | 1 | 43.48ms | 42.48ms | 42.17ms |
| unscoped | 3 | 49.89ms | 48.54ms | 47.68ms |
| scoped 3 IDs | 3 | 50.60ms | 49.27ms | 48.43ms |
| unscoped | 6 | 64.69ms | 62.66ms | 61.17ms |
| scoped 3 IDs | 6 | 65.00ms | 63.01ms | 61.32ms |
| unscoped | 12 | 117.53ms | 114.16ms | 111.11ms |
| scoped 3 IDs | 12 | 115.52ms | 112.23ms | 109.63ms |

측정 파일은 서버의 `inference/outputs/evaluation/api-latency-unscoped-latest.json`, `inference/outputs/evaluation/api-latency-scoped-latest.json`에 저장했습니다.

각인/색/모양/텍스트 보정 검색 API:

```bash
curl "http://127.0.0.1:8001/products/search?imprint=W2&shape=원형&color=하양&limit=5"
```

이 endpoint는 AIHub 제품 DB를 직접 검색합니다. 앱에서는 인식 후보가 애매할 때 사용자가 읽은 각인, 앞/뒤면 촬영 결과, 또는 OCR 결과를 넣어 후보를 다시 좁히는 데 사용합니다.

제품 후보의 `reference_image_url`은 AIHub reference crop 이미지입니다. 후보 확인 UI에서 제품명/성분과 함께 표시할 수 있습니다.

```bash
curl http://127.0.0.1:8001/products/K-000001

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

앱에 사용자의 복약목록이 있으면 `allowed_pill_ids`를 함께 보내 후보 공간을 좁힙니다. 이 경우 이미지 후보와 metadata search 결과는 해당 K-ID 목록 안에서만 반환됩니다. 전체 1000종에서 맞히는 것보다 사용자가 실제로 복용 중인 약 5~20개 안에서 재정렬하는 흐름이 실서비스 정확도에 더 유리합니다.

```bash
curl -X POST http://127.0.0.1:8001/products/refine \
  -H "Content-Type: application/json" \
  -d '{
    "allowed_pill_ids": ["K-001732", "K-012914", "K-000845"],
    "candidates": [
      {"pill_id": "K-999999", "score": 97.0, "source": "aihub_resnet_retrieval"},
      {"pill_id": "K-001732", "score": 62.0, "source": "aihub_resnet_retrieval"}
    ],
    "imprint": "W2",
    "limit": 3
  }'
```

응답의 `candidate_scope`는 복약목록 제한 적용 여부, 입력된 K-ID 개수, AIHub metadata에서 실제 찾은 개수, 알 수 없는 K-ID를 반환합니다.

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
사용자의 실제 복약목록을 알고 있는 평가 사진이면 root의 `allowed_pill_ids`에 해당 K-ID 목록을 넣습니다.

```json
{
  "image": "IMG_0001.jpg",
  "allowed_pill_ids": ["K-000000", "K-000001", "K-000002"],
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

초안의 `class_name`, `product_name`, `bbox_xyxy`는 반드시 사람이 확인해야 합니다. `candidate_hints`에는 현재 pipeline의 후보가 들어가며, 맞는 후보가 없으면 AIHub 제품 DB 검색 탭이나 원본 AIHub K-ID 목록으로 확인합니다. JSON만 보지 말고 HTML 리뷰 리포트를 만들어 bbox, crop, Top-3 후보를 한 화면에서 검수합니다.

```bash
python -m pill_recognition.render_real_annotation_review \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --output-dir outputs/real-review
```

생성된 `outputs/real-review/index.html`을 브라우저에서 열고 다음을 확인합니다.

- bbox가 실제 알약 하나만 감싸는지
- `class_name`이 검증된 AIHub K-ID인지
- 후보 Top-3에 정답이 있는지
- 흐림, 과노출, 겹침처럼 평가셋에서 제외하거나 별도 태깅할 촬영 문제가 있는지

평가를 돌리기 전에 annotation이 검수 완료 상태인지 확인합니다. 기본 검증은 `needs_review=true`, 이미지 범위를 벗어난 bbox, AIHub metadata/retrieval index에 없는 K-ID, `allowed_pill_ids`에 정답 K-ID가 빠진 케이스를 error로 보고합니다.

```bash
python -m pill_recognition.validate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --output outputs/evaluation/real-smartphone-validation.json
```

초안 검수 중이라 `needs_review=true`를 허용하고 구조만 확인하려면 `--allow-needs-review`를 추가합니다.

평가:

```bash
python -m pill_recognition.evaluate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --top-k 5 \
  --output outputs/evaluation/real-smartphone.json
```

복약목록 scope 효과를 같이 보려면 같은 평가셋을 mode별로 돌립니다.

```bash
# 전체 AIHub 1000종 검색 기준
python -m pill_recognition.evaluate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --scope-mode none \
  --top-k 5 \
  --output outputs/evaluation/real-smartphone-unscoped.json

# annotation의 allowed_pill_ids 기준
python -m pill_recognition.evaluate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --scope-mode annotation \
  --top-k 5 \
  --output outputs/evaluation/real-smartphone-annotation-scope.json

# 정답 K-ID를 복약목록으로 넣는 oracle 실험. 실서비스 수치가 아니라 scope의 최대 개선폭 확인용입니다.
python -m pill_recognition.evaluate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --scope-mode ground-truth \
  --top-k 5 \
  --output outputs/evaluation/real-smartphone-oracle-scope.json
```

평가 결과 비교:

```bash
python -m pill_recognition.compare_real_evaluations \
  --baseline outputs/evaluation/real-smartphone-unscoped.json \
  --candidate outputs/evaluation/real-smartphone-annotation-scope.json \
  --name-baseline unscoped \
  --name-candidate annotation-scope \
  --output outputs/evaluation/real-smartphone-scope-compare.json
```

세 평가와 비교 리포트를 한 번에 만들려면 suite runner를 사용합니다.

```bash
python -m pill_recognition.run_real_evaluation_suite \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --top-k 5 \
  --output-dir outputs/evaluation \
  --prefix real-smartphone
```

이 명령은 validation, `unscoped`, `annotation-scope`, `oracle-scope` 평가와 `annotation-vs-unscoped`, `oracle-vs-unscoped`, `oracle-vs-annotation` 비교 JSON을 생성합니다. 먼저 실행 계획만 보려면 `--dry-run`, 이미 생성된 결과를 건너뛰려면 `--skip-existing`, validation을 생략하려면 `--skip-validation`을 추가합니다.

이 결과의 핵심 지표는 `detector_f1`, `recognition_top3_on_matched`, `end_to_end_top3_on_gt`, `mean_total_ms`, `p95_total_ms`입니다.

결과 JSON에는 `analysis` 섹션도 포함됩니다. 이 섹션은 다음 케이스를 따로 모아 실제 개선 우선순위를 정하는 데 씁니다.

- `count_mismatch`: 사진별 실제 알약 수와 탐지 수가 다른 경우
- `detector_misses`: 실제 알약이 탐지되지 않은 경우
- `false_positives`: 알약이 아닌 영역 또는 중복 bbox가 탐지된 경우
- `recognition_top3_misses`: bbox는 맞았지만 정답 K-ID가 Top-3에 없는 경우
- `status_review`: `no_candidate`, `low_confidence`, `ambiguous`로 사용자 재확인이 필요한 detection
- `warning_images`: 흐림, 과노출, 저해상도 등 촬영 품질 경고가 있는 사진

Gemini는 비교 실험용으로만 유지합니다. 실서비스 기본 경로에서는 켜지지 않으며, 의도적으로 실험 플래그를 함께 지정해야 합니다.

```bash
export PILL_RECOGNIZER=gemini
export PILL_ENABLE_EXPERIMENTAL_GEMINI=1
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```
