# RTMDet Single-Class Training

제품 118종을 동시에 분류하던 RTMDet를 제품 종류와 무관한 `pill` 단일 클래스 탐지기로 재학습하는 영역입니다.

## 입출력

```text
입력 데이터:  datasets/raw/
변환 데이터:  datasets/processed/rtmdet-single-class/
학습 산출물:  training/runs/rtmdet-single-class/
최종 가중치:  inference/artifacts/rtmdet-single-class/model.pth
```

## 데이터 준비

현재 baseline은 `wony98/healtheat-pill-synthetic-v3`를 사용합니다.

```bash
cd /home/gyuha_lee/pill/code/ai
source .venv/bin/activate

python -m training.rtmdet_single_class.scripts.download_datasets synthetic-v3
python -m training.rtmdet_single_class.scripts.prepare_single_class \
  datasets/raw/healtheat-pill-synthetic-v3/extracted
```

변환 결과는 다음과 같습니다.

| 분할 | 이미지 | Bounding Box |
|---|---:|---:|
| train | 6,509 | 14,680 |
| validation | 800 | 1,808 |
| 합계 | 7,309 | 16,488 |

누락 라벨과 잘못된 Bounding Box는 모두 0건입니다. 원본 클래스 ID는 전부 `pill=0`으로 변환하며, RTMDet 학습용 COCO JSON도 함께 생성합니다.

## AI Hub 1000종 합성셋 생성

AI Hub 공식 crop 이미지 1000종을 섞어 한 장에 1~10개 알약이 들어간 합성 scene을 생성할 수 있습니다.

```bash
python -m training.rtmdet_single_class.scripts.generate_aihub_synthetic \
  --output datasets/processed/rtmdet-aihub-synthetic-realistic-max10 \
  --train-count 8000 \
  --val-count 1000 \
  --min-pills 1 \
  --max-pills 10 \
  --overwrite
```

출력 구조는 RTMDet 학습에 바로 사용할 수 있는 형태입니다.

```text
datasets/processed/rtmdet-aihub-synthetic-realistic-max10/
├── images/train/*.jpg
├── images/val/*.jpg
├── labels/train/*.txt          # YOLO pill=0 bbox 라벨
├── labels/val/*.txt
├── metadata/train/*.json       # K-ID, 제품명, 품목기준코드, source image
├── metadata/val/*.json
├── train_coco.json
├── val_coco.json
├── pill.yaml
└── manifest.json
```

생성된 라벨은 detector 학습용 단일 클래스 `pill=0`입니다. 제품명과 AI Hub K-ID는 `metadata/`에 보존해 end-to-end 검증용 manifest를 만들 때 사용합니다.
기본 배경은 나무 책상, 종이, 대리석, 패브릭, 손바닥, 조리대 느낌의 절차적 표면입니다. 기존 단순 배경이 필요하면 `--background-mode simple`을 붙입니다.

현재 생성기는 crop을 그대로 붙이지 않고 다음 후처리를 적용합니다.

- 전경 mask를 distance transform 기반 soft alpha로 feathering
- mask 품질(`area_ratio`, `bbox_fill_ratio`, `soft_edge_ratio`) 기록 및 불량 mask 제외
- 배경 밝기/색에 맞춘 약한 patch tone adaptation
- 접촉 그림자와 약한 blur로 스티커처럼 떠 보이는 경계 완화

이 합성셋은 RTMDet 단일 클래스 detector 학습과 5~10개 알약 위치 탐지 stress test용입니다. 제품명 retrieval 정확도 판단은 실제 스마트폰 평가셋을 기준으로 합니다.
`manifest.json`에는 `requested_count_distribution`과 `placed_count_distribution`을 모두 기록합니다. 두 분포 차이가 크면 crop 크기, mask 품질 threshold, image size를 먼저 확인해야 합니다.

생성 직후에는 합성 품질 리포트와 bbox preview를 확인합니다.

```bash
python -m training.rtmdet_single_class.scripts.audit_aihub_synthetic \
  --dataset-root datasets/processed/rtmdet-aihub-synthetic-realistic-max10 \
  --split train \
  --limit 200 \
  --report-output outputs/evaluation/synthetic-train-audit.json \
  --preview-output outputs/evaluation/synthetic-train-preview.jpg
```

리포트에서 `count_match_rate`가 1.0에 가깝고, `requested_count_distribution`과 `placed_count_distribution`이 거의 같아야 합니다. `low_quality_masks`나 `high_attempt_images`가 많으면 학습 전에 합성 파라미터를 조정합니다.

## 학습

118 클래스 RTMDet v4 체크포인트에서 backbone, neck, box regression 가중치를 재사용하고 118 클래스 분류 출력층은 제거합니다. 새 출력층은 `pill` 한 클래스만 예측합니다.

```bash
# 8개 train / 4개 validation으로 전체 코드 경로 확인
python -m training.rtmdet_single_class.scripts.train --smoke

# 전체 데이터 학습
python -m training.rtmdet_single_class.scripts.train
```

개선된 AI Hub 합성셋으로 학습할 때는 dataset root와 work dir을 명시합니다.

```bash
python -m training.rtmdet_single_class.scripts.train \
  --data-root datasets/processed/rtmdet-aihub-synthetic-realistic-max10 \
  --work-dir training/runs/rtmdet-aihub-synthetic-realistic-max10 \
  --num-workers 4
```

기본 설정은 RTX 4060 Laptop 8GB 기준 `1024x1024`, batch 8, AMP입니다. WSL에서 `CachedMosaic` worker 교착을 피하기 위해 `num_workers=0`을 사용합니다.

## 결과

2026-06-24 실행에서는 3 epoch 이후 validation 성능이 정체되어 12 epoch 설정을 조기 종료했습니다.

| Epoch | COCO bbox mAP | mAP@50 | mAP@75 |
|---:|---:|---:|---:|
| 1 | 0.928 | 0.990 | 0.990 |
| 2 | **0.939** | **0.990** | **0.990** |
| 3 | 0.935 | 0.990 | 0.990 |

최고 체크포인트는 `training/runs/rtmdet-single-class/best_coco_bbox_mAP_epoch_2.pth`이며, 추론용 복사본은 `inference/artifacts/rtmdet-single-class/model.pth`입니다.

2026-06-28 실행에서는 개선된 AI Hub synthetic v2 dataset으로 전체 12 epoch 학습을 완료했습니다.

| 항목 | 값 |
|---|---:|
| train images | 8,000 |
| train boxes | 43,633 |
| val images | 1,000 |
| val boxes | 5,640 |
| train/val count match rate | 1.0 / 1.0 |
| audit low-quality masks | 0 |
| best epoch | 7 |
| best COCO bbox mAP | 0.990 |
| best mAP@50 | 0.990 |
| best mAP@75 | 0.990 |

API detector wrapper 경로로도 같은 validation split을 평가했습니다.

| Metric | Value |
|---|---:|
| images | 1,000 |
| ground truth boxes | 5,640 |
| predicted boxes | 5,644 |
| count exact accuracy | 0.994 |
| precision | 0.9991 |
| recall | 0.9998 |
| F1 | 0.9994 |
| mean matched IoU | 0.9958 |
| false positives | 5 |
| false negatives | 1 |

기존 default detector와 v2 후보를 같은 synthetic v2 validation split에서 비교한 결과입니다.

| Metric | Default | v2 candidate | Delta |
|---|---:|---:|---:|
| count exact accuracy | 0.865 | 0.994 | +0.129 |
| precision | 0.9901 | 0.9991 | +0.009 |
| recall | 0.9771 | 0.9998 | +0.0227 |
| F1 | 0.9836 | 0.9994 | +0.0158 |
| mean matched IoU | 0.8238 | 0.9958 | +0.172 |
| false positives | 55 | 5 | -50 |
| false negatives | 129 | 1 | -128 |
| paired image wins | - | 996 / 1000 | - |

`compare_detector_evaluations` 기준으로 v2 후보가 996장 개선, default가 4장 우세였습니다. 이 4장은 대부분 v2의 추가 false positive 때문이라, 실제 사진 평가에서도 regression 목록을 반드시 확인합니다.

서버 후보 가중치는 다음 위치에 staging했습니다.

```text
inference/artifacts/rtmdet-single-class/model-aihub-synthetic-v2.pth
```

기본 `model.pth`는 아직 교체하지 않습니다. 합성 validation 성능은 detector 학습 sanity check로는 충분하지만, 실제 스마트폰 사진 검증셋에서 count recall/false positive를 확인한 뒤 default로 승격합니다. 후보 모델을 테스트할 때는 환경변수로 명시합니다.

```bash
export PILL_DETECTOR_CHECKPOINT=/home/gyu/pill/code/ai/pill_recognition/inference/artifacts/rtmdet-single-class/model-aihub-synthetic-v2.pth
export PILL_DETECTOR_CLASSES=/home/gyu/pill/code/ai/pill_recognition/inference/artifacts/rtmdet-single-class/pill.yaml
```

실제 스마트폰 detector 검증셋이 준비되면 기존 default와 v2 후보를 같은 이미지/라벨로 각각 평가한 뒤 비교합니다.

```bash
# default detector
python -m pill_recognition_legacy.evaluate_detector \
  --images ../datasets/evaluation/real-smartphone-yolo/images \
  --labels ../datasets/evaluation/real-smartphone-yolo/labels \
  --output-dir outputs/evaluation/real-detector-default \
  --save-annotated \
  --annotated-limit 30

# v2 candidate detector
PILL_DETECTOR_CHECKPOINT=/home/gyu/pill/code/ai/pill_recognition/inference/artifacts/rtmdet-single-class/model-aihub-synthetic-v2.pth \
PILL_DETECTOR_CLASSES=/home/gyu/pill/code/ai/pill_recognition/inference/artifacts/rtmdet-single-class/pill.yaml \
python -m pill_recognition_legacy.evaluate_detector \
  --images ../datasets/evaluation/real-smartphone-yolo/images \
  --labels ../datasets/evaluation/real-smartphone-yolo/labels \
  --output-dir outputs/evaluation/real-detector-v2 \
  --save-annotated \
  --annotated-limit 30

python -m pill_recognition_legacy.compare_detector_evaluations \
  --baseline outputs/evaluation/real-detector-default \
  --candidate outputs/evaluation/real-detector-v2 \
  --name-baseline default \
  --name-candidate aihub-synthetic-v2 \
  --output outputs/evaluation/real-detector-compare.json
```

승격 기준은 실제 사진에서 `recall`을 떨어뜨리지 않으면서 `false_positive`, `count_mean_abs_error`, 이미지별 regression이 줄어드는 것입니다.

로컬 다중 알약 샘플 5장에서는 각 4개, 총 20개를 모두 검출했습니다. 이 샘플은 학습 데이터와 유사한 합성 도메인이므로 실제 촬영 성능을 뜻하지 않습니다.

```bash
PYTHONPATH=inference python \
  training/rtmdet_single_class/scripts/evaluate_runtime_samples.py
```

평가 결과는 `inference/outputs/rtmdet-single-class/summary.json`과 주석 이미지로 저장됩니다.

## 서버에서 학습

서버 최초 설치, `tmux` 학습, 체크포인트 회수와 SSH 터널 사용법은 [`SERVER_TRAINING.md`](./SERVER_TRAINING.md)를 따릅니다.

## 진행 순서

1. 기존 YOLO 라벨의 class_id를 모두 `0`으로 변환합니다.
2. 이미지와 라벨 누락, 잘못된 Bounding Box를 검사합니다.
3. scene 단위로 train/validation을 분리합니다.
4. RTMDet-tiny의 출력 클래스를 `pill` 하나로 설정합니다.
5. 알약 개수 구간별로 detection recall과 false positive를 측정합니다.
6. 검증된 가중치만 `inference/artifacts/rtmdet-single-class/`에 배치합니다.

현재 1~4개 합성 샘플 baseline까지 완료했습니다. 다음 검증 대상은 `datasets/evaluation/`의 실제 스마트폰 사진과 5~7개, 8~10개 구간입니다.
