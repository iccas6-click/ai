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

## 학습

118 클래스 RTMDet v4 체크포인트에서 backbone, neck, box regression 가중치를 재사용하고 118 클래스 분류 출력층은 제거합니다. 새 출력층은 `pill` 한 클래스만 예측합니다.

```bash
# 8개 train / 4개 validation으로 전체 코드 경로 확인
python -m training.rtmdet_single_class.scripts.train --smoke

# 전체 데이터 학습
python -m training.rtmdet_single_class.scripts.train
```

기본 설정은 RTX 4060 Laptop 8GB 기준 `1024x1024`, batch 8, AMP입니다. WSL에서 `CachedMosaic` worker 교착을 피하기 위해 `num_workers=0`을 사용합니다.

## Baseline 결과

2026-06-24 실행에서는 3 epoch 이후 validation 성능이 정체되어 12 epoch 설정을 조기 종료했습니다.

| Epoch | COCO bbox mAP | mAP@50 | mAP@75 |
|---:|---:|---:|---:|
| 1 | 0.928 | 0.990 | 0.990 |
| 2 | **0.939** | **0.990** | **0.990** |
| 3 | 0.935 | 0.990 | 0.990 |

최고 체크포인트는 `training/runs/rtmdet-single-class/best_coco_bbox_mAP_epoch_2.pth`이며, 추론용 복사본은 `inference/artifacts/rtmdet-single-class/model.pth`입니다.

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
