# Pill Recognition Baseline

여러 알약이 놓인 사진 한 장에서 알약 위치를 탐지하고, 알약별 Top-3 제품 후보를 반환하는 baseline입니다.

## 환경

이 프로젝트는 Python 3.11과 다음 버전 조합을 사용합니다.

- PyTorch 2.1.0 + CUDA 11.8
- torchvision 0.16.0
- MMEngine 0.10.4
- MMCV 2.1.0
- MMDetection 3.3.0

```bash
cd /home/gyuha_lee/pill/code/ai
uv venv --python 3.11 .venv

uv pip install --python .venv/bin/python \
  'torch==2.1.0+cu118' 'torchvision==0.16.0+cu118' \
  --index-url https://download.pytorch.org/whl/cu118

uv pip install --python .venv/bin/python \
  'numpy<2' 'mmengine==0.10.4' 'mmdet==3.3.0' \
  'gradio>=6,<7' 'huggingface-hub>=1,<2' \
  'opencv-python-headless>=4.8,<5' 'PyYAML>=6,<7' 'pytest>=8,<10'

uv pip install --python .venv/bin/python 'mmcv==2.1.0' \
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

## 실행

`artifacts/rtmdet-single-class/model.pth`가 있으면 로컬 단일 클래스 RTMDet를 우선 사용합니다. 없으면 Apache-2.0으로 공개된 118 클래스 RTMDet v4 가중치와 `pill.yaml`을 Hugging Face에서 내려받습니다.
AI Hub class01 가중치와 라벨 JSON이 기본 경로에 있으면 1,000종 분류기도 자동으로 활성화됩니다.

```bash
cd /home/gyuha_lee/pill/code/ai/inference
source ../.venv/bin/activate
python -m pill_recognition.app
```

접속 주소: `http://127.0.0.1:7860`

CLI 추론:

```bash
python -m pill_recognition.cli path/to/image.png --output outputs/result.png
```

폴더 단위 평가:

```bash
python -m pill_recognition.evaluate_dataset \
  --images ../datasets/evaluation/real-smartphone/images \
  --manifest ../datasets/evaluation/real-smartphone/manifest.csv \
  --output-dir outputs/evaluation/real-smartphone
```

정답 manifest가 없으면 이미지별 탐지·분류 결과와 annotated 이미지만 저장합니다.

```bash
python -m pill_recognition.evaluate_dataset \
  --images artifacts/samples/server-validation \
  --output-dir outputs/evaluation/server-validation
```

출력 파일은 `results.json`, `results.csv`, `summary.json`, `annotated/`입니다.

RTMDet detector-only 평가:

```bash
python -m pill_recognition.evaluate_detector \
  --images ../datasets/processed/rtmdet-single-class/images/val \
  --labels ../datasets/processed/rtmdet-single-class/labels/val \
  --output-dir outputs/evaluation/rtmdet-val-800
```

검증을 빠르게 해볼 때는 `--limit 50`을 붙입니다. 출력은 `results.json`, `results.csv`, `summary.json`이며 IoU 0.5 기준 count accuracy, precision, recall, F1, matched IoU를 기록합니다.

다른 단일 클래스 검출기를 사용할 때는 가중치와 클래스 파일을 함께 지정합니다.

```bash
export PILL_DETECTOR_CHECKPOINT=/path/to/model.pth
export PILL_DETECTOR_CLASSES=/path/to/pill.yaml
```

## 테스트 이미지

알약 조합과 촬영 각도를 분산한 검증 이미지 60장을 내려받습니다.

```bash
python -m pill_recognition.download_samples --count 60
```

검증셋 2,090장을 모두 받으려면 다음 명령을 사용합니다. 전체 데이터는 수 GB이므로 학습이나 대규모 평가가 필요할 때만 사용합니다.

```bash
python -m pill_recognition.download_samples --all
```

내려받은 이미지는 `artifacts/samples/diverse/images/val/`에 저장되며 Gradio 테스트 이미지 목록에 자동으로 나타납니다.

## AI Hub 1,000종 분류기 연결

AI Hub 공식 20.8GB 패키지는 다음 URL에서 서버가 직접 내려받습니다.

```text
https://www.aihub.or.kr/file/down.do?fileSn=10697&aiModelFileSn=10697&dataSetSn=576
```

Linux에서는 CP949 파일명 인코딩을 지정해 압축 해제합니다.

```bash
unzip -O CP949 <downloaded-package>.zip -d aihub_official_code/package
```

공식 패키지는 `aihub_official_code/package/평가용 데이터셋/pill_data`에서 자동 탐색합니다. class01 모델은 `proj_pill`, 1,000개 K-ID 매핑은 `pill_data_croped` 아래에 있으며 이미지와 JSON은 약 259만 개입니다.

공식 패키지 경로가 없으면 다음 레거시 경로를 fallback으로 사용합니다.

```text
aihub_official_code/docker img/proj_pill/pill_resnet152_dataclass01_aug0.pt
aihub_official_code/docker img/proj_pill/pill_label_path_sharp_score.json
```

다른 위치를 사용할 때는 환경 변수로 지정합니다.

```bash
export PILL_AIHUB_WEIGHTS=/path/to/pill_resnet152_dataclass01_aug0.pt
export PILL_AIHUB_MAPPING=/path/to/pill_label_path_sharp_score.json
```

파이프라인은 RTMDet가 찾은 모든 Bounding Box를 12% 확장해 Crop하고, Crop들을 한 번에 ResNet152로 분류합니다. 결과는 알약마다 `K-xxxxx` 형식의 Top-3 후보로 반환되며 사용자가 최종 확인해야 합니다.

`aihub_official_code/`는 대용량 가중치와 AI Hub 배포 파일을 포함하므로 Git에서 제외합니다.

## EfficientNet 연결

GitHub EfficientNet 가중치와 클래스 매핑을 사용할 수 있는 경우 다음 환경 변수로 118종 비교 결과를 활성화합니다.

```bash
export PILL_CNN_WEIGHTS=/path/to/cls119_classifier_v4.pt
export PILL_CNN_MAPPING=/path/to/class_mapping.csv
```

변수를 설정하지 않으면 GitHub CNN 열만 비어 있습니다. AI Hub 분류기는 독립적으로 동작합니다.

현재 머신처럼 아래 경로에 자산이 있으면 환경 변수 없이 자동으로 연결됩니다.

```text
artifacts/cnn/cls119_classifier_v4.pt
artifacts/cnn/class_mapping.csv
```

`artifacts/`는 Git에서 제외됩니다. CNN 자산의 재배포 가능 여부를 확인하기 전에는 원격 저장소에 올리지 않습니다.

## 현재 제한

- AI Hub 분류 범위는 전문의약품 600종과 일반의약품 400종입니다.
- 현재 K-ID를 제품명·품목기준코드로 변환하는 제품 마스터가 없습니다.
- 단일 클래스 RTMDet는 1~4개 합성 이미지에서만 검증됐으며 실제 촬영과 5~10개 장면의 재현율은 아직 측정되지 않았습니다.
- 한쪽 면만 보이는 사진으로 제품을 확정할 수 없으므로 사용자 확인이 필요합니다.
- 알약끼리 겹치거나 너무 작으면 탐지 성능이 떨어집니다.
- 이 결과는 복약 판단에 직접 사용할 수 없습니다.

## 모델 출처

- 초기 탐지 가중치: `wony98/healtheat-pill-rtmdet-v4`
- 초기 가중치 고정 revision: `91fa48ea31327c7c724e7c104a61b55119a6ae31`
- 학습 탐지 모델: RTMDet-tiny `pill` 단일 클래스 baseline
- 초기 탐지 가중치 라이선스: Apache-2.0
- 분류 모델: AI Hub 경구약제 이미지 데이터 공식 ResNet152 class01
