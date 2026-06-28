# Evaluation Dataset

서비스 검증용 사진과 정답 파일을 두는 위치입니다. 이 디렉터리의 실제 이미지는 Git에 올리지 않습니다.

## 권장 구성

```text
datasets/evaluation/
├── real-smartphone/
│   ├── images/
│   │   ├── real_001.jpg
│   │   └── real_002.jpg
│   └── manifest.csv
├── synthetic-count/
└── README.md
```

## manifest.csv

정답을 모르는 사진도 평가 스크립트에 넣을 수 있지만, 성능 지표를 계산하려면 `manifest.csv`를 작성합니다.

```csv
image,expected_count,expected_class_names,expected_item_seqs
real_001.jpg,3,K-011015;K-002049;K-006323,200103013;198701672;199701398
real_002.jpg,5,,
```

- `image`: `images/` 아래 이미지 파일명
- `expected_count`: 사진 안의 알약 개수
- `expected_class_names`: AI Hub K-ID. 여러 개면 `;`로 구분
- `expected_item_seqs`: 품목기준코드. 여러 개면 `;`로 구분

초기에는 `image`, `expected_count`만 채워도 충분합니다. 제품명까지 검증하려면 Gradio 결과나 약 포장지 정보를 보고 `expected_class_names` 또는 `expected_item_seqs`를 채웁니다.

## 촬영 세트

최소 검증 세트는 다음처럼 나눕니다.

- `count-1-4`: 학습 데이터와 비슷한 쉬운 구간
- `count-5-7`: 서비스 목표 구간
- `count-8-10`: 탐지 한계 확인 구간
- `hard-cases`: 겹침, 흔들림, 어두운 조명, 반사, 손바닥, 무늬 배경

각 구간마다 최소 20장씩 모으면 count 성능을 대략 볼 수 있고, 제품명 검증은 실제 보유한 약 20~50종부터 시작합니다.
