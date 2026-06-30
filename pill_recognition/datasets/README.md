# Local Datasets

학습 및 평가 데이터는 이 디렉터리 아래에 저장하며 Git으로 추적하지 않습니다.

```text
datasets/
├── raw/                         # 다운로드한 원본, 수정 금지
│   ├── healtheat-pill-yolo/
│   ├── healtheat-pill-synthetic-v3/
│   └── aihub-oral-pill/
├── processed/
│   └── rtmdet-single-class/     # 모든 YOLO class_id를 pill=0으로 변환
└── evaluation/
    ├── count-1-4/
    ├── count-5-7/
    └── count-8-10/
```

## 원칙

- `raw/` 파일은 수정하지 않습니다.
- 변환 스크립트의 출력은 `processed/`에만 기록합니다.
- 학습·평가 분할은 원본 사진 및 합성 scene 단위로 수행합니다.
- 실제 스마트폰 검증 사진과 정답 라벨은 `evaluation/`에 둡니다.
- AI Hub 원본 데이터와 가중치는 배포 조건을 확인하기 전까지 저장소에 올리지 않습니다.
