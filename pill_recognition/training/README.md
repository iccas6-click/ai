# Training

학습 코드와 재현 가능한 설정만 Git으로 관리합니다. 데이터셋은 `datasets/`, 체크포인트와 로그는 `training/runs/`에 저장합니다.

```text
training/
└── rtmdet_single_class/
    ├── configs/     # MMDetection 설정
    ├── scripts/     # 데이터 변환, 학습, 평가 명령
    └── README.md
```

검증을 통과한 최종 가중치만 `inference/artifacts/rtmdet-single-class/`로 복사합니다.
