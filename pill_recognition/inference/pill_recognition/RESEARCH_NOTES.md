# Pill Recognition Strategy Notes

이 문서는 CLICK 알약 인식 파이프라인의 모델 선택 근거를 정리합니다.

## 현재 운영 방향

기본 경로는 외부 멀티모달 LLM이 아니라 로컬 GPU 기반 파이프라인입니다.

```text
multi-pill photo
-> RTMDet single-class detector
-> crop per pill
-> AI Hub ResNet152 feature retrieval
-> Top-3 candidates with product metadata/reference image
-> optional imprint/color/shape refinement
```

이 구조를 유지하는 이유는 다음과 같습니다.

- 앱 응답 시간과 비용을 통제할 수 있습니다.
- 같은 입력에 대한 결과 일관성이 높습니다.
- 제품명을 하나로 단정하지 않고 Top-3 후보와 근거 이미지를 줄 수 있습니다.
- detector, retrieval, metadata/OCR 보정을 따로 평가하고 개선할 수 있습니다.

Gemini 같은 멀티모달 LLM은 비교 실험 또는 보조 설명 생성에는 쓸 수 있지만, 제품 식별 기본 경로로 두지 않습니다. 약 식별은 환각과 출력 변동을 허용하기 어렵고, 여러 crop을 반복 호출하면 latency와 비용이 커집니다.

## 외부 자료에서 얻은 방향성

- `Fast and accurate medication identification`은 consumer-quality pill images가 조명, 초점, 기기 조건에 따라 달라지는 점을 전제로 pill recognition challenge를 다룹니다. 실서비스 검증은 reference crop이 아니라 사용자 촬영 분포에서 해야 합니다.
- `An Accurate Deep Learning-Based System for Automatic Pill Identification`은 shape, color, imprint character를 추출하고 database retrieval로 제품을 찾는 구성을 제안합니다. 이미지 하나로 제품명을 단정하기보다 외형/각인 metadata를 결합하는 방향이 맞습니다.
- `Few-Shot Pill Recognition` / CURE 계열은 통제되지 않은 촬영 조건과 도메인 차이가 pill recognition의 핵심 난점임을 보여줍니다.
- `ePillID`는 수천 appearance class의 low-shot fine-grained retrieval 문제로 pill identification을 정의합니다. 우리가 AIHub 1000종 reference를 retrieval index로 쓰는 선택과 맞닿아 있습니다.
- multi-pill detection 연구들은 현실 사진에서 겹침, 반사, clutter가 localization recall을 크게 흔든다고 봅니다. RTMDet 단일 클래스 detector는 합성 scene으로 시작하되, 최종 판단은 실제 스마트폰 평가셋에서 해야 합니다.

## 당장 바꾸지 않는 것

- DINOv2 foundation embedding은 서버에서 전체 1000클래스 held-out 평가를 돌렸고 AIHub ResNet152보다 낮았습니다. 기본값으로 쓰지 않습니다.
- 합성 multi-pill scene crop으로 제품명 retrieval 성능을 판단하지 않습니다. 합성 crop은 detector stress test에는 유용하지만, 제품 인식 평가에서는 분포 차이가 큽니다.
- LLM으로 crop별 제품명을 직접 맞히는 구조를 기본값으로 두지 않습니다.

## 다음 정확도 루프

1. 실제 스마트폰 사진 30~50장, 총 150~300알을 모읍니다.
2. `draft_real_annotations`로 detector/retrieval 초안을 만듭니다.
3. `render_real_annotation_review` HTML 리포트에서 bbox, crop, Top-3 후보를 검수합니다.
4. 검수된 annotation으로 `evaluate_real_dataset`을 돌립니다.
5. `analysis.recognition_top3_misses`, `analysis.detector_misses`, `analysis.warning_images`를 보고 개선 대상을 분리합니다.
6. Top-3 miss가 많은 제품군은 실제 crop을 소량 추가해 retrieval index augmentation 또는 metric learning fine-tune 후보로 올립니다.

## 참고 자료

- Fast and accurate medication identification: https://www.nature.com/articles/s41746-019-0086-0
- Automatic pill identification system: https://pmc.ncbi.nlm.nih.gov/articles/PMC9883737/
- Few-Shot Pill Recognition: https://openaccess.thecvf.com/content_CVPR_2020/papers/Ling_Few-Shot_Pill_Recognition_CVPR_2020_paper.pdf
- ePillID benchmark: https://arxiv.org/abs/2005.14288
- Image-based Contextual Pill Recognition with Medical Knowledge Graph: https://arxiv.org/abs/2208.02432
- High Accurate and Explainable Multi-Pill Detection Framework: https://arxiv.org/abs/2303.09782
