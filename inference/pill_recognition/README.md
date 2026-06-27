# Pill Recognition v2

Provider 기반 알약 인식 파이프라인입니다.

```text
image
-> RTMDet single-class detector
-> crop per pill
-> vision provider extracts imprint/color/shape/text
-> AI Hub 1000-class product DB search
-> ranked product candidates
```

기존 `RTMDet + AIHub ResNet152 + EfficientNet` baseline은 `inference/pill_recognition_legacy/`에 보존합니다.

## 실행

기본값은 외부 API 없이 실행되는 local provider입니다. 이 모드는 색상 정도만 추정하므로 제품명 식별력은 낮습니다.

```bash
cd /home/gyuha_lee/pill/code/ai/inference
source ../.venv/bin/activate
python -m pill_recognition.app
```

Gemini를 사용할 때는 다음 환경 변수를 지정합니다.

```bash
export PILL_VISION_PROVIDER=gemini
export GEMINI_API_KEY=...
export PILL_GEMINI_MODEL=gemini-3.5-flash
python -m pill_recognition.app
```

Gemini 출력은 최종 정답으로 사용하지 않습니다. 각인, 색, 모양, 후보명 단서만 받아 AI Hub 제품 DB 검색으로 검증합니다.
