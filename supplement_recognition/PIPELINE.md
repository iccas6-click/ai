# 건강기능식품 인식 파이프라인

## 개요

사용자가 촬영한 건강기능식품 포장 이미지에서 제품명을 추출하고, 식약처(MFDS) DB에서 일치 제품을 찾아 성분 목록을 반환하는 파이프라인입니다.  
DB에 없는 제품은 Gemini가 식약처 공전 기준으로 성분을 직접 추출하여 반환합니다.  
성분명은 한국어로 반환하며, 번역은 백엔드에서 처리합니다.

```
이미지
  → [1] 전처리
  → [2] Gemini Vision — 제품명 후보 최대 3개 추출
  → [3] 후보 전체 DB 매칭 — confidence 최고 결과 선택
  → [4-A] DB 등재 제품: 동명이제 감지 → 자동 해소 or candidates[] 반환
  → [4-B] DB 미등재 제품: Gemini로 식약처 기준 성분 추출 (ingredients_source: "gemini")
  → [5] SupplementRecognitionResult 반환
```

---

## 디렉터리 구조

```
supplement_recognition/
├── src/
│   ├── extraction/
│   │   ├── llm_extractor.py        # Gemini Vision 제품명 후보 추출 (최대 3개)
│   │   ├── image_preprocessor.py   # 이미지 전처리
│   │   └── ingredient_parser.py    # MFDS DB 성분 필드 파싱
│   ├── matching/
│   │   ├── mfds_client.py          # FULLTEXT/LIKE 검색 + RapidFuzz 재랭킹
│   │   └── matcher.py              # 유사도 기반 최종 매칭 + 성분 파싱 결합
│   ├── enrichment/
│   │   ├── ingredient_lookup.py    # DB 미등재 제품 Gemini 성분 추출
│   │   └── official_image_lookup.py # 공식 제품 이미지 URL 보강
│   ├── schema/
│   │   └── result.py               # 응답 스키마 (Pydantic)
│   └── pipeline.py                 # 전체 파이프라인 오케스트레이션
└── scripts/
    └── evaluate_accuracy.py        # 정확도 평가 스크립트
```

---

## 파이프라인 단계

### 1단계 — 이미지 전처리 (`image_preprocessor.py`)

| 처리 | 내용 |
|---|---|
| auto-crop | Canny 엣지 검출로 제품 영역 추출. 면적 < 원본 20%이면 건너뜀 |
| 리사이즈 | 짧은 변 512px 이상, 긴 변 1024px 이하 |
| 노이즈 제거 | GaussianBlur (radius 0.5) |
| 대비·선명도 강화 | 대비 1.3×, 선명도 2.0×, 밝기 1.05× |

전처리 결과는 파이프라인 완료 후 자동 삭제됩니다. 전처리 실패 시 원본으로 진행합니다.

---

### 2단계 — Gemini Vision 제품명 후보 추출 (`llm_extractor.py`)

#### API 우선순위

| 순위 | API | 모델 | 환경 변수 |
|---|---|---|---|
| 1순위 | 충북대 AI Gateway | `gemini-3.5-flash` | `CBNUAI_API_KEY` |
| 2순위 (fallback) | Google Gemini API | `gemini-2.0-flash` | `GEMINI_API_KEY` |

#### 프롬프트 전략

제품명 후보를 최대 3개 반환합니다.

```
- 후보 1: 브랜드명(회사명) 포함 전체 제품명
- 후보 2: 브랜드명 제외 제품 고유명
- 후보 3: 다른 표기 방식 (없으면 생략)
- 용량, 정수, 광고 문구 제외
- 한글로 반환
```

예시:
```
1. 고려은단 비타민C 1000
2. 비타민C 1000
3. 고려은단 비타민씨 1000
```

---

### 3단계 — MFDS DB 매칭 (`mfds_client.py`, `matcher.py`)

후보 3개를 전부 DB에서 조회한 뒤 confidence가 가장 높은 결과를 선택합니다.  
(이전 방식처럼 첫 후보에서 early break하지 않음)

#### 쿼리 정규화

검색 전 제품명을 정규화합니다.

```python
# 단일 알파벳·숫자가 앞 토큰과 공백으로 분리된 경우 붙임
"메가도스 B" → "메가도스B"
"비타민 C 1000" → "비타민C 1000"  # 단일 토큰만 대상
```

DB에 브랜드 없이 저장된 제품명(`메가도스B`)과 Gemini가 추출한 이름(`메가도스 B`) 간의 공백 차이를 해소합니다.

#### 검색 방식

1. **FULLTEXT 검색** — MySQL ngram parser 기반 후보 검색
2. **Fallback** — FULLTEXT 결과 없을 때 LIKE 토큰 분리 검색  
   (`제품명 전체`, `공백·괄호로 분리된 2자 이상 토큰` 순서로 OR 조건)
3. **RapidFuzz 재랭킹**

```
score = token_set_ratio * 0.6 + partial_ratio * 0.4
      × (0.7 + 0.3 * 길이비율)
```

- `token_set_ratio`: 쿼리 토큰이 DB명에 포함되는지 (순서 무관)
- `partial_ratio`: 순서 있는 부분 문자열 매칭
- 길이비율 패널티: 쿼리가 짧을수록 긴 DB 제품명에 무분별하게 걸리는 것 방지

4. **브랜드 보정** — Gemini 후보1(브랜드 포함)과 후보2(브랜드 제외)의 차이로 브랜드명 추출,  
   DB 제품명에 해당 브랜드가 포함되면 점수 ×1.05 보너스 적용

5. **임계값** — 유사도 70 미만이면 미매칭 처리

#### 동명이제(같은 이름, 다른 제조사) 처리

DB 매칭 완료 후 `search_top_products()`로 유사도 상위 제품을 추가 조회합니다.

- **제조사명 정규화**: 공백·괄호·(주)/(유) 제거 후 소문자 비교  
  예) `고려은단 헬스케어(주)` = `고려은단헬스케어(주)` → 동일 처리
- **동일 제조사**: product_code가 큰 값(최신 등록)으로 자동 교체
- **다른 제조사**: `candidates[]` 배열에 담아 프론트로 전달  
  → 프론트: "같은 이름의 다른 제품이 있습니다. 선택해주세요" UI 표시

---

### 4단계 — DB 미등재 제품 처리 (`ingredient_lookup.py`)

유사도 임계값을 넘는 DB 매칭이 없을 경우 Gemini에게 성분을 직접 요청합니다.

#### 프롬프트

```
'{product_name}' 건강기능식품의 식약처(MFDS) 등록 기능성 원료 성분명을 알려줘.
- 식약처 건강기능식품 공전에 등재된 기능성 원료명 기준으로 반환.
- 성분명만 콤마(,)로 구분해서 한 줄로 반환. 함량·단위·설명 없이.
- 모르거나 확인 불가하면 빈 문자열만 반환.
```

- 성분명은 **한국어**로 반환 (번역은 백엔드에서 처리)
- `ingredients_source: "gemini"` 로 마킹해 프론트에서 출처 구분 가능
- `warnings[]`에 "Gemini 추출 성분, 반드시 확인하세요" 메시지 포함
- `status: needs_confirmation` 으로 반환

---

### 5단계 — DB 성분 파싱 (`ingredient_parser.py`)

DB 등재 제품의 경우 `supplement_product_markers` 테이블에서 사전 파싱된 성분명을 조회합니다.  
없을 경우 `main_fnctn`(기능성 내용)과 `base_standard`(기준·규격) 필드를 규칙 기반으로 파싱합니다.

---

## 응답 스키마

```python
class SupplementProduct(BaseModel):
    product_code: Optional[str]           # 식약처 제품 코드 (미등재이면 None)
    product_name: str                     # 제품명
    manufacturer: Optional[str]           # 제조사
    product_image_url: Optional[str]      # 공식 제품 이미지 URL
    product_image_source_url: Optional[str]
    main_function: Optional[str]          # 기능성 내용 원문
    base_standard: Optional[str]          # 기준·규격 원문
    ingredients: list[str]               # 성분명 리스트 (한국어)
    ingredients_source: str              # "db" | "gemini"
    confidence: float                    # 매칭 유사도 (0.0~1.0)

class SupplementRecognitionResult(BaseModel):
    request_id: str
    status: str                          # completed | needs_confirmation | failed
    product: Optional[SupplementProduct]
    candidates: list[SupplementProduct]  # 동명이제 후보 목록
    needs_confirmation: bool
    warnings: list[str]
    error_code: Optional[str]
    error_detail: Optional[str]
```

### status별 프론트 처리

| status | candidates | 상황 | 프론트 처리 |
|---|---|---|---|
| `completed` | 빈 배열 | 정상 매칭 | 제품 정보 표시 |
| `completed` | 항목 있음 | 동명이제 존재 | "같은 이름의 다른 제품 있음" 선택 UI |
| `needs_confirmation` | — | DB 미등재 또는 낮은 신뢰도 | warnings 표시, 사용자 확인 요청 |
| `failed` | — | 이미지 처리 실패 | 에러 메시지 표시 |

---

## 환경 변수

| 변수 | 설명 | 필수 |
|---|---|---|
| `CBNUAI_API_KEY` | 충북대 AI Gateway API 키 | 1순위 (선택) |
| `GEMINI_API_KEY` | Google Gemini API 키 | fallback (둘 중 하나 필수) |
| `MYSQL_HOST` | MySQL 호스트 | 필수 |
| `MYSQL_PORT` | MySQL 포트 | 필수 |
| `MYSQL_DATABASE` | 데이터베이스명 (`click_db`) | 필수 |
| `MYSQL_USER` | DB 사용자 | 필수 |
| `MYSQL_PASSWORD` | DB 비밀번호 | 필수 |

---

## 정확도

측정 기준: `supplement_recognition/data/samples/` 내 30장, 파일명 = 정답 제품명, `partial_ratio ≥ 70%` 시 정확 매칭 판정

| 지표 | 결과 |
|---|---|
| Gemini 제품명 추출 성공률 | 30/30 = **100%** |
| Gemini 제품명 유사도 (추출명 vs 정답) | **89.5%** |
| DB 정확 매칭률 | 24/30 = **80.0%** |

상세 분석: [`docs/accuracy-report.md`](../docs/accuracy-report.md)

---

## 한계점

- **DB 미등재 제품**: 식약처 DB에 등록되지 않은 신제품·수입품은 Gemini 성분 추출로 대응하지만 정확성은 보장되지 않음 — 반드시 사용자 확인 필요
- **Gemini 환각**: Gemini가 존재하지 않는 성분명을 반환할 수 있음 (`ingredients_source: "gemini"` 로 구분 가능)
- **규칙 기반 성분 파싱**: `base_standard` 형식이 제품마다 달라 예외 케이스 발생 가능
