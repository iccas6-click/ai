-- ============================================================
-- CLICK AI DB (click_db)
-- 건기식 인식 + 알약 제품 조회용
-- drug-supplement schema v3 기준
-- ============================================================

-- 건기식 제품 마스터 (식약처 MFDS, 44,885건) -----------------

CREATE TABLE IF NOT EXISTS supplement_info (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    sttemnt_no    VARCHAR(30)   NOT NULL UNIQUE COMMENT '품목제조신고번호',
    prduct        VARCHAR(200)  NOT NULL COMMENT '제품명',
    entrps        VARCHAR(200)  COMMENT '업체명',
    regist_dt     VARCHAR(10)   COMMENT '등록일자',
    distb_pd      VARCHAR(100)  COMMENT '유통기한',
    sungsang      TEXT          COMMENT '성상',
    srv_use       TEXT          COMMENT '섭취량 및 섭취방법',
    prsrv_pd      VARCHAR(200)  COMMENT '보존기준',
    intake_hint1  TEXT          COMMENT '섭취 시 주의사항',
    main_fnctn    TEXT          COMMENT '주요기능',
    base_standard TEXT          COMMENT '기준규격',
    product_image_url TEXT      COMMENT '공식 제품 이미지 URL',
    product_image_source_url TEXT COMMENT '공식 제품 이미지 출처 URL',
    product_image_checked_at TIMESTAMP NULL COMMENT '공식 이미지 확인 시각',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_prduct (prduct)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 건기식 제품 ↔ supplement_entities 연결 마커 ----------------
-- marker_source_column 값은 supplement_info의 실제 컬럼명 기준
-- (prduct, main_fnctn, base_standard, intake_hint1)

CREATE TABLE IF NOT EXISTS supplement_product_markers (
    marker_id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    supplement_info_id   INT          NOT NULL,
    marker_text          VARCHAR(500) NOT NULL,
    marker_text_normalized VARCHAR(500) NOT NULL,
    marker_source_column VARCHAR(50)  COMMENT 'supplement_info 컬럼명 (prduct/main_fnctn/base_standard/intake_hint1)',
    marker_type          VARCHAR(50),
    supplement_id        VARCHAR(20)  NOT NULL COMMENT 'supplement_entities.supplement_id (백엔드 DB)',
    mapping_status       VARCHAR(30)  NOT NULL DEFAULT 'confirmed',
    FOREIGN KEY (supplement_info_id) REFERENCES supplement_info(id),
    KEY idx_marker_supplement_id (supplement_id),
    KEY idx_marker_info_id (supplement_info_id),
    KEY idx_marker_mapping_status (mapping_status)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 알약 제품 마스터 (AIHub 기반, 46,836건) --------------------

CREATE TABLE IF NOT EXISTS pill_products (
    pill_product_id       VARCHAR(64)  NOT NULL PRIMARY KEY,
    product_name          VARCHAR(255) NOT NULL,
    product_name_normalized VARCHAR(255) NOT NULL,
    UNIQUE KEY uq_product_name_normalized (product_name_normalized),
    KEY idx_product_name (product_name)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 알약 제품 ↔ canonical drug 연결 (26,362건) -----------------
-- canonical_drug_id는 백엔드 DB canonical_drug_entities 참조
-- (cross-DB FK 불가 → VARCHAR 저장, 애플리케이션에서 일관성 보장)

CREATE TABLE IF NOT EXISTS pill_product_ingredients (
    id                       BIGINT AUTO_INCREMENT PRIMARY KEY,
    pill_product_id          VARCHAR(64)  NOT NULL,
    ingredient_name          VARCHAR(255) NOT NULL,
    ingredient_name_normalized VARCHAR(255) NOT NULL,
    canonical_drug_id        VARCHAR(64)  NOT NULL COMMENT '백엔드 DB canonical_drug_entities.canonical_drug_id',
    UNIQUE KEY uq_pill_ingredient (pill_product_id, ingredient_name_normalized),
    KEY idx_ppi_product_id (pill_product_id),
    KEY idx_ppi_canonical_drug_id (canonical_drug_id),
    FOREIGN KEY (pill_product_id) REFERENCES pill_products(pill_product_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
