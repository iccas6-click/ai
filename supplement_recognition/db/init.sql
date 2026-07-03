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
