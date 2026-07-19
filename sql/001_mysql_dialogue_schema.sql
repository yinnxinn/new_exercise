CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    job_name VARCHAR(128) NOT NULL,
    source_path TEXT NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    total_rows INT NOT NULL DEFAULT 0,
    inserted_rows INT NOT NULL DEFAULT 0,
    skipped_rows INT NOT NULL DEFAULT 0,
    duplicate_rows INT NOT NULL DEFAULT 0,
    error_rows INT NOT NULL DEFAULT 0,
    config JSON NULL,
    stats JSON NULL,
    started_at DATETIME NULL,
    finished_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_ingestion_jobs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS medical_dialogues (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    external_id VARCHAR(128) NOT NULL UNIQUE,
    source_dataset VARCHAR(128) NOT NULL,
    source_category VARCHAR(128) NULL,
    source_file VARCHAR(512) NULL,
    source_row_number INT NULL,
    department VARCHAR(128) NOT NULL,
    title TEXT NOT NULL,
    question MEDIUMTEXT NOT NULL,
    answer MEDIUMTEXT NOT NULL,
    question_length INT NOT NULL DEFAULT 0,
    answer_length INT NOT NULL DEFAULT 0,
    content_hash CHAR(64) NOT NULL UNIQUE,
    quality_status VARCHAR(32) NOT NULL DEFAULT 'active',
    quality_flags JSON NULL,
    raw_payload JSON NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_dialogues_department (department),
    KEY idx_dialogues_source_category (source_category),
    KEY idx_dialogues_quality_status (quality_status),
    KEY idx_dialogues_content_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS medical_documents (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    external_id VARCHAR(128) NOT NULL UNIQUE,
    dialogue_id BIGINT NULL,
    title TEXT NOT NULL,
    department VARCHAR(128) NOT NULL,
    question MEDIUMTEXT NULL,
    answer MEDIUMTEXT NULL,
    content MEDIUMTEXT NOT NULL,
    source_type VARCHAR(64) NOT NULL DEFAULT 'dialogue',
    source_dataset VARCHAR(128) NULL,
    source_ref VARCHAR(512) NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    metadata JSON NULL,
    tag_codes_json JSON NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_documents_dialogue
        FOREIGN KEY (dialogue_id) REFERENCES medical_dialogues(id)
        ON DELETE CASCADE,
    KEY idx_documents_external_id (external_id),
    KEY idx_documents_dialogue_id (dialogue_id),
    KEY idx_documents_department (department),
    KEY idx_documents_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS medical_tags (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tag_code VARCHAR(255) NOT NULL UNIQUE,
    tag_name VARCHAR(255) NOT NULL,
    tag_type VARCHAR(64) NOT NULL,
    normalized_name VARCHAR(255) NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'auto',
    metadata JSON NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_tags_type (tag_type),
    KEY idx_tags_name (tag_name),
    KEY idx_tags_code_type (tag_code, tag_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS medical_document_tags (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id BIGINT NOT NULL,
    tag_id BIGINT NOT NULL,
    tag_code VARCHAR(255) NOT NULL,
    tag_type VARCHAR(64) NOT NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'auto',
    confidence DECIMAL(5,4) NOT NULL DEFAULT 1.0000,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_document_tags_document
        FOREIGN KEY (document_id) REFERENCES medical_documents(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_document_tags_tag
        FOREIGN KEY (tag_id) REFERENCES medical_tags(id)
        ON DELETE CASCADE,
    UNIQUE KEY uk_document_tags_document_code (document_id, tag_code),
    KEY idx_document_tags_tag_id (tag_id),
    KEY idx_document_tags_code (tag_code),
    KEY idx_document_tags_type (tag_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS document_embedding_jobs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id BIGINT NOT NULL,
    milvus_collection VARCHAR(128) NOT NULL,
    embedding_model VARCHAR(128) NOT NULL,
    embedding_dim INT NOT NULL,
    text_hash CHAR(64) NOT NULL,
    vector_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    milvus_primary_key BIGINT NULL,
    error_message TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_embedding_jobs_document
        FOREIGN KEY (document_id) REFERENCES medical_documents(id)
        ON DELETE CASCADE,
    UNIQUE KEY uk_embedding_jobs_document_model (document_id, embedding_model),
    KEY idx_embedding_jobs_status (vector_status),
    KEY idx_embedding_jobs_collection (milvus_collection)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS retrieval_eval_queries (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    query_text TEXT NOT NULL,
    gold_document_id BIGINT NULL,
    gold_dialogue_id BIGINT NULL,
    gold_department VARCHAR(128) NULL,
    eval_split VARCHAR(32) NOT NULL DEFAULT 'dev',
    query_type VARCHAR(64) NOT NULL DEFAULT 'original_question',
    metadata JSON NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_eval_queries_document
        FOREIGN KEY (gold_document_id) REFERENCES medical_documents(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_eval_queries_dialogue
        FOREIGN KEY (gold_dialogue_id) REFERENCES medical_dialogues(id)
        ON DELETE SET NULL,
    KEY idx_eval_queries_split (eval_split),
    KEY idx_eval_queries_department (gold_department)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
