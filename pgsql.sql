-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- LangChain creates this table, but understanding the schema helps:
CREATE TABLE langchain_pg_embedding (
    id          UUID PRIMARY KEY,
    collection_id UUID,
    embedding   vector(1536),       -- dimension matches your model
    document    TEXT,               -- original chunk text
    cmetadata   JSONB               -- source, page, custom metadata
);

-- Index for fast ANN search
CREATE INDEX ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
-- OR for better recall:
-- USING hnsw (embedding vector_cosine_ops);