# Core
pip install langchain langchain-community langchain-core
pip install langchain-openai          # or langchain-anthropic / langchain-huggingface

# Vector store
pip install langchain-postgres        # preferred (newer)
# OR
pip install psycopg2-binary pgvector  # lower-level

# Document loaders (pick what you need)
pip install pypdf unstructured docx2txt

# Optional but common
pip install langchain-text-splitters
pip install tiktoken                  # token counting for OpenAI