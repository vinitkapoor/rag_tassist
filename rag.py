import argparse
import re
import uuid
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CONNECTION_STRING = "postgresql+psycopg://raguser:ragpass@localhost:5432/ragdb"

# Each dataset lives in its own pgvector collection so they can be queried separately or together
COLLECTIONS = {
    "jira": "jira_tickets",
    "sop": "sop_docs",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)


def get_vectorstore(dataset, reset=False):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTIONS[dataset],
        connection=CONNECTION_STRING,
        pre_delete_collection=reset,   # True wipes the collection before re-indexing
    )


def get_retriever(dataset="all", k=5):
    """dataset: "jira", "sop" or "all" (searches both and merges results)."""
    if dataset != "all":
        return get_vectorstore(dataset).as_retriever(search_kwargs={"k": k})

    retrievers = [get_vectorstore(d).as_retriever(search_kwargs={"k": k}) for d in COLLECTIONS]
    return RunnableLambda(lambda q: [doc for r in retrievers for doc in r.invoke(q)])


# ---------- Jira tickets (.xls / .xlsx export) ----------

# Column names as they appear in a standard Jira Excel export; missing columns are skipped
JIRA_KEY_COLUMN = "Issue Key"
JIRA_TEXT_COLUMNS = ["Description", "Environment", "Resolution"]   # Summary goes in the header
JIRA_COMMENT_COLUMN = "Comment"   # Jira exports one "Comment" column per comment
JIRA_METADATA_COLUMNS = {
    "Issue Type": "issue_type",
    "Status": "status",
    "Priority": "priority",
    "Assignee": "assignee",
    "Reporter": "reporter",
    "Created": "created",
    "Resolved": "resolved",
    "Labels": "labels",
    "Component/s": "components",
}


def _columns_named(df, name):
    # pandas renames duplicate headers to "Comment", "Comment.1", "Comment.2", ...
    return [c for c in df.columns if c == name or re.fullmatch(rf"{re.escape(name)}\.\d+", c)]


def _clean_comment(text):
    # Jira exports comments as "date;author;body" - keep the author and body
    parts = text.split(";", 2)
    return f"{parts[1]}: {parts[2]}" if len(parts) == 3 else text


def load_jira(path):
    df = pd.read_excel(path, dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    if JIRA_KEY_COLUMN not in df.columns:
        raise ValueError(f"'{JIRA_KEY_COLUMN}' column not found. Columns: {list(df.columns)}")

    comment_columns = _columns_named(df, JIRA_COMMENT_COLUMN)
    docs = []
    for _, row in df.iterrows():
        key = row[JIRA_KEY_COLUMN].strip()
        if not key:
            continue

        header = f"Ticket {key}: {row.get('Summary', '')}".strip()
        body = [f"{col}: {row[col]}" for col in JIRA_TEXT_COLUMNS if col in df.columns and row[col].strip()]
        comments = [_clean_comment(row[c].strip()) for c in comment_columns if row[c].strip()]
        if comments:
            body.append("Comments:\n" + "\n".join(f"- {c}" for c in comments))

        metadata = {"dataset": "jira", "source": str(path), "ticket": key}
        metadata.update({meta: row[col] for col, meta in JIRA_METADATA_COLUMNS.items()
                         if col in df.columns and row[col]})

        # Long tickets get split; every chunk keeps the ticket header so it stays identifiable
        for i, chunk in enumerate(splitter.split_text("\n\n".join(body)) or [""]):
            docs.append(Document(
                page_content=f"{header}\n\n{chunk}".strip(),
                metadata={**metadata, "chunk": i},
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"jira:{key}:{i}")),
            ))
    return docs


# ---------- SOP documents (folder of PDF / DOCX / MD / TXT) ----------

SOP_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def load_sops(directory):
    docs = []
    for path in sorted(Path(directory).rglob("*")):
        loader_cls = SOP_LOADERS.get(path.suffix.lower())
        if not loader_cls:
            continue
        pages = loader_cls(str(path)).load()
        for i, chunk in enumerate(splitter.split_documents(pages)):
            chunk.metadata.update({"dataset": "sop", "source": str(path), "title": path.stem, "chunk": i})
            chunk.id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sop:{path}:{i}"))
            docs.append(chunk)
    return docs


LOADERS = {"jira": load_jira, "sop": load_sops}


def index_documents(dataset, path, reset=False):
    docs = LOADERS[dataset](path)
    # Stable ids mean re-running upserts instead of duplicating rows
    get_vectorstore(dataset, reset=reset).add_documents(docs, ids=[d.id for d in docs])
    print(f"Indexed {len(docs)} chunks into '{COLLECTIONS[dataset]}'")


# Run once per dataset to index, e.g.
#   python rag.py jira tickets.xlsx
#   python rag.py sop ./sops --reset
# Importing this module (e.g. from chain.py) only gives you the retrievers
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a dataset into pgvector")
    parser.add_argument("dataset", choices=COLLECTIONS.keys())
    parser.add_argument("path", help="Jira .xls/.xlsx export, or a folder of SOP documents")
    parser.add_argument("--reset", action="store_true", help="delete the collection before indexing")
    args = parser.parse_args()
    index_documents(args.dataset, args.path, reset=args.reset)
