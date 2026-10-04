from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CONNECTION_STRING = "postgresql+psycopg://raguser:ragpass@localhost:5432/ragdb"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=CONNECTION_STRING,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr" for diversity
    search_kwargs={"k": 5}
)


def index_documents():
    # 1. Load
    loader = PyPDFLoader("Admit Card.pdf")
    docs = loader.load()

    # 2. Split (chunk)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    # 3. Index your chunks
    vectorstore.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks")


# Run `python rag.py` once to index; importing this module (e.g. from chain.py) only gives you the retriever
if __name__ == "__main__":
    index_documents()
